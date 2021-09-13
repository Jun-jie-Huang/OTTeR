# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for question-answering on SQuAD (DistilBERT, Bert, XLM, XLNet)."""

import argparse
import glob
import logging
import os
import re
import collections
import random
import timeit
import json
import numpy as np
from datetime import datetime
from tqdm import tqdm, trange
from multiprocessing import Pool, cpu_count
from functools import partial
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, BertConfig, BertTokenizer,
                          BertForQuestionAnswering, get_linear_schedule_with_warmup,
                          XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer,
                          RobertaConfig, RobertaForQuestionAnswering, RobertaTokenizer,
                          ElectraConfig, ElectraForQuestionAnswering, ElectraTokenizer,
                          AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer,
                          LongformerConfig, LongformerForQuestionAnswering, LongformerTokenizer,
                          BigBirdConfig, BigBirdForQuestionAnswering, BigBirdTokenizer)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

import sys
sys.path.append('../')

# from transformers.data.metrics.squad_metrics import compute_predictions_logits
from qa.utils import compute_predictions_logits

# from utils import readGZip
from qa.qa_dataset import prepare_qa_data, OTTQAProcessor, OTTQAFeatures, ottqa_convert_examples_to_features, \
    load_and_cache_examples, get_test_data
from qa.config import qa_args
from qa.utils import normalize_answer, compute_f1, compute_exact

logger = logging.getLogger(__name__)
args = qa_args()

# ALL_MODELS = sum(
#     (
#         tuple(conf.pretrained_config_archive_map.keys())
#         for conf in (BertConfig, )
#     ),
#     (),
# )

# MODEL_CLASSES = {"bert": (BertConfig, BertForQuestionAnswering, BertTokenizer)}
MODEL_CLASSES = {
    "bert": (BertConfig, BertForQuestionAnswering, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer),
    "roberta": (RobertaConfig, RobertaForQuestionAnswering, RobertaTokenizer),
    "electra": (ElectraConfig, ElectraForQuestionAnswering, ElectraTokenizer),
    "albert": (AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer),
    'longformer': (LongformerConfig, LongformerForQuestionAnswering, LongformerTokenizer),
    'bigbird': (BigBirdConfig, BigBirdForQuestionAnswering, BigBirdTokenizer)
}


class SquadResult(object):
    """
    Constructs a SquadResult which can be used to evaluate a model's output on the SQuAD dataset.
    Args:
        unique_id: The unique identifier corresponding to that example.
        start_logits: The logits corresponding to the start of the answer
        end_logits: The logits corresponding to the end of the answer
    """

    def __init__(self, unique_id, start_logits, end_logits, start_top_index=None, end_top_index=None, cls_logits=None):
        self.start_logits = start_logits
        self.end_logits = end_logits
        self.unique_id = unique_id

        if start_top_index:
            self.start_top_index = start_top_index
            self.end_top_index = end_top_index
            self.cls_logits = cls_logits


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def train(args, train_dataset, model, tokenizer, dev_dataset=None, dev_examples=None, dev_features=None):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir=args.output_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(epochs_trained, int(args.num_train_epochs))
    # Added here for reproductibility
    set_seed(args)
    best_f1 = 0.0
    for epoch in train_iterator:
        tr_loss=0.0
        epoch_iterator = tqdm(train_dataloader)
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }

            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            avg_loss = round(tr_loss * args.gradient_accumulation_steps / (step + 1), 4)
            epoch_iterator.set_description("epoch {} step {} loss {}".format(epoch + 1, step + 1, avg_loss))
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

        # Save model checkpoint
        if args.local_rank in [-1, 0]:
            if args.local_rank == -1 and args.do_eval and args.evaluate_during_training:
                predictions, nbests = evaluate_simplified(dev_dataset, dev_examples, dev_features, args, model, tokenizer, prefix='dev')
                # results = get_raw_scores(dev_examples, nbests)
                results = get_raw_scores_finer(dev_examples, nbests)
                logger.info("***** Eval Results (running) *****")
                for key, value in results.items():
                    tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    logger.info("  %s = %s", key, round(value, 4))
                # if results['F1'] > best_f1:
                #     best_f1 = results['F1']
                if results['F1-table'] > best_f1:
                    best_f1 = results['F1-table']
                    output_dir = os.path.join(args.output_dir, "checkpoint-best")
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                        # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate_simplified(dataset, examples, features, args, model, tokenizer, prefix=""):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_results = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }
            example_indices = batch[3]
            outputs = model(**inputs)

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)

            start_logits = to_list(outputs[0][i])
            end_logits = to_list(outputs[1][i])
            result = SquadResult(unique_id, start_logits, end_logits)
            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    # logger.info("  Evaluation done in total %f secs (%f sec per example) for %d examples", evalTime, evalTime / len(dataset), len(all_results))

    # Compute predictions
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))
    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None
    predictions, nbests = compute_predictions_logits(
        examples,
        features,
        all_results,
        args.n_best_size,
        args.max_answer_length,
        args.do_lower_case,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        args.verbose_logging,
        args.version_2_with_negative,
        args.null_score_diff_threshold,
        tokenizer,
    )

    return predictions, nbests


def test(args, model, tokenizer, prefix=""):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    test_dataset, test_examples, test_features, qid2idx = get_test_data(args, tokenizer)
    # Note that DistributedSampler samples randomly
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test {} *****".format(prefix))
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_results = []
    start_time = timeit.default_timer()

    for batch in tqdm(test_dataloader, desc="Evaluating Test: "):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }
            example_indices = batch[3]
            outputs = model(**inputs)  #

        for i, example_index in enumerate(example_indices):
            test_feature = test_features[example_index.item()]
            unique_id = int(test_feature.unique_id)

            start_logits = to_list(outputs[0][i])
            end_logits = to_list(outputs[1][i])
            result = SquadResult(unique_id, start_logits, end_logits)
            all_results.append(result)

    testTime = timeit.default_timer() - start_time
    # logger.info("  Evaluation done in total %f secs (%f sec per example) for %d examples", testTime, testTime / len(test_dataset), len(all_results))

    # Compute predictions
    output_prediction_file = os.path.join(args.output_dir, "predictions_test.json")
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_test.json")

    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_test.json")
    else:
        output_null_log_odds_file = None

    predictions, nbests = compute_predictions_logits(
        test_examples,
        test_features,
        all_results,
        args.n_best_size,
        args.max_answer_length,
        args.do_lower_case,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        args.verbose_logging,
        args.version_2_with_negative,
        args.null_score_diff_threshold,
        tokenizer,
    )
    # data = {step: {} for step in qid2idx.values()}
    # for k, step in qid2idx.items():
    #     data[step]['pred'] = predictions.get(k, 'None')
    data = []
    for k, step in qid2idx.items():
        data.append({"question_id": k, "pred": predictions.get(k, 'None')})

    return data


def get_raw_scores(examples, preds):
    """
    Computes the exact and f1 scores from the examples and the model predictions
    """
    exact_scores = {}
    f1_scores = {}
    logger.info("length of examples: {}, length of preds: {}".format(len(examples), len(preds)))
    # logger.info(examples[0].qas_id)

    for example in examples:
        qas_id = example.qas_id
        gold_answers = [answer["text"] for answer in example.answers if normalize_answer(answer["text"])]
        logger.info(gold_answers)
        logger.info(len(gold_answers))

        if not gold_answers:
            # For unanswerable questions, only correct answer is empty string
            gold_answers = [""]

        if qas_id not in preds:
            logger.info("Missing prediction for %s" % qas_id)
            continue

        prediction = preds[qas_id]
        exact_scores[qas_id] = max(compute_exact(a, prediction) for a in gold_answers)
        f1_scores[qas_id] = max(compute_f1(a, prediction) for a in gold_answers)

    qid_list = [k for k in exact_scores]
    total = len(qid_list)

    return collections.OrderedDict(
        [
            ("EM", 100.0 * sum(exact_scores[k] for k in qid_list) / total),
            ("F1", 100.0 * sum(f1_scores[k] for k in qid_list) / total),
            ("Total", total),
        ]
    )


def get_raw_scores_finer(examples, preds):
    """
    Computes the exact and f1 scores from the examples and the model predictions
    """
    exact_scores_tblock = {}
    f1_scores_tblock = {}
    logger.info("length of examples: {}, length of preds: {}".format(len(examples), len(preds)))
    # logger.info(examples[0].qas_id)

    table_answers = {}
    table_preds = {}
    for example in examples:
        qas_id = example.qas_id
        gold_answers = [answer["text"] for answer in example.answers if normalize_answer(answer["text"])]

        if not gold_answers:
            # For unanswerable questions, only correct answer is empty string
            gold_answers = [""]

        if qas_id not in preds:
            logger.info("Missing prediction for %s" % qas_id)
            continue

        prediction = preds[qas_id][0]['text']
        # prediction = normalize_answer(preds[qas_id][0]['text'])
        exact_scores_tblock[qas_id] = max(compute_exact(a, prediction) for a in gold_answers)
        f1_scores_tblock[qas_id] = max(compute_f1(a, prediction) for a in gold_answers)

        # Select table predictions from highest scored tblock predictions
        qas_id_table = qas_id.split('-')[0]
        table_answers[qas_id_table] = gold_answers
        table_preds[qas_id_table] = [(prediction, preds[qas_id][0]['abs_score'])] if qas_id_table not in table_preds \
            else table_preds[qas_id_table] + [(prediction, preds[qas_id][0]['abs_score'])]

    exact_score_table = {}
    f1_scores_table = {}
    for qas_id_table, answers in table_answers.items():
        prediction = sorted(table_preds[qas_id_table], key=lambda x: x[1], reverse=True)[0][0]
        exact_score_table[qas_id_table] = max(compute_exact(a, prediction) for a in answers)
        f1_scores_table[qas_id_table] = max(compute_f1(a, prediction) for a in answers)


    qid_list_tblock = [k for k in exact_scores_tblock]
    total_tblock = len(qid_list_tblock)
    qid_list_table = [k for k in exact_score_table]
    total_able = len(qid_list_table)

    return collections.OrderedDict(
        [
            ("EM-tblock", 100.0 * sum(exact_scores_tblock[k] for k in qid_list_tblock) / total_tblock),
            ("F1-tblock", 100.0 * sum(f1_scores_tblock[k] for k in qid_list_tblock) / total_tblock),
            ("Total-tblock", total_tblock),
            ("EM-table", 100.0 * sum(exact_score_table[k] for k in qid_list_table) / total_able),
            ("F1-table", 100.0 * sum(f1_scores_table[k] for k in qid_list_table) / total_able),
            ("Total-table", total_able),
        ]
    )


def main():
    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )
    assert (args.local_rank == -1)
    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda")
    args.n_gpu = torch.cuda.device_count()

    args.device = device
    # args.output_dir = os.path.join(args.output_dir, datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    # Set seed
    set_seed(args)
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    if args.add_special_tokens:
        if args.num_tokenizer_vocab == 50271:
            special_tokens_dict = {'additional_special_tokens': ["[HEADER]", "[PASSAGE]", "[TB]","[DATA]","[TITLE]","[SECTITLE]"]}
        elif args.num_tokenizer_vocab == 50272:
            special_tokens_dict = {'additional_special_tokens': ["[HEADER]", "[PASSAGE]", "[SEP]", "[TB]", "[DATA]", "[TITLE]", "[SECTITLE]"]}
        else:
            special_tokens_dict = {'additional_special_tokens': ["[HEADER]", "[PASSAGE]", "[TB]", "[DATA]", "[TITLE]", "[SECTITLE]"]}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))
    # model.load_state_dict(torch.load(args.model_name_or_path))

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)
        if args.do_eval and args.evaluate_during_training:
            dev_dataset, dev_examples, dev_features = load_and_cache_examples(args, tokenizer, output_examples=True,
                                                                              evaluate=True)
        else:
            dev_dataset, dev_examples, dev_features = None, None, None
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, dev_dataset, dev_examples, dev_features)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory    
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoint_prefix = 'checkpoint-best'
        output_dir = os.path.join(args.output_dir, checkpoint_prefix)
        logger.info('loading model from {}'.format(os.path.join(output_dir, 'pytorch_model.bin')))
        model.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin')))
        model.to(args.device)

        dev_dataset, dev_examples, dev_features = load_and_cache_examples(args, tokenizer, output_examples=True,
                                                                          evaluate=True)
        predictions, nbests = evaluate_simplified(dev_dataset, dev_examples, dev_features, args, model, tokenizer, prefix='eval')
        results = get_raw_scores_finer(dev_examples, nbests)
        # results = get_raw_scores(dev_examples, nbests)

        logger.info("***** Eval results *****")
        for key in results.keys():
            logger.info("  Eval %s = %s", key, str(results[key]))
        logger.info("Eval Model From: {}".format(os.path.join(output_dir, 'pytorch_model.bin')))
        logger.info("***** Eval results *****")
        with open(os.path.join(args.output_dir, 'eval_results.txt'), 'a+') as f:
            f.write("checkpoint {}, results: {}".format(os.path.join(output_dir, 'pytorch_model.bin'), results))

    if args.do_predict and args.local_rank in [-1, 0]:
        logger.info("***** Testing results *****")
        checkpoint_prefix = 'checkpoint-best'
        if checkpoint_prefix not in args.output_dir and \
                os.path.exists(os.path.join(args.output_dir, checkpoint_prefix)):
            output_dir = os.path.join(args.output_dir, checkpoint_prefix)
        else:
            output_dir = args.output_dir
        if not args.pred_model_dir:
            model_path = os.path.join(output_dir, 'pytorch_model.bin')
        else:
            model_path = os.path.join(args.pred_model_dir, 'pytorch_model.bin')
        logger.info("Loading checkpoint %s for test", model_path)
        model.load_state_dict(torch.load(model_path))
        # tokenizer = tokenizer_class.from_pretrained(output_dir)
        model.to(args.device)

        # test_features, test_dataset, qid2idx = get_test_data(args)
        # prediction = evaluate_simplified(test_dataset, test_examples, test_features, args, model, tokenizer, prefix='test')
        # data = {step: {} for step in qid2idx.values()}
        # for k, step in qid2idx.items():
        #     data[step]['pred'] = prediction.get(k, 'None')
        data = test(args, model, tokenizer, prefix='test')

        with open(args.predict_output_file, 'w') as f:
            json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()
