# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.
"""
# DPR baseline shared encoder

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_1hop_tb_retrieval.py \
    --do_train \
    --prefix nq_dpr_shared \
    --predict_batch_size 5000 \
    --model_name bert-base-uncased \
    --train_batch_size 256 \
    --gradient_accumulation_steps 1 \
    --accumulate_gradients 1 \
    --learning_rate 2e-5 \
    --fp16 \
    --train_file /private/home/xwhan/data/nq-dpr/nq-with-neg-train.txt \
    --predict_file /private/home/xwhan/data/nq-dpr/nq-with-neg-dev.txt \
    --seed 16 \
    --eval-period -1 \
    --max_c_len 300 \
    --max_q_len 50 \
    --warmup-ratio 0.1 \
    --shared-encoder \
    --num_train_epochs 50

# WebQ single train
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_1hop_tb_retrieval.py \
    --do_train \
    --prefix wq_dpr_shared \
    --predict_batch_size 5000 \
    --model_name bert-base-uncased \
    --train_batch_size 256 \
    --gradient_accumulation_steps 1 \
    --accumulate_gradients 1 \
    --learning_rate 2e-5 \
    --fp16 \
    --train_file /private/home/xwhan/data/WebQ/wq-train-simplified.txt \
    --predict_file /private/home/xwhan/data/WebQ/wq-dev-simplified.txt \
    --seed 16 \
    --eval-period -1 \
    --max_c_len 300 \
    --max_q_len 50 \
    --warmup-ratio 0.1 \
    --shared-encoder \
    --num_train_epochs 50

# FEVER single-hop retrieval
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_1hop_tb_retrieval.py \
    --do_train \
    --prefix fever_single \
    --predict_batch_size 5000 \
    --model_name bert-base-uncased \
    --train_batch_size 256 \
    --gradient_accumulation_steps 1 \
    --accumulate_gradients 1 \
    --learning_rate 2e-5 \
    --fp16 \
    --train_file /private/home/xwhan/data/fever/retrieval/train_tfidf_neg.txt \
    --predict_file /private/home/xwhan/data/fever/retrieval/dev_tfidf_neg.txt \
    --seed 16 \
    --eval-period -1 \
    --max_c_len 400 \
    --max_q_len 45 \
    --shared-encoder \
    --num_train_epochs 40

# HotpotQA single-hop
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_1hop_tb_retrieval.py \
    --do_train \
    --prefix hotpot_single \
    --predict_batch_size 5000 \
    --model_name roberta-base \
    --train_batch_size 256 \
    --gradient_accumulation_steps 1 \
    --accumulate_gradients 1 \
    --learning_rate 2e-5 \
    --fp16 \
    --train_file /private/home/xwhan/data/hotpot/hotpot_train_with_neg_v0.json \
    --predict_file /private/home/xwhan/data/hotpot/hotpot_val_with_neg_v0.json \
    --seed 16 \
    --eval-period -1 \
    --max_c_len 300 \
    --max_q_len 70 \
    --shared-encoder \
    --warmup-ratio 0.1 \
    --num_train_epochs 50
    
"""
import logging
import os
import sys
import random
from tqdm import tqdm
from datetime import date
import numpy as np
import apex
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.optim import Adam
from functools import partial

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.append('../')

from transformers import AdamW, get_linear_schedule_with_warmup, TapasTokenizer
from transformers import (AutoConfig, AutoModel, AutoTokenizer,
                          BertConfig, BertModel, BertTokenizer,
                          XLNetConfig, XLNetModel, XLNetTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          ElectraConfig, ElectraModel, ElectraTokenizer,
                          AlbertConfig, AlbertModel, AlbertTokenizer,
                          LongformerConfig, LongformerModel, LongformerTokenizer)

from torch.utils.tensorboard import SummaryWriter
# from retrieval.data.sp_datasets import SPDataset,OTTDataset, sp_collate, NQMhopDataset, FeverSingleDataset
from retrieval.data.tr_dataset import TRDatasetNega, TRDatasetNegaMeta, tb_collate, tb_collate_tapas
from retrieval.data.tr_dataset import TRDatasetNegaMetaThreeCat
from retrieval.utils.utils import move_to_cuda, AverageMeter, load_saved
from retrieval.config import train_args
from retrieval.criterions import loss_single
# from retrieval.models.retriever import BertRetrieverSingle, RobertaRetrieverSingle, MomentumRetriever
from retrieval.models.retriever import TAPASRetrieverPure
from retrieval.models.retriever import SingleRetriever, RobertaSingleRetriever, MomentumRetriever, RobertaMomentumRetriever
from retrieval.models.tb_retriever import SingleRetrieverThreeCatPool, RobertaSingleRetrieverThreeCatPool
from retrieval.models.tb_retriever import SingleRetrieverThreeCatAtt, RobertaSingleRetrieverThreeCatAtt


def set_seed(seed=666):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    args = train_args()

    if args.fp16:
        apex.amp.register_half_function(torch, 'einsum')

    # date_curr = date.today().strftime("%m-%d-%Y")
    # model_name = f"addmeta-{args.prefix}-seed{args.seed}-bsz{args.train_batch_size}-fp16{args.fp16}-lr{args.learning_rate}-decay{args.weight_decay}-warm{args.warmup_ratio}-{args.model_name}"
    model_name = f"{args.prefix}-seed{args.seed}-bsz{args.train_batch_size}-fp16{args.fp16}-lr{args.learning_rate}-decay{args.weight_decay}-warm{args.warmup_ratio}-{args.model_name}"
    model_name += f"-k{args.k}" if args.momentum else ""
    # args.output_dir = os.path.join(args.output_dir, model_name)
    tb_logger = SummaryWriter(os.path.join(args.output_dir.replace("logs", "tflogs")))

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print(f"output directory {args.output_dir} already exists and is not empty.")
    os.makedirs(args.output_dir, exist_ok=True)
    os.system("touch {}".format(os.path.join(args.output_dir, model_name)))

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(args.output_dir, "log.txt")),
                                  logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(args)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        n_gpu = 1
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, n_gpu, bool(args.local_rank != -1), args.fp16)

    set_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    args.train_batch_size = int(args.train_batch_size / args.accumulate_gradients)
    if not args.do_train and not args.do_predict:
        raise ValueError(
            "At least one of `do_train` or `do_predict` must be True.")

    ################## MODEL SELECTION ####################################
    bert_config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    collate_fc = partial(tb_collate, pad_id=tokenizer.pad_token_id)
    if args.momentum and "roberta" not in args.model_name and "longformer" not in args.model_name:
        if args.three_cat:
            raise Exception("bert momentum dont have ThreeCat retriever setting...")
        else:
            model = MomentumRetriever(bert_config, args)
            logger.info("Model Using MomentumRetriever...")
    elif args.momentum and ("roberta" in args.model_name or "longformer" in args.model_name):
        if args.three_cat:
            raise Exception("roberta momentum dont have ThreeCat retriever setting...")
        else:
            model = RobertaMomentumRetriever(bert_config, args)
            logger.info("Model Using RobertaMomentumRetriever...")
    elif "roberta" in args.model_name or "longformer" in args.model_name:
        if args.three_cat:
            if args.part_pooling[:3] == 'att':
                model = RobertaSingleRetrieverThreeCatAtt(bert_config, args)
                logger.info("Model Using RobertaSingleRetrieverThreeCatPool...")
            else:
                model = RobertaSingleRetrieverThreeCatPool(bert_config, args)
                logger.info("Model Using RobertaSingleRetrieverThreeCatPool...")
        else:
            model = RobertaSingleRetriever(bert_config, args)
            logger.info("Model Using RobertaSingleRetriever...")
    elif "tapas" in args.model_name:
        if args.three_cat:
            raise Exception ("tapas dont have ThreeCat retriever setting...")
        else:
            tokenizer = TapasTokenizer.from_pretrained(args.model_name, cell_trim_length=args.cell_trim_length)
            collate_fc = partial(tb_collate_tapas, pad_id=tokenizer.pad_token_id)
            model = TAPASRetrieverPure(bert_config, args)
            logger.info("Model Using TAPASRetrieverPure...")
    else:
        if args.three_cat:
            if args.part_pooling[:3] == 'att':
                model = SingleRetrieverThreeCatAtt(bert_config, args)
                logger.info("Model Using SingleRetrieverThreeCatPool...")
            else:
                model = SingleRetrieverThreeCatPool(bert_config, args)
                logger.info("Model Using SingleRetrieverThreeCatPool...")
        else:
            model = SingleRetriever(bert_config, args)
            logger.info("Model Using SingleRetriever...")
    if args.add_special_tokens:
        special_tokens_dict = {'additional_special_tokens': ["[HEADER]", "[PASSAGE]","[SEP]", "[TB]","[DATA]","[TITLE]","[SECTITLE]"]}
        # special_tokens_dict = {'additional_special_tokens': ["[HEADER]", "[PASSAGE]", "[TB]","[DATA]","[TITLE]","[SECTITLE]"]}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))

    if args.do_train and args.max_c_len > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (args.max_c_len, bert_config.max_position_embeddings))

    ################## EVAL DATASET SELECTION, Try TRAIN!!!####################################
    if args.metadata:
        if args.three_cat:
            eval_dataset = TRDatasetNegaMetaThreeCat(tokenizer, args.predict_file, args)
            logger.info("Eval Dataset Using TRDatasetNegaMetaThreeCat...")
        else:
            eval_dataset = TRDatasetNegaMeta(tokenizer, args.predict_file, args)
            logger.info("Eval Dataset Using TRDatasetNegaMeta...")
    else:
        eval_dataset = TRDatasetNega(tokenizer, args.predict_file, args)
        logger.info("Eval Dataset Using TRDatasetNega...")
    eval_dataset.processing_data()

    eval_dataloader = DataLoader(eval_dataset,
                                 batch_size=args.predict_batch_size,
                                 collate_fn=collate_fc,
                                 pin_memory=True,
                                 num_workers=args.num_workers)
    logger.info(f"Num of dev batches: {len(eval_dataloader)}")

    if args.init_checkpoint != "":
        model = load_saved(model, args.init_checkpoint)

    model.to(device)
    # logger.info(model)
    logger.info(f"number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    if args.do_train:
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = Adam(optimizer_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

        if args.fp16:
            model, optimizer = apex.amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    else:
        if args.fp16:
            model = apex.amp.initialize(model, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    if args.do_train:
        # args.train_batch_size = args.per_gpu_train_batch_size * max(1, n_gpu)
        global_step = 0  # gradient update step
        batch_step = 0  # forward batch count
        best_mrr = 0
        train_loss_meter = AverageMeter()
        model.train()

        # train_dataset = TRDatasetNega(tokenizer, args.train_file, args, train=True)
        # train_dataset = TRDatasetNegaMeta(tokenizer, args.train_file, args, train=True)
        if args.metadata:
            if args.three_cat:
                train_dataset = TRDatasetNegaMetaThreeCat(tokenizer, args.train_file, args, train=True)
                logger.info("Training Dataset Using TRDatasetNegaMetaThreeCat...")
            else:
                train_dataset = TRDatasetNegaMeta(tokenizer, args.train_file, args, train=True)
                logger.info("Training Dataset Using TRDatasetNegaMeta...")
        else:
            train_dataset = TRDatasetNega(tokenizer, args.train_file, args, train=True)
            logger.info("Training Dataset Using TRDatasetNega...")
        train_dataset.processing_data()
        if args.data_augmentation:
            train_dataset.add_augment_data(args.augment_file)
        # train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
        # sampler = train_sampler,
        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, pin_memory=True,
                                      collate_fn=collate_fc, num_workers=args.num_workers, shuffle=True)
        logger.info("main: ")
        logger.info("length: dataset: {}, dataloader: {}".format(len(train_dataset), len(train_dataloader)))

        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        warmup_steps = t_total * args.warmup_ratio
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=t_total)
        model.zero_grad()

        logger.info('Start training....')
        for epoch in range(int(args.num_train_epochs)):

            for batch in tqdm(train_dataloader):
                batch_step += 1
                batch = move_to_cuda(batch)
                model.train()

                loss = loss_single(model, batch, args.momentum)

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(apex.amp.master_params(optimizer), args.max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                train_loss_meter.update(loss.item())

                if (batch_step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    model.zero_grad()
                    global_step += 1

                    tb_logger.add_scalar('batch_train_loss', loss.item(), global_step)
                    tb_logger.add_scalar('smoothed_train_loss', train_loss_meter.avg, global_step)

                    if args.eval_period != -1 and global_step % args.eval_period == 0:
                        mrr = predict(args, model, eval_dataloader, device, logger)
                        logger.info("Step %d Train loss %.2f MRR %.2f on epoch=%d" % (
                        global_step, train_loss_meter.avg, mrr * 100, epoch))

                        if best_mrr < mrr:
                            logger.info("Saving model with best MRR %.2f -> MRR %.2f on epoch=%d" %
                                        (best_mrr * 100, mrr * 100, epoch))
                            logger.info("Saving to {}".format(os.path.join(args.output_dir, "checkpoint_best.pt")))
                            torch.save(model.state_dict(), os.path.join(args.output_dir, "checkpoint_best.pt"))
                            model = model.to(device)
                            best_mrr = mrr

            mrr = predict(args, model, eval_dataloader, device, logger)
            logger.info("Step %d Train loss %.2f MRR %.2f on epoch=%d" % (
                global_step, train_loss_meter.avg, mrr * 100, epoch))
            tb_logger.add_scalar('dev_mrr', mrr * 100, epoch)
            if best_mrr < mrr:
                torch.save(model.state_dict(), os.path.join(
                    args.output_dir, f"checkpoint_last.pt"))
                logger.info("Saving model with best MRR %.2f -> MRR %.2f on epoch=%d" %
                            (best_mrr * 100, mrr * 100, epoch))
                torch.save(model.state_dict(), os.path.join(
                    args.output_dir, f"checkpoint_best.pt"))
                model = model.to(device)
                best_mrr = mrr

        logger.info("Training finished!")

    elif args.do_predict:
        output_dir = args.output_dir
        model.load_state_dict(torch.load(os.path.join(output_dir, 'checkpoint_best.pt')))
        acc = predict(args, model, eval_dataloader, device, logger)
        logger.info(f"test performance {acc}")


def predict(args, model, eval_dataloader, device, logger):
    model.eval()
    num_correct = 0
    num_total = 0.0
    rrs = []  # reciprocal rank
    for batch in tqdm(eval_dataloader):
        batch_to_feed = move_to_cuda(batch)
        with torch.no_grad():
            outputs = model(batch_to_feed)

            q = outputs['q']
            c = outputs['c']
            neg_c = outputs['neg_c']

            product_in_batch = torch.mm(q, c.t())
            product_neg = (q * neg_c).sum(-1).unsqueeze(1)
            product = torch.cat([product_in_batch, product_neg], dim=-1).cpu().detach()

            target = torch.arange(product.size(0)).to(product.device)
            ranked = product.argsort(dim=1, descending=True)
            prediction = product.argmax(-1)

            # MRR
            idx2rank = ranked.argsort(dim=1)
            for idx, t in enumerate(target.tolist()):
                rrs.append(1 / (idx2rank[idx][t].item() + 1))

            pred_res = prediction == target
            num_total += pred_res.size(0)
            num_correct += pred_res.sum(0)

    acc = num_correct / num_total
    mrr = np.mean(rrs)
    logger.info(f"evaluated {num_total} examples...")
    logger.info(f"avg. Acc: {acc}")
    logger.info(f'MRR: {mrr}')
    model.train()
    return mrr


if __name__ == "__main__":
    main()
