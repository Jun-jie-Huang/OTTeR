# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.

"""
Description: encode text corpus into a store of dense vectors. 

Usage (adjust the batch size according to your GPU memory):

CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/encode_corpus.py \
    --do_predict \
    --predict_batch_size 1000 \
    --model_name roberta-base \
    --predict_file ${CORPUS_PATH} \
    --init_checkpoint ${MODEL_CHECKPOINT} \
    --embed_save_path ${SAVE_PATH} \
    --fp16 \
    --max_c_len 300 \
    --num_workers 20 

"""

import sys
sys.path.append('../')
import collections
import logging
import json
import os
import random
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer, TapasTokenizer
from torch.utils.data import DataLoader
from functools import partial

# from retrieval.data.encode_datasets import EmDatasetTapas, EmDatasetBert, em_collate_q, em_collate_tb
from retrieval.data.encode_datasets import EmDataset, em_collate_bert, em_collate_tapas
from retrieval.models.retriever import TapasEncoder, TapasEncoderPure, CtxEncoder, RobertaCtxEncoder
from retrieval.data.encode_datasets import EmDataset, EmDatasetFilter, EmDatasetMeta, EmDatasetMetaThreeCat
from retrieval.models.retriever import TapasEncoder, SingleRetriever, SingleEncoder, RobertaSingleEncoder, MomentumEncoder
from retrieval.models.tb_retriever import SingleEncoderThreeCatPool, RobertaSingleEncoderThreeCatPool
from retrieval.models.tb_retriever import SingleEncoderThreeCatAtt, RobertaSingleEncoderThreeCatAtt
from retrieval.config import encode_args
from retrieval.utils.utils import move_to_cuda, load_saved

logger = logging.getLogger(__name__)
def main():
    args = encode_args()
    if args.fp16:
        import apex
        apex.amp.register_half_function(torch, 'einsum')

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')

    if not args.predict_file:
        raise ValueError(
            "If `do_predict` is True, then `predict_file` must be specified.")

    # select encoing model
    bert_config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    collate_fc = partial(em_collate_bert, pad_id=tokenizer.pad_token_id)
    if args.momentum:
        model = MomentumEncoder(bert_config, args)
        logger.info("Model Using MomentumEncoder...")
    elif "roberta" in args.model_name:
        if args.three_cat:
            if args.part_pooling[:3] == 'att':
                model = RobertaSingleEncoderThreeCatAtt(bert_config, args)
                logger.info("Model Using RobertaSingleEncoderThreeCatAtt...")
            else:
                model = RobertaSingleEncoderThreeCatPool(bert_config, args)
                logger.info("Model Using RobertaSingleEncoderThreeCatPool...")
        else:
            model = RobertaSingleEncoder(bert_config, args)
            logger.info("Model Using RobertaSingleEncoder...")
    elif "tapas" in args.model_name:
        tokenizer = TapasTokenizer.from_pretrained(args.model_name, cell_trim_length=args.cell_trim_length)
        collate_fc = partial(em_collate_tapas, pad_id=tokenizer.pad_token_id)
        model = TapasEncoderPure(bert_config, args)
        logger.info("Model Using TapasEncoderPure...")
    else:
        if args.three_cat:
            if args.part_pooling[:3] == 'att':
                model = SingleEncoderThreeCatAtt(bert_config, args)
                logger.info("Model Using SingleEncoderThreeCatAtt...")
            else:
                model = SingleEncoderThreeCatPool(bert_config, args)
                logger.info("Model Using SingleEncoderThreeCatPool...")
        else:
            model = SingleEncoder(bert_config, args)
            logger.info("Model Using SingleEncoder...")
    if args.add_special_tokens:
        # special_tokens_dict = {'additional_special_tokens': ["[HEADER]", "[PASSAGE]","[SEP]", "[TB]","[DATA]","[TITLE]","[SECTITLE]"]}
        special_tokens_dict = {'additional_special_tokens': ["[HEADER]", "[PASSAGE]", "[TB]","[DATA]","[TITLE]","[SECTITLE]"]}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))

    # select dataset
    if args.tfidf_filter and args.encode_table:
        eval_dataset = EmDatasetFilter(tokenizer, args.predict_file, args.tfidf_result_file, args.encode_table, args)
        logger.info("Dataset Using EmDatasetFilter...")
    elif args.metadata:
        if args.three_cat:
            eval_dataset = EmDatasetMetaThreeCat(tokenizer, args.predict_file, args.encode_table, args)
            logger.info("Dataset Using EmDatasetMetaThreeCat...")
        else:
            eval_dataset = EmDatasetMeta(tokenizer, args.predict_file, args.encode_table, args)
            logger.info("Dataset Using EmDatasetMeta...")
    else:
        eval_dataset = EmDataset(tokenizer, args.predict_file, args.encode_table, args)
        logger.info("Dataset Using EmDataset...")
    eval_dataset.processing_data()
    eval_dataloader = DataLoader(eval_dataset,
                                 batch_size=args.predict_batch_size,
                                 collate_fn=collate_fc,
                                 pin_memory=True,
                                 num_workers=args.num_workers)

    assert args.init_checkpoint != ""
    model = load_saved(model, args.init_checkpoint, exact=False)
    model.to(device)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model = amp.initialize(model, opt_level=args.fp16_opt_level)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # import pdb; pdb.set_trace()

    embeds = predict(model, eval_dataloader)
    logger.info(embeds.size())

    if not os.path.exists(os.path.dirname(args.embed_save_path)):
        os.makedirs(os.path.dirname(args.embed_save_path))
        logger.info("making dir :{}".format(os.path.dirname(args.embed_save_path)))
    logger.info("saving to :{}".format(args.embed_save_path))
    np.save(args.embed_save_path, embeds.cpu().numpy())


def predict(model, eval_dataloader):
    if type(model) == list:
        model = [m.eval() for m in model]
    else:
        model.eval()

    embed_array = []
    # import pdb; pdb.set_trace()
    # logger.info("start from 379200")
    for idx, batch in enumerate(tqdm(eval_dataloader)):
        batch = move_to_cuda(batch)
        with torch.no_grad():
            try:
                results = model(batch)
            except Exception as e:
                logger.info(e)
                # logger.info("Error Batch: {}, instance: {}".format(idx, idx*1600))
                continue
            embed = results['embed'].cpu()
            embed_array.append(embed)

    ## linear combination tuning on dev data
    embed_array = torch.cat(embed_array, dim=0)

    # model.train()
    return embed_array


if __name__ == "__main__":
    main()
