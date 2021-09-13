import argparse
import json
import pickle
import sys
import os
import re
import copy
import random
import logging
from functools import partial
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pandas as pd

from utils_preprocess import write_jsonl, read_jsonl, read_json, write_json


def get_negative_idx(kv):
    idx, tbid = kv
    tbid_list = tbid.split('@')
    if len(tbid_list) != 3:
        tbid_list = ['@'.join(tbid_list[:-2]), tbid_list[-2], tbid_list[-1]]

    try:
        if args.nega == 'all_rand':
            choices = list(range(idx)) + list(range(idx + 1, length))
            neg_idx = random.choice(choices)
        elif args.nega == 'rand_row':
            table_id = tbid_list[0]
            choices_str = copy.deepcopy(tableid2idx[table_id])
            choices_str.remove([tbid, idx])
            if len(choices_str) > 0:
                neg_idx = random.choice(choices_str)[1]
            else:
                neg_idx = random.choice(list(range(idx)) + list(range(idx + 1, length)))
        elif args.nega == 'rand_cell':
            table_id = tbid_list[0]
            table_row_id = tbid_list[1]
            choices_str = copy.deepcopy(tableid2idx[table_id])
            choices_str.remove([tbid, idx])
            choices_str = [item for item in chocies_str if item[0].split("@")[-2] == table_row_id]
            if len(choices_str) > 0:
                neg_idx = random.choice(choices_str)[1]
            else:
                neg_idx = random.choice(list(range(idx)) + list(range(idx + 1, length)))
        else:
            logger.info('wrong args.nega')
            choices = list(range(idx)) + list(range(idx + 1, length))
            neg_idx = random.choice(choices)
    except:
        choices = list(range(idx)) + list(range(idx + 1, length))
        neg_idx = random.choice(choices)
        logger.info('error in pos:{}, tb_id: {}, neg:{}'.format(idx, tbid, neg_idx))

    return neg_idx


# def merge_negatives(pos, neg):
def merge_negatives(pos_index, neg_index):
    pos = bart_data[pos_index]
    neg = bart_data[neg_index]

    processed_data = []

    pos_table_idx = pos['index'].split('@')
    neg_table_idx = neg['index'].split('@')
    if len(pos_table_idx) != 3:
        pos_table_idx = ['@'.join(pos_table_idx[:-2]), pos_table_idx[-2], pos_table_idx[-1]]
    if len(neg_table_idx) != 3:
        neg_table_idx = ['@'.join(neg_table_idx[:-2]), neg_table_idx[-2], neg_table_idx[-1]]
    example = {}
    example['table_id'] = pos_table_idx[0]
    example['question'] = pos['output']
    example['label'] = 1
    example['passages'] = pos['input']
    example['row_id'] = int(pos_table_idx[0])
    example['cell_id'] = int(pos_table_idx[1])

    example['neg_passages'] = neg['input']
    example['neg_row_id'] = int(neg_table_idx[0])
    example['neg_cell_id'] = int(neg_table_idx[1])

    processed_data.append(example)
    return {'retrieval_data': processed_data}


def pretrain_bart_preprocess(pos_index):
    neg_index = get_negative_idx([pos_index, idx2tbid[pos_index]])
    # logger.info("neg: {}".format(neg_index))

    pos = bart_data[pos_index]
    neg = bart_data[neg_index]
    processed_data = []
    pos_table_idx = pos['index'].split('@')
    neg_table_idx = neg['index'].split('@')
    if len(pos_table_idx) != 3:
        pos_table_idx = ['@'.join(pos_table_idx[:-2]), pos_table_idx[-2], pos_table_idx[-1]]
    if len(neg_table_idx) != 3:
        neg_table_idx = ['@'.join(neg_table_idx[:-2]), neg_table_idx[-2], neg_table_idx[-1]]
    example = {}
    example['table_id'] = pos_table_idx[0]
    example['question'] = pos['output']
    example['label'] = 1
    example['passages'] = pos['input']
    example['row_id'] = int(pos_table_idx[1][1:])
    example['cell_id'] = int(pos_table_idx[2][1:])

    example['neg_passages'] = neg['input']
    example['neg_row_id'] = int(neg_table_idx[1][1:])
    example['neg_cell_id'] = int(neg_table_idx[2][1:])

    processed_data.append(example)
    return {'retrieval_data': processed_data}


basic_dir = '..'
def get_filename(args):
    prefix = args.prefix
    prefix += '_{}'.format(args.nega)

    save_path = '{}/preprocessed_data/pretrain/bart/{}_{}.json'.format(basic_dir, args.split, prefix)
    logger.info(" save_path :{}".format(save_path))
    return save_path


if __name__ == '__main__':
    """
        python prepro_bart_to_pretrain.py --split table_corpus_wiki --prefix tPrun_title --nega rand_row
        python prepro_bart_to_pretrain.py --split table_corpus_wiki --prefix tPrun_title --nega all_rand    
    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='table_corpus_wiki', type=str, choices=['table_corpus_wiki'])
    parser.add_argument('--bart_generate_file', default='../preprocessed_data/pretrain/generation_rand_row_tPrun_title/evidence_generation_output_pretrain_with_index.json', type=str, help="")
    parser.add_argument('--prefix', default='tPrun_title', type=str, help="['tPrun_title',]")

    parser.add_argument('--nega', default='rand_row', type=str, choices=['rand_row', 'rand_cell', 'all_rand'])
    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    bart_data = read_jsonl(args.bart_generate_file)
    length = len(bart_data)
    save_path = get_filename(args)

    # get index dict
    idx2tbid = {idx: item['index'] for idx, item in enumerate(bart_data)}
    tbid2idx = {item['index']: idx for idx, item in enumerate(bart_data)}
    idx2tableid = {idx: item['index'].split('@')[0] for idx, item in enumerate(bart_data)}
    tableid2idx = {}
    for key, value in idx2tableid.items():
        tableid2idx.setdefault(value, [])
        tableid2idx[value].append([idx2tbid[key], key])
    logger.info("finish process index.")

    all_positive_index = list(range(length))
    # zipped = zip(all_positive_index, all_negative_index)
    n_threads = 24
    with Pool(n_threads) as p:
        func_ = partial(pretrain_bart_preprocess)
        all_results = list(tqdm(p.imap(func_, all_positive_index, chunksize=16), total=len(all_positive_index),
                                desc="get negatives and process posneg", ))
        results = [res['retrieval_data'] for res in all_results]

    logger.info(len(results))
    results = [item for inst in results for item in inst]
    random.shuffle(results)
    logger.info(len(results))
    logger.info("start saving to {}".format(save_path))
    write_jsonl(results, save_path)

