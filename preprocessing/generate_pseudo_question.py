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

import nltk
import nltk.data
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from difflib import SequenceMatcher
from fuzzywuzzy import fuzz
import dateparser
from dateparser.search import search_dates
from dateparser import parse

logger = logging.getLogger()
nltk.download('stopwords')

stopWords = set(stopwords.words('english'))
tfidf = TfidfVectorizer(strip_accents="unicode", ngram_range=(2, 3), stop_words=stopWords)
# tfidf = TfidfVectorizer(strip_accents="unicode", ngram_range=(1, 3), stop_words=stopWords)
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
# basic_dir = '/home/t-wzhong/v-wanzho/ODQA/data'
basic_dir = '..'
resource_path = f'{basic_dir}/data_wikitable/'
best_threshold = 0.80

# from utils_preprocess import *
from utils_preprocess import read_jsonl, read_json, write_json, write_jsonl
from drqa import retriever
from pretrain_corpus_preprocess import process_wiki_tables_with_hyphen

def get_sorted_sentences(tfidf, tfidf_query, all_index, all_passages):
    keys = []
    paras = []
    # threshold = 0.99
    for k, v in zip(all_index, all_passages):
        for _ in tokenizer.tokenize(v):
            keys.append(k)
            paras.append(_)
    try:
        para_feature = tfidf.fit_transform(paras)
        transformed = True
    except Exception:
        # logger.info("only containing stop words, skip it")
        transformed = False
    if transformed:
        q_feature = tfidf.transform(tfidf_query)
        para_tfidf_dist = pairwise_distances(q_feature, para_feature, 'cosine')[0]
        sorted_passages = [(k, para, d) for k, para, d in zip(keys, paras, para_tfidf_dist)]
        sorted_passages = sorted(sorted_passages, key=lambda x: x[2], reverse=False)  # 升序
        return sorted_passages
    else:
        return [(k, para, 1) for k, para in zip(keys, paras)]


def get_sorted_passages(tfidf, tfidf_query, all_index, all_passages):
    keys = []
    paras = []
    # threshold = 0.99
    for k, v in zip(all_index, all_passages):
        keys.append(k)
        paras.append(v)
    try:
        para_feature = tfidf.fit_transform(paras)
        transformed = True
    except Exception:
        # logger.info("only containing stop words, skip it")
        transformed = False
    if transformed:
        q_feature = tfidf.transform(tfidf_query)
        para_tfidf_dist = pairwise_distances(q_feature, para_feature, 'cosine')[0]
        sorted_passages = [(k, para, d) for k, para, d in zip(keys, paras, para_tfidf_dist)]
        sorted_passages = sorted(sorted_passages, key=lambda x: x[2], reverse=False)  # 升序
        return sorted_passages
    else:
        return [(k, para, 1) for k, para in zip(keys, paras)]


def get_passages_traindev(tfidf_query, first_batch_passages, second_batch_passages, mode):
    fir_index, fir_passages = first_batch_passages['idx'], first_batch_passages['psg']  # often answer_node_passages
    sec_index, sec_passages = second_batch_passages['idx'], second_batch_passages['psg']  # often gpt_passages to append

    output_index = []
    output_passages = []
    if mode=='sort_sent':
        sorted_instances = get_sorted_sentences(tfidf, tfidf_query, fir_index, fir_passages)

        output_passages = [item[1] for item in sorted_instances]
        output_index = [item[0] for item in sorted_instances]

    elif mode=='sort_psg':
        sorted_instances = get_sorted_passages(tfidf, tfidf_query, fir_index, fir_passages)

        output_passages = [item[1] for item in sorted_instances]
        output_index = [item[0] for item in sorted_instances]

    elif mode=='append_sort_psg':
        # if len(sec_passages) > 0:
        sorted_instances = get_sorted_passages(tfidf, tfidf_query, sec_index, sec_passages)
        sorted_passages = [item[1] for item in sorted_instances]
        sorted_index = [item[0] for item in sorted_instances]
        appended_index = [num_id for num_id, idx in enumerate(sorted_index) if idx not in fir_index]

        output_index = fir_index + [sorted_index[num_id] for num_id in appended_index]
        output_passages = fir_passages + [sorted_passages[num_id] for num_id in appended_index]

    elif mode=='sort_append_sort_psg':
        # if len(sec_passages) > 0:
        sorted_instances = get_sorted_passages(tfidf, tfidf_query, fir_index, fir_passages)
        sorted_passages_fir = [item[1] for item in sorted_instances]
        sorted_index_fir = [item[0] for item in sorted_instances]

        sorted_instances = get_sorted_passages(tfidf, tfidf_query, sec_index, sec_passages)
        sorted_passages = [item[1] for item in sorted_instances]
        sorted_index = [item[0] for item in sorted_instances]
        appended_index = [num_id for num_id, idx in enumerate(sorted_index) if idx not in sorted_index_fir]

        output_index = sorted_index_fir + [sorted_index[num_id] for num_id in appended_index]
        output_passages = sorted_passages_fir + [sorted_passages[num_id] for num_id in appended_index]
    stats = {'num-fir': len(fir_index), 'num-sec': len(sec_index), 'num-out': len(output_index)}
    assert(len(output_index)==len(output_passages))
    return {'idx': output_index, 'psg': output_passages, 'stats': stats}


def prepare_generation_data_cell_level(zipped_data):
    # load data
    data, passage_links = zipped_data[0], zipped_data[1]
    table_id = data['table_id']
    # Loading the table/request information
    with open(f'{resource_path}/{args.table_path}/{table_id}.json'.encode('utf8')) as f:
        table = json.load(f)
    with open(f'{resource_path}/{args.request_path}/{table_id}.json'.encode('utf8')) as f:
        requested_documents = json.load(f)

    header = [inst[0] for inst in table['header']]
    contents = [[cell[0] for cell in row] for row in table['data']]
    if args.table_pruning:
        header = [' '.join(inst[0].split(' ')[:args.max_header_cell_tokens]) for inst in table['header']]
        contents = [[' '.join(cell[0].split(' ')[:args.max_cell_tokens]) for cell in row] for row in table['data']]
    if not args.maintain_ori_table:
        header, contents = remove_null_header_column(header, contents)
        header, contents = remove_sequence_number_column(header, contents)
        table['data'] = remove_removed_contents(table['data'], contents)
    if len(' '.join(header).split(' ')) > 100:
        logger.info("HEA::{}::{}".format(len(' '.join(header).split(' ')), header))

    tables = []
    for i, row in enumerate(contents):
        # new_row = [' '.join(cell.split(' ')[:args.max_cell_tokens]) for cell in row]
        # new_header = [' '.join(cell.split(' ')[:args.max_header_cell_tokens]) for cell in header]
        # df = pd.DataFrame(data=[new_row], columns=new_header)
        df = pd.DataFrame(data=[row], columns=header)
        df = df.applymap(str)
        tables.append(df)

    meta_data = {'title': table['title'],
                 'section_title': table['section_title'],
                 'header': header}
    answer = data['answer-text']

    # Mapping entity link to cell, entity link to surface word, 确定每个有link的cell的表格位置，
    mapping_entity = {}
    position2wiki = []
    for row_idx, row in enumerate(table['data']):
        position2wiki_i = {}
        for col_idx, cell in enumerate(row):
            for i, ent in enumerate(cell[1]):
                mapping_entity[ent] = mapping_entity.get(ent, []) + [(row_idx, col_idx)]
                position2wiki_i[f'{row_idx},{col_idx}'] = position2wiki_i.get(f'{row_idx},{col_idx}', []) + [ent]
        position2wiki.append(position2wiki_i)

    positive_index = [node[1] for node in data['answer-node']]
    positive_row_index_in_answer_node = list(set([node[1][0] for node in data['answer-node']]))
    positive_index_dict = {node[1][0]: [node[1][1]] for node in data['answer-node']} # 固定一行只有一个answer

    processed_data = []
    for row_idx, row in enumerate(table['data']):
        tfidf_query = [' ' + ' '.join(header) + ' ' + ' '.join(contents[row_idx])]
        if row_idx in positive_row_index_in_answer_node:
            true_cell_id = positive_index_dict[row_idx][0]
            header_j = header[true_cell_id]
            cell_j = row[true_cell_id][0]
            _index = row[true_cell_id][1]
            other_index = [item[1] for item in row[:true_cell_id]+row[true_cell_id+1:]]
            other_index = [item2 for item in other_index for item2 in item]
            _index.extend(other_index)
            _index = list(filter(None, _index))
            raw_passages = [requested_documents[index] for index in _index]
            raw_passages = list(filter(None, raw_passages))
            out_psg = get_passages_traindev(tfidf_query=tfidf_query,
                                            first_batch_passages={'idx': _index, 'psg': raw_passages},
                                            second_batch_passages={'idx': [], 'psg': []},
                                            mode='sort_psg')
            psg_str = ' '.join('[SEP] '.join(out_psg['psg']).split(' ')[:600])
            _header = copy.deepcopy(header)
            _content = copy.deepcopy(contents[row_idx])
            # document = '[TAB] {} [DATA] {} [PASSAGE] {}'.format(
            #     meta_data['section_title'],
            #     ' '.join(['{} is {}.'.format(h, c) for h, c in zip(_header, _content)]),
            #     psg_str)
            document = '[TAB] [TITLE] {} [SECTITLE] {} [DATA] {} [PASSAGE] {}'.format(
                meta_data['title'],
                meta_data['section_title'],
                ' '.join(['{} is {}.'.format(h, c) for h, c in zip(_header, _content)]),
                psg_str)

            fake_query = 'What is {} of {} in {}?'.format(cell_j, header_j, table['title'])
            psg_title = [' '.join(item[6:].split('_')) for item in _index]
            fake_query += ' What is {}?'.format(' and '.join(psg_title))

            query = data['question']

            processed_data.append({'input': document,
                                   'output': query,
                                   'fake': fake_query})

    return {'traindev_data': processed_data}


def prepare_generation_table_corpus(kv):
    table_id, table = kv

    if 'wiki' in args.split:
        table['header'] = [[' '.join(item[0]), [item[1]]] for item in table['header']]
        table['data'] = [[[' '.join(item2[0]), item2[1]] for item2 in item1] for item1 in table['data']]
    header = [inst[0] for inst in table['header']]
    contents = [[cell[0] for cell in row] for row in table['data']]
    if args.table_pruning:
        header = [' '.join(inst[0].split(' ')[:args.max_header_cell_tokens]) for inst in table['header']]
        contents = [[' '.join(cell[0].split(' ')[:args.max_cell_tokens]) for cell in row] for row in table['data']]
    if not args.maintain_ori_table:
        header, contents = remove_null_header_column(header, contents)
        header, contents = remove_sequence_number_column(header, contents)
        table['data'] = remove_removed_contents(table['data'], contents)
    if len(' '.join(header).split(' ')) > 100:
        logger.info("HEA::{}::{}".format(len(' '.join(header).split(' ')), header))

    tables = []
    for i, row in enumerate(contents):
        new_row = [' '.join(cell.split(' ')[:args.max_cell_tokens]) for cell in row]
        # new_header, new_row, new_max_cell = pruning_tables_with_max_cell_length(header, new_row, args.max_cell_tokens)
        new_header = [' '.join(cell.split(' ')[:args.max_header_cell_tokens]) for cell in header]
        df = pd.DataFrame(data=[new_row], columns=new_header)
        df = df.applymap(str)
        tables.append(df)

    meta_data = {'title': table['title'],
                 'section_title': table['section_title'],
                 'header': header}

    # # Mapping entity link to cell, entity link to surface word, 确定每个有link的cell的表格位置，
    mapping_entity = {}
    position2wiki = []
    for row_idx, row in enumerate(table['data']):
        position2wiki_i = {}
        for col_idx, cell in enumerate(row):
            for i, ent in enumerate(cell[1]):
                mapping_entity[ent] = mapping_entity.get(ent, []) + [(row_idx, col_idx)]
                position2wiki_i[f'{row_idx},{col_idx}'] = position2wiki_i.get(f'{row_idx},{col_idx}', []) + [ent]
        position2wiki.append(position2wiki_i)

    processed_data = []
    for row_idx, row in enumerate(table['data']):
        tfidf_query = [' ' + ' '.join(header) + ' ' + ' '.join(contents[row_idx])]
        for col_idx, cell in enumerate(row):
            # header_j = header[col_idx]
            # cell_j = cell[0]
            _index = cell[1]
            _index = list(filter(None, _index))
            if 'wiki' in args.split:
                _index = [item.replace('-', '_') for item in _index]
            if len(_index) > 0:
                # query = 'What is {} of {} in {}?'.format(cell_j, header_j, table['title'])
                # if args.query_add_psg_title:
                #     psg_title = [' '.join(item[6:].split('_')) for item in _index]
                #     query += ' What is {}?'.format(' and '.join(psg_title))
                raw_passages = [all_requests.get(index, None) for index in _index]
                raw_passages = list(filter(None, raw_passages))
                out_psg = get_passages_traindev(tfidf_query=tfidf_query,
                                                first_batch_passages={'idx': _index, 'psg': raw_passages},
                                                second_batch_passages={'idx': [], 'psg': []},
                                                mode='sort_psg')
                psg_str = ' '.join('[SEP] '.join(out_psg['psg']).split(' ')[:600])
                _header = copy.deepcopy(header)
                _content = copy.deepcopy(contents[row_idx])
                # document = '[TAB] {} [DATA] {} [PASSAGE] {}'.format(
                #     meta_data['section_title'],
                #     ' '.join(['{} is {}.'.format(h, c) for h, c in zip(_header, _content)]),
                #     psg_str)
                document = '[TAB] [TITLE] {} [SECTITLE] {} [DATA] {} [PASSAGE] {}'.format(
                    meta_data['title'],
                    meta_data['section_title'],
                    ' '.join(['{} is {}.'.format(h, c) for h, c in zip(_header, _content)]),
                    psg_str)

                full_index = '{}@r{}@c{}'.format(table_id, row_idx, col_idx)
                processed_data.append({'input': document,
                                       'output': '',
                                       'index': full_index})

    return {'corpus_data': processed_data}


def get_filename(args):
    # prefix = args.nega
    # if args.maintain_ori_table:
    #     prefix += '_oriTable'
    # if args.query_add_psg_title:
    #     prefix += '_qPsgTitle'
    if args.table_pruning:
        prefix += '_tPrun'
    prefix += '_{}'.format(args.prefix) if args.prefix else ''

    dir = '{}/preprocessed_data/pretrain/generation_{}'.format(basic_dir, prefix)
    if not os.path.exists(dir):
        os.makedirs(dir)
    save_path = os.path.join(dir, '{}.json'.format(args.split))
    logger.info(" save_path :{}".format(save_path))
    return save_path


if __name__ == '__main__':
    """
        python generate_pseudo_question.py --split dev --table_pruning 
        python generate_pseudo_question.py --split dev --table_pruning --prefix title
        python generate_pseudo_question.py --split table_corpus_wiki --table_pruning --prefix title
        cp -r ../preprocessed_data/pretrain/generation_rand_row/ ~/table-odqa/Data/evidence_chain/question_tb_pretraining/
        cp -r ../preprocessed_data/pretrain/generation_rand_row_tPrun_title/ ~/table-odqa/Data/evidence_chain/question_tb_pretraining/
        
    """
    parser = argparse.ArgumentParser()
    # parser.add_argument('--split', required=True, type=str)
    parser.add_argument('--split', default='train', type=str, choices=['train', 'dev', 'table_corpus_wiki'])
    parser.add_argument('--model', default='../../OTT-QA/retriever/title_sectitle_schema/index-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz', type=str)
    # parser.add_argument('--nega', default='rand_row', type=str, choices=['rand_row', 'rand_cell', 'rand_andcellrow'])
    parser.add_argument('--prefix', default='', type=str, help="blink ")
    parser.add_argument('--max_passages', default=3, type=int)
    parser.add_argument('--max_tokens', default=360, type=int)
    parser.add_argument('--max_cell_tokens', default=36, type=int)
    parser.add_argument('--max_header_cell_tokens', default=20, type=int)
    parser.add_argument('--run_id', default=1, type=int)
    parser.add_argument('--reprocess', action='store_true')
    parser.add_argument('--table_pruning', action='store_true')
    parser.add_argument('--maintain_ori_table', action='store_false')
    # parser.add_argument('--query_add_psg_title', action='store_true')
    args = parser.parse_args()

    args.table_path = 'traindev_tables_tok'
    args.request_path = 'traindev_request_tok'
    args.maintain_ori_table = True
    # args.query_add_psg_title = True

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    basic_dir = '..'  # junjie
    n_threads = 24
    logger.info("using {}".format(args.split))
    save_path = get_filename(args)
    logger.info("save file to {}".format(save_path))

    if args.split in ['train', 'dev', 'traindev']:
        if not os.path.exists(save_path) or args.reprocess:
            data1, data2 = [], []
            if 'train' in args.split:
                with open('{}/data_ottqa/train.traced.json'.format(basic_dir), 'r') as f:
                    data1 = json.load(f)
            if 'dev' in args.split:
                with open('{}/data_ottqa/dev.traced.json'.format(basic_dir), 'r') as f:
                    data2 = json.load(f)
            data = data1 + data2
            logger.info("num of data: {}, data train: {}, data dev: {} ".format(len(data), len(data1), len(data2)))
            table2tbid = {item['table_id']: {} for item in data}
            zipped_data = [(item, table2tbid[item['table_id']]) for item in data]

            results = []
            # import pdb;  pdb.set_trace()
            with Pool(n_threads) as p:
                func_ = partial(prepare_generation_data_cell_level,)
                all_results = list(tqdm(p.imap(func_, zipped_data, chunksize=16), total=len(zipped_data), desc="convert to pretrain data",))
                results = [res['traindev_data'] for res in all_results]
            # # debug
            # for d in zipped_data:
            #     all_results = prepare_training_data_intable(d)
            #     results = all_results['traindev_data']

            logger.info(len(results))
            results = [item for inst in results for item in inst]
            random.shuffle(results)
            logger.info(len(results))
            write_jsonl(results, save_path)

    elif args.split in ['table_corpus_wiki']:
        if not os.path.exists(save_path) or args.reprocess:

            if args.split == 'table_corpus_wiki':
                table_path = '{}/data_wikitable/all_tables.json'.format(basic_dir)
            else:
                raise NotImplementedError("{} not in args.split choices.".format(args.split))
            with open(table_path, 'r') as f:
                all_tables = json.load(f)
            with open('{}/data_wikitable/all_passages.json'.format(basic_dir), 'r') as f:
                all_requests = json.load(f)
            if args.split == 'table_corpus_wiki':
                all_requests = process_wiki_tables_with_hyphen(all_requests)
            logger.info("Finish loading all tables and all requests.")

            logger.info('Start processing tables.')
            with Pool(n_threads) as p:
                func_ = partial(prepare_generation_table_corpus)
                all_results = list(tqdm(p.imap(func_, all_tables.items(), chunksize=16), total=len(all_tables),
                                        desc="process tables", ))
                results = [res['corpus_data'] for res in all_results]

            logger.info(len(results))
            results = [item for inst in results for item in inst]
            random.shuffle(results)
            logger.info(len(results))
            logger.info("start saving to {}".format(save_path))
            write_jsonl(results, save_path)


