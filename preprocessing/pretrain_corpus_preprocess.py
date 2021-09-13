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
best_threshold = 0.80

# from utils_preprocess import *
from utils_preprocess import read_jsonl, read_json, write_jsonl, write_json
from drqa import retriever


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


def process_wiki_tables_with_hyphen(all_passages):
    logger.info("all passages number: {}".format(len(all_passages)))
    hephen_keys = [item for item in all_passages.keys() if '-' in item]
    hephen_keys_convert = [key.replace('-', '_') for key in hephen_keys]
    logger.info("length of keys in all passages with hyphen:{}".format(len(all_passages)))
    added_passages = {k2: all_passages[k1] for k1, k2 in zip(hephen_keys, hephen_keys_convert)}
    logger.info("length of added_passages with replaced hyphen:{}".format(len(added_passages)))
    all_passages.update(added_passages)
    logger.info("new all passages number: {}".format(len(all_passages)))
    return all_passages


def prepare_pretraining_table_corpus(kv):
    table_id, table = kv

    if 'wiki' in args.split:
        table['header'] = [[' '.join(item[0]), [item[1]]] for item in table['header']]
        table['data'] = [[[' '.join(item2[0]), item2[1]] for item2 in item1] for item1 in table['data']]
    header = [inst[0] for inst in table['header']]
    contents = [[cell[0] for cell in row] for row in table['data']]
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

    idx2rowid = {}
    cell_idx = []
    queries = []
    documents = []
    passages_index = []
    for row_idx, row in enumerate(table['data']):
        tfidf_query = [' ' + ' '.join(header) + ' ' + ' '.join(contents[row_idx])]
        for col_idx, cell in enumerate(row):
            header_j = header[col_idx]
            cell_j = cell[0]
            _index = cell[1]
            _index = list(filter(None, _index))
            if 'wiki' in args.split:
                _index = [item.replace('-', '_') for item in _index]
            if len(_index) > 0:
                query = 'What is {} of {} in {}?'.format(cell_j, header_j, table['title'])
                if args.query_add_psg_title:
                    psg_title = [' '.join(item[6:].split('_')) for item in _index]
                    query += ' What is {}?'.format(' and '.join(psg_title))
                raw_passages = [all_requests.get(index, None) for index in _index]
                raw_passages = list(filter(None, raw_passages))
                out_psg = get_passages_traindev(tfidf_query=tfidf_query,
                                                first_batch_passages={'idx': _index, 'psg': raw_passages},
                                                second_batch_passages={'idx': [], 'psg': []},
                                                mode='sort_psg')
                psg_str = ' '.join('[SEP] '.join(out_psg['psg']).split(' ')[:600])
                _header = copy.deepcopy(header)
                _content = copy.deepcopy(contents[row_idx])
                if args.mask_cell:
                    _header = _header[:col_idx] + _header[col_idx+1:]
                    _content = _content[:col_idx] + _content[col_idx+1:]
                document = '[TAB] {} [DATA] {} [PASSAGE] {}'.format(
                    meta_data['section_title'],
                    ' '.join(['{} is {}.'.format(h, c) for h, c in zip(_header, _content)]),
                    psg_str)

                idx2rowid[len(queries)] = row_idx
                cell_idx.append(col_idx)
                queries.append(query)
                documents.append(document)
                passages_index.append({'all_index': _index, 's_index': out_psg['idx']})

    # Merge a question instance to multiple positive and negative examples
    processed_data = []
    # import pdb;  pdb.set_trace()
    for i, q in enumerate(queries):
        example = {}
        # example['quesiton_id'] = data['question_id'] + f'-{len(processed_data)}'
        example['table_id'] = table_id
        example['question'] = q
        example['label'] = 1
        # example['meta_data'] = meta_data
        example['passages'] = documents[i]
        # example['passages_id'] = passages_index[i]
        example['row_id'] = idx2rowid[i]
        example['cell_id'] = cell_idx[i]

        # process negatives
        if args.nega == 'rand_cell':
            row_cells = [i for i, r_i in idx2rowid.items() if r_i == idx2rowid[i]]
            row_cells.remove(i)
            if len(row_cells) > 1:
                neg_i = random.sample(row_cells, 1)[0]
            else:
                population = list(range(len(queries)))
                population.remove(i)
                if len(population) == 0:
                    continue
                neg_i = random.sample(population, 1)[0]
        elif args.nega == 'rand_andcellrow':
            row_cells = [i for i, r_i in idx2rowid.items() if r_i == idx2rowid[i]]
            row_cells.remove(i)
            flag = random.sample([0, 1], 1)[0]
            if len(row_cells) > 1 and flag:
                neg_i = random.sample(row_cells, 1)[0]
            else:
                population = list(range(len(queries)))
                population.remove(i)
                if len(population) == 0:
                    continue
                neg_i = random.sample(population, 1)[0]
        elif args.nega == 'rand_row':
            population = list(range(len(queries)))
            population.remove(i)
            if len(population) == 0:
                continue
            neg_i = random.sample(population, 1)[0]
        else:
            population = list(range(len(queries)))
            population.remove(i)
            if len(population) == 0:
                continue
            neg_i = random.sample(population, 1)[0]

        example['neg_passages'] = documents[neg_i]
        # example['neg_passages_id'] = passages_index[neg_i]
        example['neg_row_id'] = idx2rowid[neg_i]
        example['neg_cell_id'] = cell_idx[neg_i]

        processed_data.append(example)

    return {'retrieval_data': processed_data}


def get_filename(args):
    prefix = args.nega
    # if args.maintain_ori_table:
    #     prefix += '_oriTable'
    if args.query_add_psg_title:
        prefix += '_qPsgTitle'
    if args.mask_cell:
        prefix += '_maskC'
    prefix += '_{}'.format(args.prefix) if args.prefix else ''

    save_path = '{}/preprocessed_data/pretrain/{}_{}.json'.format(basic_dir, args.split, prefix)
    logger.info(" save_path :{}".format(save_path))
    return save_path


if __name__ == '__main__':
    """
        # new:
        python corpus_preprocess.py --split table_corpus_wiki
        python corpus_preprocess.py --split table_corpus_wiki --query_add_psg_title --mask_cell
            
    """
    parser = argparse.ArgumentParser()
    # parser.add_argument('--split', required=True, type=str)
    parser.add_argument('--split', default='table_corpus_wiki', type=str, choices=['table_corpus_wiki'])
    parser.add_argument('--model', default='../../OTT-QA/retriever/title_sectitle_schema/index-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz', type=str)
    parser.add_argument('--nega', default='rand_row', type=str, choices=['rand_row', 'rand_cell', 'rand_andcellrow'])
    parser.add_argument('--prefix', default='', type=str, help="['CellPsg',]")
    # parser.add_argument('--task', default='ottqa-train', type=str, choices=['ottqa-train', 'ottqa-dev', 'ottqa-test'])
    parser.add_argument('--max_passages', default=3, type=int)
    parser.add_argument('--max_tokens', default=360, type=int)
    parser.add_argument('--max_cell_tokens', default=36, type=int)
    parser.add_argument('--max_header_cell_tokens', default=20, type=int)
    parser.add_argument('--run_id', default=1, type=int)  # TODO
    parser.add_argument('--reprocess', action='store_true')
    # parser.add_argument('--replace_link_passages', action='store_true')
    # parser.add_argument('--aug_gpt',action='store_true', help="augment retrieval data with gpt links")
    parser.add_argument('--maintain_ori_table', action='store_false')
    parser.add_argument('--query_add_psg_title', action='store_true')
    parser.add_argument('--mask_cell', action='store_true')
    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    basic_dir = '..'  # junjie
    n_threads = 24

    task = 'ottqa'
    save_path = get_filename(args)
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
            func_ = partial(prepare_pretraining_table_corpus)
            all_results = list(tqdm(p.imap(func_, all_tables.items(), chunksize=16), total=len(all_tables), desc="process tables", ))
            results = [res['retrieval_data'] for res in all_results]

        # results = []
        # RUN_INDEX = [0, 70000, 140000, 210000, 280000, 350000, 419183]
        # for kv in tqdm(list(all_tables.items())[RUN_INDEX[args.run_id - 1]:RUN_INDEX[args.run_id]], total=RUN_INDEX[args.run_id] - RUN_INDEX[args.run_id - 1], desc="process tables"):
        #     all_results = prepare_retrieve_table_fine_selection(kv,)
        #     results.extend(all_results['retrieval_data'])

        logger.info(len(results))
        results = [item for inst in results for item in inst]
        random.shuffle(results)
        logger.info(len(results))
        logger.info("start saving to {}".format(save_path))
        write_jsonl(results, save_path)
