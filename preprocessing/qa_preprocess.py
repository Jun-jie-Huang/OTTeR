import sys
sys.path.append('../')
import argparse
import logging
import json
import pickle
import pandas as pd
import os
import random
import copy
from functools import partial
from multiprocessing import Pool, cpu_count
from tqdm import tqdm, trange
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
import nltk.data

from drqa import retriever
from utils.common import convert_tb_to_string_metadata
from utils_preprocess import remove_null_header_column, remove_removed_contents, remove_sequence_number_column
from utils_preprocess import rank_doc_tfidf
from qa.utils import read_jsonl, read_json
from preprocessing import get_sorted_passages, get_sorted_sentences

basic_dir = '..'
resource_path = f'{basic_dir}/data_wikitable/'
LINKDICT = {'ori': f'{basic_dir}/data_wikitable/all_constructed_tables.json',
            'gpt_only': f'{basic_dir}/data_wikitable/all_constructed_gptonly_tables.json',
            'blink_only': f'{basic_dir}/data_wikitable/all_constructed_blink_tables.json',
            'ori_gpt': f'{basic_dir}/data_wikitable/all_constructed_gptdoc_tables.json',
            'blink_gpt': f'{basic_dir}/data_wikitable/all_constructed_blinkgpt_tables.json',
            }

nltk.download('stopwords')
stopWords = set(stopwords.words('english'))
tfidf = TfidfVectorizer(strip_accents="unicode", ngram_range=(2, 3), stop_words=stopWords)
# tfidf = TfidfVectorizer(strip_accents="unicode", ngram_range=(1, 3), stop_words=stopWords)
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def merge_gt_linked_passages(gt_passages, linked_passages):
    if len(linked_passages) <= len(gt_passages):
        return gt_passages
    elif not gt_passages:
        return linked_passages
    else:
        gt_ids = [gt['passage_id'] for gt in gt_passages]
        neg_passages = [item for item in linked_passages if item['passage_id'] not in gt_ids]
        output = gt_passages + random.choices(neg_passages,k=min(len(neg_passages), len(linked_passages) - len(gt_passages)))
        return output

def prepare_training_data_random_nega(ziped_data, table_path, request_path, max_passages=3, max_tokens=360,
                                      max_cell_tokens=36, max_header_cell_tokens=20):
    # load data
    data, retrieval_output = ziped_data[0], ziped_data[1]
    table_id = data['table_id']
    output_compare = []
    # Loading the table/request information
    with open(f'{resource_path}/{table_path}/{table_id}.json'.encode('utf8')) as f:
        table = json.load(f)
    with open(f'{resource_path}/{request_path}/{table_id}.json'.encode('utf8')) as f:
        requested_documents = json.load(f)
    ori_header = [inst[0] for inst in table['header']]
    ori_contents = [[cell[0] for cell in row] for row in table['data']]
    header = copy.deepcopy(ori_header)
    contents = copy.deepcopy(ori_contents)
    # header, contents = remove_null_header_column(header, contents)
    # header, contents = remove_sequence_number_column(header, contents)
    # table['data'] = remove_removed_contents(table['data'], contents)
    if len(' '.join(header).split(' ')) > 100:
        logger.info("HEA::{}::{}".format(len(' '.join(header).split(' ')), header))

    meta_data = {'title': table['title'],
                 'section_title': table['section_title'],
                 'header': table['header']}
    # pd.DataFrame(data=data, columns=header)

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

    # Get the index of positive examples and negative examples from answer-node
    positive_index = [node[1] for node in data['answer-node']]
    positive_row_index = list(set([node[1][0] for node in data['answer-node']]))

    # Extract the passages for each line of the table, a list of str
    # max_passages = 3
    passages = []
    # all_raw_passages = []
    all_gt_passages = []
    passages_index = []
    passages_node_index = []
    table_passages = []

    for _i, (x, y) in enumerate(positive_index):
        if data['answer-node'][_i][2]:
            table_passages.append({'index': data['answer-node'][_i][2], 'position': [x, y],
                                   'passage': requested_documents[data['answer-node'][_i][2]]})
    for i, row in enumerate(contents):
        # 如果answer node中有这一行，
        if i in positive_row_index:
            _index = list(set([data['answer-node'][_i][2] for _i, (x, y) in enumerate(positive_index) if x == i]))
            if None in _index:
                _index.remove(None)
            gt_passages = [{'passage_id': index, 'passage': requested_documents[index]} for index in _index]
            all_gt_passages.append(gt_passages)
            # use retrived link to augment ground-truth table block
            # linked_passages = tbid2docs['{}_{}'.format(table_id, i)]
            try:
                linked_passages = [{'passage': item['text'], 'passage_id': item['id']} for item in
                                   tbid2docs['{}_{}'.format(table_id, i)]]
                joint_passages = merge_gt_linked_passages(gt_passages, linked_passages)
                ranked_passages = rank_doc_tfidf(data['question'], joint_passages)
            except Exception as e:
                logger.info(e)
                ranked_passages = rank_doc_tfidf(data['question'], gt_passages)
                # non_hit+=1
                logger.info(positive_row_index)
                logger.info('{}_{}'.format(table_id, i))
            # ranked_passages = rank_doc_tfidf(data['question'], gt_passages)
            passages.append(ranked_passages)
            passages_index.append({'all-index': _index, 'sample-index': [p['passage_id'] for p in ranked_passages]})
            passages_node_index.append([data['answer-node'][_i] for _i, (x, y) in enumerate(positive_index) if x == i])

        else:
            _index = list(set([item for sublist in position2wiki[i].values() for item in sublist]))
            raw_passages = [{'passage_id': index, 'passage': requested_documents[index]} for index in _index]
            ranked_passages = rank_doc_tfidf(data['question'], raw_passages)
            all_gt_passages.append([])
            passages.append(ranked_passages)
            passages_index.append({'all-index': _index, 'sample-index': [p['passage_id'] for p in ranked_passages]})
            passages_node_index.append([])


    # Obtain tables in pd.DataFrame
    tables = []
    for i, row in enumerate(contents):
        new_row = [' '.join(cell.split(' ')[:max_cell_tokens]) for cell in row]
        # new_header, new_row, new_max_cell = pruning_tables_with_max_cell_length(header, new_row, max_cell_tokens)
        new_header = [' '.join(cell.split(' ')[:max_header_cell_tokens]) for cell in header]
        df = pd.DataFrame(data=[new_row], columns=header)
        df = df.applymap(str)
        tables.append(df)
    raw_tables = []
    for i, row in enumerate(ori_contents):
        raw_df = pd.DataFrame(data=[row], columns=ori_header)
        raw_df = raw_df.applymap(str)
        raw_tables.append(raw_df)

    qa_data = data
    answer_node = data['answer-node']
    if len(answer_node) > 0:
        possible_passage, possible_table = 0, 0
        for answer in answer_node:
            if answer[-1] == 'passage':
                possible_passage += 1
            else:
                possible_table += 1
        # Trace back where it comes from
        if possible_passage > 0 and possible_table > 0:
            qa_data['where'] = 'both'
        elif possible_passage > 0:
            qa_data['where'] = 'passage'
        else:
            qa_data['where'] = 'table'
        qa_data['where'] = answer_node[0][-1]
    else:
        raise ValueError('wrong parsing')

    qa_data['table'] = table
    qa_data['table_metadata'] = meta_data
    qa_data['positive_passages'] = table_passages
    qa_data['passage_index'] = passages_index
    qa_data['positive_table_blocks'] = []
    gt_tb_ids = {}
    assert (len(raw_tables) == len(ori_contents))
    assert (len(tables) == len(contents))
    assert (len(passages) == len(contents))
    for i, row in enumerate(contents):
        if i in positive_row_index:
            context = convert_tb_to_string_metadata(raw_tables[i], [p['passage'] for p in passages[i]], meta_data)
            # context = convert_tb_to_string_metadata(tables[i], passages[i], meta_data)
            qa_data['positive_table_blocks'].append({'table_id': table_id,
                                                     'row_id': i,
                                                     'table_segment': raw_tables[i].to_dict(),
                                                     'passages': passages[i],
                                                     'gt_passages': all_gt_passages[i],
                                                     'context': context})
            orig_answer = data['answer-text']
            start = context.lower().find(orig_answer.lower())  # TODO start means string position or token position?
            gt_tb_ids['{}-{}'.format(table_id, i)] = {'context': context, 'passages': passages[i]}
            # randomly select a answer if there exists multiple
            #         start = random.choice(all_start_pos)
            if start == -1:
                import pdb
                pdb.set_trace()
                while context[start].lower() != orig_answer[0].lower():
                    start -= 1

    # searched table blocks
    TOPN = [item for item in list(retrieval_output.keys()) if 'top_' in item][0]
    qa_data['retrieved_tbs'] = []
    assert retrieval_output['question_id'] == data['question_id']
    searched_tbs = retrieval_output[TOPN][:args.topk_tbs]
    hit = 0
    for block in searched_tbs:
        header = block['table'][0]
        table = block['table'][1]
        table_segment = pd.DataFrame(data=table, columns=header)
        if '{}-{}'.format(block['table_id'], block['row_id']) in gt_tb_ids.keys():
            hit = 1
        # obtain tb id to passages
        try:
            passages = tbid2docs['{}_{}'.format(block['table_id'], block['row_id'])]
        except Exception as e:
            context = convert_tb_to_string_metadata(table_segment, [''], block['meta_data'])
            qa_data['retrieved_tbs'].append({
                'table_id': block['table_id'], 'row_id': block['row_id'],
                'table': block['table'], 'passages': [],
                'passage_index': [], 'context': context
            })
            # print('{}_{}'.format(block['table_id'], block['row_id']))
            continue
        # print('{}_{}'.format(block['table_id'], block['row_id']))
        # print(pids)

        find_passages = [{'passage': item['text'], 'index': item['id']} for item in passages]

        # print(find_passages)
        # input()
        if not passages:
            context = convert_tb_to_string_metadata(table_segment, [''], block['meta_data'])
            qa_data['retrieved_tbs'].append({
                'table_id': block['table_id'], 'row_id': block['row_id'],
                'table': block['table'], 'passages': [],
                'passage_index': [], 'context': context
            })
            continue
        ranked_passages = rank_doc_tfidf(data['question'], find_passages)
        ranked_raw_passages = [doc['passage'] for doc in ranked_passages]
        context = convert_tb_to_string_metadata(table_segment, ranked_raw_passages, block['meta_data'])

        # block['context'] = context
        if '{}-{}'.format(block['table_id'], block['row_id']) in gt_tb_ids.keys():
            # hit.add(d['question_id'])
            gt_context = gt_tb_ids['{}-{}'.format(block['table_id'], block['row_id'])]['context']
            output_compare.append({'searched': context, 'ground_truth': gt_context, 'answer': data['answer-text'],
                                   'id': '{}-{}'.format(block['table_id'], block['row_id']),
                                   'find_passages': ranked_passages,
                                   'gt_passages': gt_tb_ids['{}-{}'.format(block['table_id'], block['row_id'])][
                                       'passages']})
        qa_data['retrieved_tbs'].append({
            'table_id': block['table_id'], 'row_id': block['row_id'],
            'table': block['table'], 'passages': ranked_passages,
            # 'passage_index':ranked_passages,
            'context': context
        })

    # check_file = open('/home/t-wzhong/v-wanzho/ODQA/qa_log/dev_gt_retrieve_compare.json', 'w', encoding='utf8')
    # json.dump(output_compare,check_file,indent=4)
    # print('non_hit',non_hit/len(positive_row_index))
    return {'qa_data': qa_data, 'output_compare': output_compare, 'hit': hit}
    # return processed_data, qa_data


def prepare_test_data(ziped_data, max_passages=3, max_tokens=360, max_cell_tokens=36, max_header_cell_tokens=20):
    # load data
    data, retrieval_output = ziped_data[0], ziped_data[1]

    # searched table blocks
    qa_data = data
    TOPN = [item for item in list(retrieval_output.keys()) if 'top_' in item][0]
    qa_data['retrieved_tbs'] = []
    assert retrieval_output['question_id'] == data['question_id']
    searched_tbs = retrieval_output[TOPN][:args.topk_tbs]
    # hit = 0
    for block in searched_tbs:
        header = block['table'][0]
        table = block['table'][1]
        table_segment = pd.DataFrame(data=table, columns=header)
        # if '{}-{}'.format(block['table_id'], block['row_id']) in gt_tb_ids.keys():
        #     hit = 1
        # obtain tb id to passages
        try:
            passages = tbid2docs['{}_{}'.format(block['table_id'], block['row_id'])]
        except Exception as e:
            context = convert_tb_to_string_metadata(table_segment, [''], block['meta_data'])
            qa_data['retrieved_tbs'].append({
                'table_id': block['table_id'], 'row_id': block['row_id'],
                'table': block['table'], 'passages': [],
                'passage_index': [], 'context': context
            })
            # print('{}_{}'.format(block['table_id'], block['row_id']))
            continue
        # print('{}_{}'.format(block['table_id'], block['row_id']))
        # print(pids)

        find_passages = [{'passage': item['text'], 'index': item['id']} for item in passages]

        if not passages:
            context = convert_tb_to_string_metadata(table_segment, [''], block['meta_data'])
            qa_data['retrieved_tbs'].append({
                'table_id': block['table_id'], 'row_id': block['row_id'],
                'table': block['table'], 'passages': [],
                'passage_index': [], 'context': context
            })
            continue
        ranked_passages = rank_doc_tfidf(data['question'], find_passages)
        ranked_raw_passages = [doc['passage'] for doc in ranked_passages]
        context = convert_tb_to_string_metadata(table_segment, ranked_raw_passages, block['meta_data'])

        qa_data['retrieved_tbs'].append({
            'table_id': block['table_id'], 'row_id': block['row_id'],
            'table': block['table'], 'passages': ranked_passages,
            'context': context
        })

    return {'qa_data': qa_data, 'output_compare': [], 'hit': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--split', required=True, type=str)
    parser.add_argument('--split', default='train', type=str, choices=['train', 'dev', 'test'])
    parser.add_argument('--nega', default='intable', type=str, choices=['flat', 'intable', ])
    # parser.add_argument('--task', default='ottqa-train', type=str, choices=['ottqa-train', 'ottqa-dev', 'ottqa-test'])
    parser.add_argument('--max_passages', default=3, type=int)
    parser.add_argument('--max_tokens', default=360, type=int)
    parser.add_argument('--max_cell_tokens', default=50, type=int)
    parser.add_argument('--max_header_cell_tokens', default=20, type=int)
    parser.add_argument('--run_id', default=1, type=int)
    parser.add_argument('--topk_tbs', default=15, type=int)
    # parser.add_argument('--use_gpt_link_passages',action='store_true')
    parser.add_argument('--reprocess',action='store_true')
    parser.add_argument("--link_file", type=str, default='ori',
                        choices=['ori', 'gpt_only', 'blink_only', 'ori_gpt', 'blink_gpt'])
    parser.add_argument("--qa_save_path", type=str,
                        default=f'{basic_dir}/preprocessed_data/qa/train_preprocessed_gtmodify.json', )
    parser.add_argument("--retrieval_results_file", type=str,
                        default=r'{basic_dir}/preprocessed_data/qa/data4qa_v2/dev_output_k100.json',
                        help="path that store the results of table block retrieval")
    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    n_threads = 24
    if args.split in ['train', 'dev']:
        logger.info("using {}".format(args.split))
        table_path = 'traindev_tables_tok'
        request_path = 'traindev_request_tok'

        if not os.path.exists(args.qa_save_path) or args.reprocess:
            with open(f'{basic_dir}/data_ottqa/{args.split}.traced.json', 'r') as f:
                data = json.load(f)
            retrieval_outputs = read_jsonl(args.retrieval_results_file)

            '''
            for i in range(8):
                # with open(
                #         '/home/t-wzhong/v-wanzho/ODQA/OTT-QA/link_generator/row_passage_query.json-0000{}-of-00008'.format(
                #                 i), 'r', encoding='utf8') as inf:
                tmp_data = read_jsonl('/home/t-wzhong/v-wanzho/ODQA/OTT-QA/link_generator/row_passage_query.json-0000{}-of-00008'.format(i))
                for line in tmp_data:
                    tbid2docs[line[0]] = ['/wiki/' + item.replace(' ', '_') for item in line[1]]
            '''
            tmp_data = read_jsonl(f'{basic_dir}/data_wikitable/tfidf_augmentation_results.json')
            tbid2docs = {}
            for line in tmp_data:
                tbid2docs[line[0]] = line[1]
            logger.info('length of the table id 2 documents： {}'.format(len(tbid2docs.keys())))

            results = []
            output_compare = []
            running_function = prepare_training_data_random_nega
            zip_data = list(zip(data, retrieval_outputs))
            with Pool(n_threads) as p:
                func_ = partial(running_function, table_path=table_path, request_path=request_path,
                                max_passages=args.max_passages, max_tokens=args.max_tokens,
                                max_cell_tokens=args.max_cell_tokens,
                                max_header_cell_tokens=args.max_header_cell_tokens)
                all_results = list(tqdm(p.imap(func_, zip_data, chunksize=16), total=len(zip_data),
                                        desc="convert examples to trainable data", ))
                qa_results = [res['qa_data'] for res in all_results]
                hits = [res['hit'] for res in all_results]
                logger.info("{}, {};".format(len(all_results), len(qa_results)))

            logger.info(sum(hits) / len(data))
            # random.shuffle(qa_results)
            with open(args.qa_save_path, 'w') as f:
                json.dump(qa_results, f, indent=1)
            logger.info('Saving the output to {}'.format(args.qa_save_path))

        else:
            logger.info("file already exists: {}".format(args.qa_save_path))

    elif args.split in ['test']:
        logger.info("using {}".format(args.split))

        if not os.path.exists(args.qa_save_path) or args.reprocess:
            with open(f'{basic_dir}/data_ottqa/test.blind.json', 'r') as f:
                data = json.load(f)
            retrieval_outputs = read_jsonl(args.retrieval_results_file)

            tmp_data = read_jsonl(f'{basic_dir}/data_wikitable/tfidf_augmentation_results.json')
            tbid2docs = {}
            for line in tmp_data:
                tbid2docs[line[0]] = line[1]
            logger.info("length of the table id 2 documents: {}".format(len(tbid2docs.keys())))

            results = []
            running_function = prepare_test_data
            zip_data = list(zip(data, retrieval_outputs))
            with Pool(n_threads) as p:
                func_ = partial(running_function,
                                max_passages=args.max_passages, max_tokens=args.max_tokens,
                                max_cell_tokens=args.max_cell_tokens,
                                max_header_cell_tokens=args.max_header_cell_tokens)
                all_results = list(tqdm(p.imap(func_, zip_data, chunksize=16), total=len(zip_data),
                                        desc="convert examples to trainable data", ))
                qa_results = [res['qa_data'] for res in all_results]
                hits = [res['hit'] for res in all_results]
                logger.info("{}, {};".format(len(all_results), len(qa_results)))

            logger.info(sum(hits) / len(data))
            # random.shuffle(qa_results)
            with open(args.qa_save_path, 'w') as f:
                json.dump(qa_results, f, indent=1)
            logger.info('Saving the output to {}'.format(args.qa_save_path))




    else:
        raise NotImplementedError