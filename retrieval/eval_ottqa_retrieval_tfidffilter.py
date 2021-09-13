"""
Evaluating trained retrieval model.

Usage:
python eval_ottqa_retrieval.py ${EVAL_DATA} ${CORPUS_VECTOR_PATH} ${id2doc} ${MODEL_CHECKPOINT} \
     --batch-size 50 \
     --beam-size-1 20 \
     --beam-size-2 5 \
     --topk 20 \
     --shared-encoder \
     --gpu \
     --save-path ${PATH_TO_SAVE_RETRIEVAL}

"""
import sys

sys.path.append('../')
import argparse
import collections
import json
import pickle
import logging
import os
import time

import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer
from multiprocessing import Pool, cpu_count
from functools import partial

# from retrieval.models.mhop_retriever import RobertaRetriever
from retrieval.utils.basic_tokenizer import SimpleTokenizer
from retrieval.utils.utils import (load_saved, move_to_cuda, para_has_answer, found_table)
from retrieval.utils.utils import whitening

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if (logger.hasHandlers()):
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)


def convert_hnsw_query(query_vectors):
    aux_dim = np.zeros(len(query_vectors), dtype='float32')
    query_nhsw_vectors = np.hstack((query_vectors, aux_dim.reshape(-1, 1)))
    return query_nhsw_vectors

def convert_hnsw_query(query_vectors):
    aux_dim = np.zeros(len(query_vectors), dtype='float32')
    query_nhsw_vectors = np.hstack((query_vectors, aux_dim.reshape(-1, 1)))
    return query_nhsw_vectors

def retrieval_search(eval_data, args, logger, split):
    simple_tokenizer = SimpleTokenizer()
    ds_items = eval_data
    logger.info("Building index...")
    if args.three_cat:
        d = 768 * 3
    else:
        d = 768
    xb = np.load(args['corpus_embeddings_path']).astype('float32')
    logger.info("corpus size: {}".format(xb.shape))
    if args['hnsw']:
        if os.path.exists(args['faiss_save_path']):
            # index = faiss.read_index("index/ottqa_index_hnsw.index")
            index = faiss.read_index(args['faiss_save_path'])
        else:
            index = faiss.IndexHNSWFlat(d + 1, 512)
            index.hnsw.efSearch = 128
            index.hnsw.efConstruction = 200
            phi = 0
            for i, vector in enumerate(xb):
                norms = (vector ** 2).sum()
                phi = max(phi, norms)
            logger.info('HNSWF DotProduct -> L2 space phi={}'.format(phi))

            data = xb
            buffer_size = 50000
            n = len(data)
            logger.info(n)
            for i in tqdm(range(0, n, buffer_size)):
                vectors = [np.reshape(t, (1, -1)) for t in data[i:i + buffer_size]]
                norms = [(doc_vector ** 2).sum() for doc_vector in vectors]
                aux_dims = [np.sqrt(phi - norm) for norm in norms]
                hnsw_vectors = [np.hstack((doc_vector, aux_dims[idx].reshape(-1, 1))) for idx, doc_vector in
                                enumerate(vectors)]
                hnsw_vectors = np.concatenate(hnsw_vectors, axis=0)
                index.add(hnsw_vectors)
    else:
        if os.path.exists(args['faiss_save_path']):
            # index = faiss.read_index("index/ottqa_index_hnsw.index")
            index = faiss.read_index(args['faiss_save_path'])
        else:
            index = faiss.IndexFlatIP(d)
            index.add(xb)
            if args['gpu']:
                # ngpus = faiss.get_num_gpus()
                # index = faiss.index_cpu_to_all_gpus(index)
                res = faiss.StandardGpuResources()
                # res.setTempMemory(512 * 1024 * 1024)
                # index = faiss.index_cpu_to_gpu(res, 1, index)
                index = faiss.index_cpu_to_all_gpus(index)
            logger.info("Finish Building Index with IndexFlatIP")
    if not os.path.exists(args['faiss_save_path']):
        faiss.write_index(index, args['faiss_save_path'])

    logger.info(f"Loading corpus...")
    id2doc = json.load(open(args['id2doc_path']))
    if isinstance(id2doc["0"], list):
        id2doc = {k: {"title": v[0], "text": v[1]} for k, v in id2doc.items()}
    # title2text = {v[0]:v[1] for v in id2doc.values()}
    logger.info(f"Corpus size {len(id2doc)}")

    logger.info("Loading question embeddings from {}".format(args['query_embeddings_path']))
    questions = [_["question"][:-1] if _["question"].endswith("?") else _["question"] for _ in ds_items]
    metrics = []
    # metrics_eval_table_id = []
    retrieval_outputs = []
    query_embeddings = np.load(args['query_embeddings_path']).astype('float32')

    def searching(idx):
        # for idx, q_embeds_numpy in enumerate(tqdm(query_embeddings, desc='Processing: ')):
        q_embeds_numpy = np.expand_dims(query_embeddings[idx], 0)
        if args['hnsw']:
            q_embeds_numpy = convert_hnsw_query(q_embeds_numpy)
        D, I = index.search(q_embeds_numpy, args['beam_size'])

        b_idx = 0
        metric_i = {}
        output_i = {}
        topk_tbs = []
        for _, tb_id in enumerate(I[b_idx]):
            tb = id2doc[str(tb_id)]
            topk_tbs.append(tb)
        if args['eval_only_ans']:
            gold_answers = ds_items[idx]["answer-text"]
            metric_i = {
                "question": ds_items[idx]["question"],
                'table_recall': int(found_table(ds_items[idx]['table_id'], topk_tbs)),
                "ans_recall": int(para_has_answer(gold_answers, topk_tbs, simple_tokenizer)),
                "type": ds_items[idx].get("type", "single")
            }
        if args['output_save_path'] != "":
            output_i = {"question_id": ds_items[idx]["question_id"],
                        "question": ds_items[idx]["question"],
                        "top_{}".format(args['beam_size']): topk_tbs, }
            if 'answer-text' in ds_items[idx]:
                output_i['answer-text'] = ds_items[idx]['answer-text']
            if 'table_id' in ds_items[idx]:
                output_i['table_id'] = ds_items[idx]['table_id']
        return metric_i, output_i

    # n_threads = 24
    with Pool(args.n_threads) as p:
        # func_ = partial(searching)
        results = list(
            tqdm(p.imap(searching, range(len(query_embeddings)), chunksize=16), total=len(query_embeddings),
                 desc="Searching: ", ))
    metrics, retrieval_outputs = [item[0] for item in results], [item[1] for item in results]
    if args['output_save_path'] != "":
        with open(args['output_save_path'], "w") as out:
            for l in retrieval_outputs:
                out.write(json.dumps(l) + "\n")
        logger.info("Saving outputs to {}".format(args['output_save_path']))

    if split == 'dev':
        def get_recall(answers, preds, n=5):
            truth_table = []
            for idx, ans in enumerate(answers):
                truth_table.append(any([ans == inst for inst in preds[idx][:n]]))
            return sum(truth_table) / len(truth_table), sum(truth_table), len(truth_table)

        table_id_gold = [inst['table_id'] for inst in retrieval_outputs]
        table_id_preds = [[item['table_id'] for item in inst['top_100']] for inst in retrieval_outputs]
        r, t, a = get_recall(table_id_gold, table_id_preds, 1)
        logger.info("Table Recall @1: {}, {}/{}".format(r, t, a))
        r, t, a = get_recall(table_id_gold, table_id_preds, 5)
        logger.info("Table Recall @5: {}, {}/{}".format(r, t, a))
        r, t, a = get_recall(table_id_gold, table_id_preds, 10)
        logger.info("Table Recall @10: {}, {}/{}".format(r, t, a))
        r, t, a = get_recall(table_id_gold, table_id_preds, 20)
        logger.info("Table Recall @20: {}, {}/{}".format(r, t, a))
        r, t, a = get_recall(table_id_gold, table_id_preds, 50)
        logger.info("Table Recall @50: {}, {}/{}".format(r, t, a))
        r, t, a = get_recall(table_id_gold, table_id_preds, 100)
        logger.info("Table Recall @100: {}, {}/{}".format(r, t, a))

        if args.eval_only_ans:
            logger.info(f"Evaluating {len(metrics)} samples...")
            type2items = collections.defaultdict(list)
            for item in metrics:
                type2items[item["type"]].append(item)
            logger.info(f'Ans Recall: {np.mean([m["ans_recall"] for m in metrics])}')
            logger.info(f'Table Recall: {np.mean([m["table_recall"] for m in metrics])}')
            for t in type2items.keys():
                logger.info(f"{t} Questions num: {len(type2items[t])}")
                logger.info(f'Ans Recall: {np.mean([m["ans_recall"] for m in type2items[t]])}')
    return retrieval_outputs

def retrieval_search_tfidf(eval_data,args,logger,split):
    ds_items = eval_data
    # filter
    # if args.eval_only_ans:
    #     ds_items = [_ for _ in ds_items if _["answer"][0] not in ["yes", "no"]]
    if args.three_cat:
        d = 768 * 3
    else:
        d = 768
    simple_tokenizer = SimpleTokenizer()
    if os.path.exists(args['qid2tbemb_save_path']):
        with open(args.qid2tbemb_save_path, 'rb') as f:
            qid2info = pickle.load(f)
            qid2tb, qid2emb = qid2info['table_blocks'], qid2info['table_embs']
            logger.info(f"Loading question id to table block, emb mapping to {args['qid2tbemb_save_path']}")
    else:
        logger.info("Building index...")
        xb = np.load(args['corpus_embeddings_path']).astype('float32')
        logger.info("corpus size: {}".format(xb.shape))
        logger.info("Loading corpus from {}".format(args['id2doc_path']))
        id2doc = json.load(open(args['id2doc_path']))
        logger.info(f"Documents corpus size {len(id2doc)}")
        if args['tfidf_result_file'] and os.path.exists(args['tfidf_result_file']):
            with open(args['tfidf_result_file'], 'r', encoding='utf8') as f:
                tfidf_qid2docs = json.load(f)
        # title2text = {v[0]:v[1] for v in id2doc.values()}
        qid2emb, qid2tb = {}, {}
        all_counts = 0
        logger.info(f"Buiding mapping from question to doc...")
        for qid, tfidf_docs in tqdm(tfidf_qid2docs.items()):
            qid2emb[qid], qid2tb[qid] = [], {}
            count = 0
            # if len(qid2emb.keys())>5:
            #     break
            for idx, doc in id2doc.items():
                if doc['table_id'] in tfidf_docs['doc_ids'][:args['topk_doc']]:
                    qid2tb[str(qid)][str(len(qid2emb[qid]))] = doc
                    qid2emb[str(qid)].append(xb[count, :])
                    all_counts += 1
                count += 1
        with open(args['qid2tbemb_save_path'], 'wb') as f:
            qid2info = {'table_blocks': qid2tb, 'table_embs': qid2emb}
            pickle.dump(qid2info, f)
            logger.info(f"Saving question id to table block, emb mapping to {args['qid2tbemb_save_path']}")
        logger.info(f"Corpus size {all_counts}")

    logger.info("Loading question embeddings from {}".format(args['query_embeddings_path']))
    questions = [_["question"][:-1] if _["question"].endswith("?") else _["question"] for _ in ds_items]
    for qid, tbs in qid2tb.items():
        assert (len(tbs.keys()) != 0), '{}'.format(qid)
    metrics = []
    # metrics_eval_table_id = []
    retrieval_outputs = []
    query_embeddings = np.load(args['query_embeddings_path']).astype('float32')

    def searching(idx):
        # for idx, q_embeds_numpy in enumerate(tqdm(query_embeddings, desc='Processing: ')):
        q_embeds_numpy = np.expand_dims(query_embeddings[idx], 0)
        if args['hnsw']:
            q_embeds_numpy = convert_hnsw_query(q_embeds_numpy)

        qid = ds_items[idx]['question_id']
        index = faiss.IndexFlatIP(d)
        index.add(np.array(qid2emb[qid]))
        if args['gpu']:
            # ngpus = faiss.get_num_gpus()
            # index = faiss.index_cpu_to_all_gpus(index)
            res = faiss.StandardGpuResources()
            # res.setTempMemory(512 * 1024 * 1024)
            index = faiss.index_cpu_to_gpu(res, 1, index)
            # index = faiss.index_cpu_to_all_gpus(index)
        D, I = index.search(q_embeds_numpy, args['beam_size'])

        b_idx = 0
        metric_i = {}
        output_i = {}
        topk_tbs = []
        for _, tb_id in enumerate(I[b_idx]):
            if str(tb_id) == '-1':
                print(I[b_idx])
                continue
            tb = qid2tb[qid][str(tb_id)]
            topk_tbs.append(tb)
        if args['eval_only_ans']:
            gold_answers = ds_items[idx]["answer-text"]
            metric_i = {
                "question": ds_items[idx]["question"],
                'table_recall': int(found_table(ds_items[idx]['table_id'], topk_tbs)),
                "ans_recall": int(para_has_answer(gold_answers, topk_tbs, simple_tokenizer)),
                "type": ds_items[idx].get("type", "single")
            }
        if args['output_save_path'] != "":
            output_i = {"question_id": ds_items[idx]["question_id"],
                        "question": ds_items[idx]["question"],
                        "top_{}".format(args['beam_size']): topk_tbs, }
            if 'answer-text' in ds_items[idx]:
                output_i['answer-text'] = ds_items[idx]['answer-text']
            if 'table_id' in ds_items[idx]:
                output_i['table_id'] = ds_items[idx]['table_id']
        return metric_i, output_i

    # n_threads = 24
    with Pool(args.n_threads) as p:
        # func_ = partial(searching)
        results = list(
            tqdm(p.imap(searching, range(len(query_embeddings)), chunksize=16), total=len(query_embeddings),
                 desc="Searching: ", ))
        # results = list(
        #         tqdm(p.imap(searching, range(5), chunksize=16), total=5, desc="Searching: ", ))
    metrics, retrieval_outputs = [item[0] for item in results], [item[1] for item in results]


    if args['output_save_path'] != "":
        with open(args['output_save_path'], "w") as out:
            for l in retrieval_outputs:
                out.write(json.dumps(l) + "\n")
        logger.info("Saving outputs to {}".format(args['output_save_path']))

    if split == 'dev':
        def get_recall(answers, preds, n=5):
            truth_table = []
            for idx, ans in enumerate(answers):
                truth_table.append(any([ans == inst for inst in preds[idx][:n]]))
            return sum(truth_table) / len(truth_table), sum(truth_table), len(truth_table)

        table_id_gold = [inst['table_id'] for inst in retrieval_outputs]
        table_id_preds = [[item['table_id'] for item in inst['top_100']] for inst in retrieval_outputs]
        r, t, a = get_recall(table_id_gold, table_id_preds, 1)
        logger.info("Table Recall @1: {}, {}/{}".format(r, t, a))
        r, t, a = get_recall(table_id_gold, table_id_preds, 5)
        logger.info("Table Recall @5: {}, {}/{}".format(r, t, a))
        r, t, a = get_recall(table_id_gold, table_id_preds, 10)
        logger.info("Table Recall @10: {}, {}/{}".format(r, t, a))
        r, t, a = get_recall(table_id_gold, table_id_preds, 20)
        logger.info("Table Recall @20: {}, {}/{}".format(r, t, a))
        r, t, a = get_recall(table_id_gold, table_id_preds, 50)
        logger.info("Table Recall @50: {}, {}/{}".format(r, t, a))
        r, t, a = get_recall(table_id_gold, table_id_preds, 100)
        logger.info("Table Recall @100: {}, {}/{}".format(r, t, a))

        if args['eval_only_ans']:
            logger.info(f"Evaluating {len(metrics)} samples...")
            type2items = collections.defaultdict(list)
            for item in metrics:
                type2items[item["type"]].append(item)
            logger.info(f'Ans Recall: {np.mean([m["ans_recall"] for m in metrics])}')
            logger.info(f'Table Recall: {np.mean([m["table_recall"] for m in metrics])}')
            for t in type2items.keys():
                logger.info(f"{t} Questions num: {len(type2items[t])}")
                logger.info(f'Ans Recall: {np.mean([m["ans_recall"] for m in type2items[t]])}')
    return retrieval_outputs

def calculate_score(retrieval_outputs):
    def get_recall(answers, preds, n=5):
        truth_table = []
        for idx, ans in enumerate(answers):
            truth_table.append(any([ans == inst for inst in preds[idx][:n]]))
        return sum(truth_table) / len(truth_table), sum(truth_table), len(truth_table)

    table_id_gold = [inst['table_id'] for inst in retrieval_outputs]
    table_id_preds = [[item['table_id'] for item in inst['top_100']] for inst in retrieval_outputs]
    r, t, a = get_recall(table_id_gold, table_id_preds, 1)
    print("Table Recall @1: {}, {}/{}".format(r, t, a))
    r, t, a = get_recall(table_id_gold, table_id_preds, 5)
    print("Table Recall @5: {}, {}/{}".format(r, t, a))
    r, t, a = get_recall(table_id_gold, table_id_preds, 10)
    print("Table Recall @10: {}, {}/{}".format(r, t, a))
    r, t, a = get_recall(table_id_gold, table_id_preds, 20)
    print("Table Recall @20: {}, {}/{}".format(r, t, a))
    r, t, a = get_recall(table_id_gold, table_id_preds, 50)
    print("Table Recall @50: {}, {}/{}".format(r, t, a))
    r, t, a = get_recall(table_id_gold, table_id_preds, 100)
    print("Table Recall @100: {}, {}/{}".format(r, t, a))

def evaluate_retrieval_results(files, mode):
    if mode=='from_files':
        for file in files:
            retrieval_outputs = common.load_jsonl(file)
            print('Evaluate {}'.format(file))
            calculate_score(retrieval_outputs)
    else:
        retrieval_outputs = files
        calculate_score(retrieval_outputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_path', type=str, default=None)
    parser.add_argument('--query_embeddings_path', type=str, default=None)
    parser.add_argument('--corpus_embeddings_path', type=str, default=None)
    parser.add_argument('--id2doc_path', type=str, default=None)
    parser.add_argument('--faiss_save_path', type=str, default="data/ottqa_index/ottqa_index_tapas")
    parser.add_argument("--output_save_path", type=str, default="")
    parser.add_argument("--qid2tbemb_save_path", type=str, default="")
    # parser.add_argument('--model_path', type=str, default=None)

    # parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    # parser.add_argument("--do_test", action="store_true", help="Whether to run eval on the test set.")

    # parser.add_argument('--topk', type=int, default=5, help="topk paths")
    parser.add_argument('--num_workers', type=int, default=20)
    parser.add_argument('--topk_doc',type=int,default=20)
    parser.add_argument('--tfidf_result_file', type=str, default=None)
    # parser.add_argument('--max_q_len', type=int, default=70)
    # parser.add_argument('--max_c_len', type=int, default=512)
    # parser.add_argument('--max_q_sp_len', type=int, default=350)
    # parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--beam_size', type=int, default=100)
    parser.add_argument('--n_threads', type=int, default=24)
    # parser.add_argument('--model_name', type=str, default='roberta-base')
    parser.add_argument('--three_cat', action="store_true")
    parser.add_argument('--whitening', action="store_true")
    parser.add_argument('--gpu', action="store_true")
    parser.add_argument('--save_index', action="store_true")

    parser.add_argument('--eval_only_ans', action="store_true")
    parser.add_argument('--eval_table_id', action="store_true")
    # parser.add_argument('--shared_encoder', action="store_true")
    # parser.add_argument("--stop_drop", default=0, type=float)
    parser.add_argument('--hnsw', action="store_true")
    args = parser.parse_args()

    logger.info("Loading data...")
    ds_items = json.load(open(args.raw_data_path, 'r', encoding='utf8'))
    # filter
    # if args.eval_only_ans:
    #     ds_items = [_ for _ in ds_items if _["answer"][0] not in ["yes", "no"]]
    if args.three_cat:
        d = 768 * 3
    else:
        d = 768
    simple_tokenizer = SimpleTokenizer()
    if os.path.exists(args.qid2tbemb_save_path):
        with open(args.qid2tbemb_save_path,'rb') as f:
            qid2info = pickle.load(f)
            qid2tb, qid2emb = qid2info['table_blocks'], qid2info['table_embs']
            logger.info(f"Loading question id to table block, emb mapping to {args.qid2tbemb_save_path}")
    else:
        logger.info("Building index...")
        xb = np.load(args.corpus_embeddings_path).astype('float32')
        logger.info("corpus size: {}".format(xb.shape))
        if args.whitening:
            logger.info("Start doing PCA Whitening...")
            logger.info("Loading question embeddings from {}".format(args.query_embeddings_path))
            questions = [_["question"][:-1] if _["question"].endswith("?") else _["question"] for _ in ds_items]
            metrics = []
            # metrics_eval_table_id = []
            retrieval_outputs = []
            query_embeddings = np.load(args.query_embeddings_path).astype('float32')
            query_embeddings, xb = whitening(query_embeddings, xb)
            logger.info("Finish doing PCA Whitening...")

        logger.info(f"Loading corpus...")
        # id2doc = json.load(open(args.id2doc_path))
        with open(args.id2doc_path, 'r', encoding='utf8') as f:
            id2doc = json.load(f)
        logger.info(f"Documents corpus size {len(id2doc)}")
        if args.tfidf_result_file:
            with open(args.tfidf_result_file, 'r', encoding='utf8') as f:
                tfidf_qid2docs = json.load(f)
        # # title2text = {v[0]:v[1] for v in id2doc.values()}
        # qid2emb, qid2tb = {},{}
        # all_counts = 0
        # logger.info(f"Buiding mapping from question to doc...")
        # for qid, tfidf_docs in tqdm(tfidf_qid2docs.items()):
        #     qid2emb[qid], qid2tb[qid] = [],{}
        #     count = 0
        #     # if len(qid2emb.keys())>5:
        #     #     break
        #     for idx, doc in id2doc.items():
        #         if doc['table_id'] in tfidf_docs['doc_ids'][:args.topk_doc]:
        #             qid2tb[str(qid)][str(len(qid2emb[qid]))] = doc
        #             qid2emb[str(qid)].append(xb[count,:])
        #             all_counts+=1
        #         count += 1

        qid2emb = {key: [] for key in tfidf_qid2docs.keys()}
        qid2tb = {key: {} for key in tfidf_qid2docs.keys()}
        # id2doc_keys = {v['table_id']:k for k,v in id2doc.items()}
        # id2doc_keys_list = [(v['table_id'], k) for k,v in id2doc.items()]  # id2doc里的key必须是从0开始按大小排列的数字字符！
        all_counts = 0
        logger.info(f"Buiding mapping from question to doc...")
        def mapping(qt):
            qid, tfidf_docs = qt[0], qt[1]
            topk_docs = set(tfidf_docs['doc_ids'][:args.topk_doc])
            within_docs = [(idx, doc) for idx, doc in id2doc.items() if doc['table_id'] in topk_docs]
            emb = [xb[int(inst[0]),:] for range_index, inst in enumerate(within_docs)]
            tb = {str(range_index):inst[1] for range_index, inst in enumerate(within_docs)}
            assert len(emb) == len(tb)
            count = len(emb)

            # topk_docs = [key for key in tfidf_docs['doc_ids'][:args.topk_doc] if key in id2doc_keys]
            # embed_idx = [for ]
            # embed_idx = [int(id2doc_keys[tfidf_doc_id]) for tfidf_doc_id in topk_docs]
            # emb = [xb[idx, :] for idx in embed_idx]
            # tb = {str(idx):id2doc[id2doc_keys[key]] for idx, key in enumerate(topk_docs)}
            # count = len(emb)

            return {'qid':qid, 'table_blocks': tb, 'table_embeds': emb, 'count': count}
        # n_threads = 24
        logger.info(f"Buiding ...")
        with Pool(args.n_threads) as p:
            # func_ = partial(searching)
            results = list(tqdm(p.imap(mapping, tfidf_qid2docs.items(), chunksize=16), total=len(tfidf_qid2docs),
                                desc="Mapping with multiprocessing: ", ))
        qid2tb = {res['qid']:res['table_blocks'] for res in results}
        qid2emb = {res['qid']:res['table_embeds'] for res in results}
        all_counts = sum([res['count'] for res in results])

        with open(args.qid2tbemb_save_path, 'wb') as f:
            qid2info = {'table_blocks': qid2tb,'table_embs':qid2emb}
            pickle.dump(qid2info,f)
            logger.info(f"Saving question id to table block, emb mapping to {args.qid2tbemb_save_path}")
        logger.info(f"Corpus size {all_counts}")
        del id2doc

    if not args.whitening:
        logger.info("Loading question embeddings from {}".format(args.query_embeddings_path))
        questions = [_["question"][:-1] if _["question"].endswith("?") else _["question"] for _ in ds_items]
        for qid,tbs in qid2tb.items():
            assert(len(tbs.keys())!=0),'{}'.format(qid)
        metrics = []
        # metrics_eval_table_id = []
        retrieval_outputs = []
        query_embeddings = np.load(args.query_embeddings_path).astype('float32')

    def searching(idx):
        # for idx, q_embeds_numpy in enumerate(tqdm(query_embeddings, desc='Processing: ')):
        q_embeds_numpy = np.expand_dims(query_embeddings[idx], 0)
        if args.hnsw:
            q_embeds_numpy = convert_hnsw_query(q_embeds_numpy)

        qid = ds_items[idx]['question_id']
        index = faiss.IndexFlatIP(d)
        index.add(np.array(qid2emb[qid]))
        if args.gpu:
            # ngpus = faiss.get_num_gpus()
            # index = faiss.index_cpu_to_all_gpus(index)
            res = faiss.StandardGpuResources()
            # res.setTempMemory(512 * 1024 * 1024)
            index = faiss.index_cpu_to_gpu(res, 1, index)
            # index = faiss.index_cpu_to_all_gpus(index)
        D, I = index.search(q_embeds_numpy, args.beam_size)

        b_idx = 0
        metric_i = {}
        output_i = {}
        topk_tbs = []
        for _, tb_id in enumerate(I[b_idx]):
            if str(tb_id)=='-1':
                print(I[b_idx])
                continue
            tb = qid2tb[qid][str(tb_id)]
            topk_tbs.append(tb)
        if args.eval_only_ans:
            gold_answers = ds_items[idx]["answer-text"]
            metric_i = {
                "question": ds_items[idx]["question"],
                'table_recall': int(found_table(ds_items[idx]['table_id'], topk_tbs)),
                "ans_recall": int(para_has_answer(gold_answers, topk_tbs, simple_tokenizer)),
                "type": ds_items[idx].get("type", "single")
            }
        if args.output_save_path != "":
            output_i = {"question_id": ds_items[idx]["question_id"],
                        "question": ds_items[idx]["question"],
                        "top_{}".format(args.beam_size): topk_tbs, }
            if 'answer-text' in ds_items[idx]:
                output_i['answer-text'] = ds_items[idx]['answer-text']
            if 'table_id' in ds_items[idx]:
                output_i['table_id'] = ds_items[idx]['table_id']
        return metric_i, output_i

    # n_threads = 24
    with Pool(args.n_threads) as p:
        # func_ = partial(searching)
        results = list(
            tqdm(p.imap(searching, range(len(query_embeddings)), chunksize=16), total=len(query_embeddings), desc="Searching: ", ))
        # results = list(
        #         tqdm(p.imap(searching, range(5), chunksize=16), total=5, desc="Searching: ", ))
    metrics, retrieval_outputs = [item[0] for item in results], [item[1] for item in results]

    # for idx, q_embeds_numpy in enumerate(tqdm(query_embeddings, desc='Searching: ')):
    #     q_embeds_numpy = np.expand_dims(q_embeds_numpy, 0)
    #     if args.hnsw:
    #         q_embeds_numpy = convert_hnsw_query(q_embeds_numpy)
    #     D, I = index.search(q_embeds_numpy, args.beam_size)
    #
    #     # for b_idx in range(bsize):
    #     b_idx = 0
    #     topk_tbs = []
    #     for _, tb_id in enumerate(I[b_idx]):
    #         tb = id2doc[str(tb_id)]
    #         topk_tbs.append(tb)
    #     if args.eval_only_ans:
    #         gold_answers = ds_items[idx]["answer-text"]
    #         metrics.append({
    #                 "question": ds_items[idx]["question"],
    #                 'table_recall': int(found_table(ds_items[idx]['table_id'], topk_tbs)),
    #                 "ans_recall": int(para_has_answer(gold_answers, topk_tbs, simple_tokenizer)),
    #                 "type": ds_items[idx].get("type", "single")
    #         })
    #     if args.output_save_path != "":
    #         output = {"question_id": ds_items[idx]["question_id"],
    #                   "question": ds_items[idx]["question"],
    #                   "top_{}".format(args.beam_size): topk_tbs, }
    #         if 'answer-text' in ds_items[idx]:
    #             output['answer-text'] = ds_items[idx]['answer-text']
    #         if 'table_id' in ds_items[idx]:
    #             output['table_id'] = ds_items[idx]['table_id']
    #         retrieval_outputs.append(output)



    if args.output_save_path != "":
        with open(args.output_save_path, "w") as out:
            for l in retrieval_outputs:
                out.write(json.dumps(l) + "\n")
        logger.info("Saving outputs to {}".format(args.output_save_path))


    def get_recall(answers, preds, n=5):
        truth_table = []
        for idx, ans in enumerate(answers):
            truth_table.append(any([ans == inst for inst in preds[idx][:n]]))
        return sum(truth_table) / len(truth_table), sum(truth_table), len(truth_table)

    table_id_gold = [inst['table_id'] for inst in retrieval_outputs]
    table_id_preds = [[item['table_id'] for item in inst['top_100']] for inst in retrieval_outputs]
    r, t, a = get_recall(table_id_gold, table_id_preds, 1)
    logger.info("Table Recall @1: {}, {}/{}".format(r, t, a))
    r, t, a = get_recall(table_id_gold, table_id_preds, 5)
    logger.info("Table Recall @5: {}, {}/{}".format(r, t, a))
    r, t, a = get_recall(table_id_gold, table_id_preds, 10)
    logger.info("Table Recall @10: {}, {}/{}".format(r, t, a))
    r, t, a = get_recall(table_id_gold, table_id_preds, 20)
    logger.info("Table Recall @20: {}, {}/{}".format(r, t, a))
    r, t, a = get_recall(table_id_gold, table_id_preds, 50)
    logger.info("Table Recall @50: {}, {}/{}".format(r, t, a))
    r, t, a = get_recall(table_id_gold, table_id_preds, 100)
    logger.info("Table Recall @100: {}, {}/{}".format(r, t, a))

    if args.eval_only_ans:
        logger.info(f"Evaluating {len(metrics)} samples...")
        type2items = collections.defaultdict(list)
        for item in metrics:
            type2items[item["type"]].append(item)
        logger.info(f'Ans Recall: {np.mean([m["ans_recall"] for m in metrics])}')
        logger.info(f'Table Recall: {np.mean([m["table_recall"] for m in metrics])}')
        for t in type2items.keys():
            logger.info(f"{t} Questions num: {len(type2items[t])}")
            logger.info(f'Ans Recall: {np.mean([m["ans_recall"] for m in type2items[t]])}')


