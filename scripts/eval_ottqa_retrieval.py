import sys

sys.path.append('../')
import argparse
import collections
import json
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_path', type=str, default=None)
    parser.add_argument('--query_embeddings_path', type=str, default=None)
    parser.add_argument('--corpus_embeddings_path', type=str, default=None)
    parser.add_argument('--id2doc_path', type=str, default=None)
    parser.add_argument('--faiss_save_path', type=str, default="data/ottqa_index/ottqa_index_tapas")
    parser.add_argument("--output_save_path", type=str, default="")

    # parser.add_argument('--topk', type=int, default=5, help="topk paths")
    parser.add_argument('--num_workers', type=int, default=20)
    parser.add_argument('--beam_size', type=int, default=100)
    parser.add_argument('--gpu', action="store_true")
    parser.add_argument('--save_index', action="store_true")
    parser.add_argument('--eval_only_ans', action="store_true")
    parser.add_argument('--eval_table_id', action="store_true")
    parser.add_argument('--hnsw', action="store_true")
    args = parser.parse_args()

    logger.info("Loading data...")
    ds_items = json.load(open(args.raw_data_path, 'r', encoding='utf8'))
    # filter
    # if args.eval_only_ans:
    #     ds_items = [_ for _ in ds_items if _["answer"][0] not in ["yes", "no"]]

    simple_tokenizer = SimpleTokenizer()

    logger.info("Building index...")
    logger.info("Loading question embeddings from {}".format(args.query_embeddings_path))
    questions = [_["question"][:-1] if _["question"].endswith("?") else _["question"] for _ in ds_items]
    metrics = []
    # metrics_eval_table_id = []
    retrieval_outputs = []
    query_embeddings = np.load(args.query_embeddings_path).astype('float32')


    logger.info("Loading corpus embeddings from {}".format(args.corpus_embeddings_path))
    d = 768 * 3
    xb = np.load(args.corpus_embeddings_path).astype('float32')
    logger.info("corpus size: {}".format(xb.shape))

    if args.hnsw:
        if os.path.exists(args.faiss_save_path):
            # index = faiss.read_index("index/ottqa_index_hnsw.index")
            index = faiss.read_index(args.faiss_save_path)
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
        if os.path.exists(args.faiss_save_path):
            index = faiss.read_index(args.faiss_save_path)
        else:
            index = faiss.IndexFlatIP(d)
            index.add(xb)
            if args.gpu:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_all_gpus(index)
            logger.info("Finish Building Index with IndexFlatIP")
    # if not os.path.exists(args.faiss_save_path):
    #     faiss.write_index(index, args.faiss_save_path)

    logger.info(f"Loading corpus...")
    id2doc = json.load(open(args.id2doc_path))
    if isinstance(id2doc["0"], list):
        id2doc = {k: {"title": v[0], "text": v[1]} for k, v in id2doc.items()}
    # title2text = {v[0]:v[1] for v in id2doc.values()}
    logger.info(f"Corpus size {len(id2doc)}")

    def searching(idx):
        # for idx, q_embeds_numpy in enumerate(tqdm(query_embeddings, desc='Processing: ')):
        q_embeds_numpy = np.expand_dims(query_embeddings[idx], 0)
        if args.hnsw:
            q_embeds_numpy = convert_hnsw_query(q_embeds_numpy)
        D, I = index.search(q_embeds_numpy, args.beam_size)

        b_idx = 0
        metric_i = {}
        output_i = {}
        topk_tbs = []
        for _, tb_id in enumerate(I[b_idx]):
            tb = id2doc[str(tb_id)]
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

    n_threads = 24
    with Pool(n_threads) as p:
        # func_ = partial(searching)
        results = list(
            tqdm(p.imap(searching, range(len(query_embeddings)), chunksize=16), total=len(query_embeddings), desc="Searching: ", ))
    metrics, retrieval_outputs = [item[0] for item in results], [item[1] for item in results]

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

    if 'test' not in args.raw_data_path:
        table_id_gold = [inst['table_id'] for inst in retrieval_outputs]
        table_id_preds = [[item['table_id'] for item in inst['top_100']] for inst in retrieval_outputs]
        r, t, a = get_recall(table_id_gold, table_id_preds, 1)
        logger.info("Table Recall @1: {}, {}/{}".format(r, t, a))
        r, t, a = get_recall(table_id_gold, table_id_preds, 3)
        logger.info("Table Recall @5: {}, {}/{}".format(r, t, a))
        r, t, a = get_recall(table_id_gold, table_id_preds, 10)
        logger.info("Table Recall @10: {}, {}/{}".format(r, t, a))
        r, t, a = get_recall(table_id_gold, table_id_preds, 15)
        logger.info("Table Recall @15: {}, {}/{}".format(r, t, a))
        r, t, a = get_recall(table_id_gold, table_id_preds, 20)
        logger.info("Table Recall @20: {}, {}/{}".format(r, t, a))
        r, t, a = get_recall(table_id_gold, table_id_preds, 25)
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
