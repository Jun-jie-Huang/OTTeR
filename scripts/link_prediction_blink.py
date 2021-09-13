import argparse
import logging
from tqdm import trange
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import json
import torch.optim as optim
from tqdm import trange, tqdm
import math
from datetime import datetime
import sys, os
import copy

sys.path.append('../')
from preprocessing.utils_preprocess import whitelist, is_year
from utils.common import convert_tb_to_string_metadata, convert_tb_to_string
import BLINK.blink.main_dense as main_dense


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def write_json(data, path):
    with open(path, 'w', encoding='utf-8') as fp:
        json.dump(data, fp)
def read_json(path):
    with open(path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    return data
import pickle
def pickle_writer(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def pickle_reader(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

class LinkGenearationDataset(Dataset):
    def __init__(self, datapath, option, max_context_length=32, shards=None):
        super(LinkGenearationDataset, self).__init__()
        self.datapath = datapath
        self.max_context_length = max_context_length  # [32, 64, 128]
        self.option = option
        self.shards = shards
        self.mapping = {}
        assert option in ['train', 'dev', 'all']
        if option != 'all':
            with open('../data_ottqa/train_dev_test_table_ids.json', 'r') as f:
                table_ids = set(json.load(f)[option])

        with open(self.datapath) as f:
            tables = json.load(f)

        if self.option == 'all':
            assert self.shards is not None
            index, total_shard = [int(_) for _ in self.shards.split('@')]
            table_ids = list(tables.keys())
            length = len(table_ids) // total_shard
            table_ids = table_ids[index * length: (index + 1) * length]
            print("Running {} out of shard {}".format(index, total_shard))
            table_ids = set(table_ids)
            print("number of tables all:{}".format(len(tables)))
            print("number of table_ids in shards {} :{}".format(self.shards, len(table_ids)))
            tables = {k:v for k, v in tables.items() if k in table_ids}
            print("number of tables replaced:{}".format(len(tables)))

        self.data = []
        for k, table in tables.items():
            # if len(self.data) > 100:
            #     break
            if k not in table_ids:
                continue

            title = table['title']
            sec_title = table['section_title']
            #             meta_data = '[TITLE] ' + title + ' [SECTITLE] ' + sec_title + ' '
            meta_data = '[TITLE] ' + title + ' [SECTITLE] ' + sec_title + ' '
            token_num = len(meta_data.split(' '))

            if isinstance(table['header'][0], list):
                headers = [_[0] for _ in table['header']]
            else:
                headers = table['header']

            for i, row in enumerate(table['data']):
                row_id = k + '_{}'.format(i)
                str_row = [cell[0] if isinstance(cell, list) else cell for cell in row]
                str_row = [' '.join(item.split(' ')[:5]) for item in str_row]

                # #### link entities in the table block
                # # inputs = meta_data + '[HEADER] ' + ' [SEP] '.join(headers) + ' [DATA] '+' [SEP] '.join(str_row)
                # inputs = meta_data + ' [SEP] '.join(['{} is {}'.format(h, c) for h, c in zip(headers, str_row)])
                #
                # links = []
                # raw_links = []
                # for cell in row:
                #     if isinstance(cell, list):
                #         for link in cell[1]:
                #             links.append(link.replace('/wiki/', '').replace('_', ' '))
                #             raw_links.append(link)
                #     else:
                #         # For plain tables
                #         pass
                # outputs = ' # '.join(links)
                # self.data.append((row_id, raw_links, inputs, outputs))
                # #### link entities in the table block

                # #### link entities in the cell
                # for cell_j, (header, cell) in enumerate(zip(headers, row)):
                #     cell_id = row_id + '_{}'.format(cell_j)
                #     content = cell[0] if isinstance(cell, list) else cell
                #     assert isinstance(content, str)
                #     if not whitelist(content):
                #         continue
                #
                #     mention = ' '.join(content.split(' ')[:8])
                #     len_mention = len(mention.split(' '))
                #     inputs_left = meta_data + ' . '.join(['{} is {}'.format(h, c) for h, c in zip(headers[:cell_j], str_row[:cell_j])])
                #     inputs_left = ' '.join(inputs_left.split(' ')[:self.max_context_length-len_mention])
                #     inputs_left = inputs_left + '. {} is '.format(headers[cell_j])
                #     len_left = len(inputs_left.split(' '))
                #     inputs_right = '. ' + ' . '.join(['{} is {}'.format(h, c) for h, c in zip(headers[cell_j+1:], str_row[cell_j+1:])])
                #     inputs_right = ' '.join(inputs_right.split(' ')[:self.max_context_length-len_left])
                #
                #     links = []
                #     if isinstance(cell, list):
                #         for link in cell[1]:
                #             links.append(link.replace('/wiki/', '').replace('_', ' '))
                #     else:
                #         # For plain tables
                #         pass
                #     outputs = ' # '.join(links)
                #     self.data.append((cell_id, links, inputs_left, mention, inputs_right, outputs))
                # #### link entities in the cell

                #### link entities just with title and sectitle
                for cell_j, (header, cell) in enumerate(zip(headers, row)):
                    cell_id = row_id + '_{}'.format(cell_j)
                    content = cell[0] if isinstance(cell, list) else cell
                    assert isinstance(content, str)
                    if not whitelist(content):
                        continue

                    mention = ' '.join(content.split(' ')[:4])
                    len_mention = len(mention.split(' '))
                    inputs_left = meta_data
                    inputs_left = ' '.join(inputs_left.split(' ')[:self.max_context_length - len_mention])
                    inputs_left = inputs_left + '. {} is '.format(headers[cell_j])
                    len_left = len(inputs_left.split(' '))
                    inputs_right = '. '

                    links = []
                    if isinstance(cell, list):
                        for link in cell[1]:
                            links.append(link.replace('/wiki/', '').replace('_', ' '))
                    else:
                        # For plain tables
                        pass
                    outputs = ' # '.join(links)
                    self.data.append((cell_id, links, inputs_left, mention, inputs_right, outputs))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # row_id, raw_links, inputs, outputs = self.data[index]
        # return row_id, raw_links, inputs, outputs
        cell_id, links, inputs_left, content, inputs_right, outputs = self.data[index]
        return cell_id, links, inputs_left, content, inputs_right, outputs

        # prefix = self.tokenizer.encode(inputs, add_special_tokens=False)
        # prefix = prefix[max(0, len(prefix) - self.source_max_len):]
        # prefix = [self.tokenizer.eos_token_id] * (self.source_max_len - len(prefix)) + prefix
        #
        # outputs = self.tokenizer.encode('[START] ' + outputs + ' [EOS]', add_special_tokens=False)
        # outputs = outputs[:self.target_max_len]
        # outputs = outputs + [self.tokenizer.eos_token_id] * (self.target_max_len - len(outputs))
        # trg_input = outputs[:-1]
        # trg_output = outputs[1:]
        #
        # prefix = torch.LongTensor(prefix)
        # trg_input = torch.LongTensor(trg_input)
        # trg_output = torch.LongTensor(trg_output)
        #
        # mask = (trg_output != self.tokenizer.eos_token_id).float()
        #
        # return row_id, links, prefix, trg_input, trg_output, mask

    def replace_links(self, new_link_predictions, save_path):
        assert len(new_link_predictions) == len(self.data)

        if self.option != 'all':
            with open('../data_ottqa/train_dev_test_table_ids.json', 'r') as f:
                table_ids = set(json.load(f)[self.option])
        with open(self.datapath) as f:
            tables = json.load(f)
        if self.option == 'all':
            assert self.shards is not None
            index, total_shard = [int(_) for _ in self.shards.split('@')]
            table_ids = list(tables.keys())
            length = len(table_ids) // total_shard
            table_ids = table_ids[index * length: (index + 1) * length]
            print("Running {} out of shard {}".format(index, total_shard))
            table_ids = set(table_ids)
            print("number of tables all:{}".format(len(tables)))
            print("number of table_ids in shards {} :{}".format(self.shards, len(table_ids)))
            tables = {k:v for k, v in tables.items() if k in table_ids}
            print("number of tables replaced:{}".format(len(tables)))
            if save_path[-5:] == '.json':
                save_path = save_path[:-5] + '-' + self.shards + '.json'
            else:
                print("self.save_path is not a json dir!")
            print("saving processed tables to {}".format(save_path))

            for k in tables.keys():
                tables[k]['header'] = [[_, []] for _ in tables[k]['header']]
                tables[k]['data'] = [[[_, []] for _ in r] for r in tables[k]['data']]
            print("finish convert plain tables with constructed header and data")
            # print(tables[list(tables.keys())[0]])

        for idx, (ori_item, cell_predictions) in enumerate(zip(self.data, new_link_predictions)):
            row_id = int(ori_item[0].split('_')[-2])
            cell_id = int(ori_item[0].split('_')[-1])
            table_id = '_'.join(ori_item[0].split('_')[:-2])
            tables[table_id]['data'][row_id][cell_id][1] = cell_predictions[1]
        with open(save_path, 'w') as f:
            json.dump(tables, f, indent=2)
        pickle_writer(new_link_predictions, save_path+'_rawoutput')


def sample_sequence(model, length, context, args, num_samples=1, temperature=1, stop_token=None, \
                    top_k=0, top_p=0.0, device='cuda'):
    if isinstance(context, list):
        context = torch.tensor(context, dtype=torch.long, device=device)
        context = context.unsqueeze(0).repeat(num_samples, 1)

    generated = context
    batch_size = generated.shape[0]

    finished_sentence = [False for _ in range(batch_size)]
    with torch.no_grad():
        for _ in range(length):
            outputs = model(generated, *args)
            if isinstance(outputs, list) or isinstance(outputs, tuple):
                next_token_logits = outputs[0][:, -1, :] / (temperature if temperature > 0 else 1.)
            else:
                next_token_logits = outputs[:, -1, :] / (temperature if temperature > 0 else 1.)

            if temperature == 0:  # greedy sampling:
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1)

            generated = torch.cat((generated, next_token), dim=1)

            if all(finished_sentence):
                break

    return generated


def process_predictions(ori_data, predictions, scores, do_evaluate=True):
    assert (len(ori_data) == len(predictions)), "ori: {}, predictions: {}".format(len(ori_data), len(predictions))
    ourputs = []
    succ, prec_total, recall_total, f1 = 0, 0, 0, 0
    all_urls = []
    for ori_item, cell_predictions, cell_scores in zip(ori_data, predictions, scores):
        row_id = '_'.join(ori_item[0].split('_')[:-1])
        required_scores = [s for s in cell_scores if s > 0]
        required_predictions = cell_predictions[:len(required_scores)]
        urls = ['/wiki/{}'.format(item.replace(' ', '_')) for item in required_predictions]
        all_urls.append(urls)
        if do_evaluate:
            true_answers = ori_item[1]
            succ += len(set(true_answers) & set(required_predictions))
            prec_total += len(required_predictions)
            recall_total += len(true_answers)
    if do_evaluate:
        precision = succ / prec_total
        recall = succ / recall_total
        f1 = 2 * precision * recall / (precision + recall + 0.001)
        print('Precision: {}, Recall:{}, F1:{}; succ:{}, prec_total:{}, recall_total:{}'.format(precision, recall, f1, succ,
                                                                                                prec_total, recall_total))
    return list(zip(predictions, all_urls, scores))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='gpt2', type=str)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--dataset', default='./blink_output/results.json', type=str, help="Whether to use dataset")
    parser.add_argument('--data_output', default=None, type=str, help="Whether to use dataset")
    parser.add_argument('--load_from', default=None, type=str, help="Whether to use dataset")
    parser.add_argument('--batch_size', default=128, type=int, help="Whether to use dataset")
    parser.add_argument('--every', default=50, type=int, help="Whether to use dataset")
    parser.add_argument('--max_source_len', default=32, type=int, help="Whether to use dataset")
    parser.add_argument('--max_target_len', default=32, type=int, help="Whether to use dataset")
    parser.add_argument('--do_train', default=False, action="store_true", help="link training tables")
    parser.add_argument('--do_all', default=False, action="store_true", help="link all tables")
    parser.add_argument('--do_val', default=False, action="store_true", help="link dev tables")
    parser.add_argument('--shard', default=None, type=str, help="whether to train or test the model")
    parser.add_argument('--mode', default='zeshel', type=str, choices=['zeshel', 'ottqa_blink', 'ottqa_blink_continue'])
    parser.add_argument("--fast", action="store_true")
    parser.add_argument('--topk', default=5, type=int, help="topk for biencoder")

    args = parser.parse_args()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    args.device = torch.device("cuda")
    args.n_gpu = torch.cuda.device_count()

    if args.mode == 'zeshel':
        models_path = "../BLINK/models/"  # the path where you stored the BLINK models
        config = {
            "test_entities": None,
            "test_mentions": None,
            "interactive": False,
            "top_k": args.topk,
            "bi_batch_size": 192,
            "biencoder_model": models_path + "biencoder_wiki_large.bin",
            "biencoder_config": models_path + "biencoder_wiki_large.json",
            "entity_catalogue": models_path + "entity.jsonl",  # 5903527
            "entity_encoding": models_path + "all_entities_large.t7",
            "crossencoder_model": models_path + "crossencoder_wiki_large.bin",
            "crossencoder_config": models_path + "crossencoder_wiki_large.json",
            "fast": args.fast,  # set this to be true if speed is a concern
            "output_path": "blink_output/",  # logging directory
            "no_cuda": False,  # set this to be true if speed is a concern
        }
    elif args.mode == 'ottqa_blink_continue':
        # biencoder_models_path = "../BLINK/output/blink_ottqa_continue_train/biencoder/"
        # crossencoder_models_path = "../BLINK/models/"
        models_path = "../BLINK/models/"
        config = {
            "test_entities": None,
            "test_mentions": None,
            "interactive": False,
            "top_k": args.topk,
            "bi_batch_size": 32,
            "cr_batch_size": 8,
            "biencoder_model": "../BLINK/output/blink_ottqa_continue_train/biencoder/pytorch_model.bin",
            "biencoder_config": models_path + "biencoder_wiki_large.json",
            # "biencoder_config": "../BLINK/output/blink_ottqa_continue_train/biencoder/training_params.txt",
            "entity_catalogue": "../entity_linking/data/ottqa_blink_format/ottqa_wiki_entities.jsonl",   # 6057029
            "entity_encoding": "../BLINK/output/blink_ottqa_continue_train/biencoder/cand_encode_path.bin",
            "crossencoder_model": None,
            "crossencoder_config": None,
            "fast": args.fast,  # set this to be true if speed is a concern
            "output_path": os.path.dirname(args.data_output),
            "no_cuda": False,  # set this to be true if speed is a concern
        }
        if not args.fast:
            config.update({
                "crossencoder_model": "../BLINK/models/crossencoder_wiki_large.bin",
                "crossencoder_config": "../BLINK/models/crossencoder_wiki_large.json",
            })
    elif args.mode == 'ottqa_blink':
        # biencoder_models_path = "../BLINK/output/blink_ottqa_continue_train/biencoder/"
        # crossencoder_models_path = "../BLINK/models/"
        models_path = "../BLINK/models/"
        config = {
            "test_entities": None,
            "test_mentions": None,
            "interactive": False,
            "top_k": args.topk,
            "bi_batch_size": 32,
            "cr_batch_size": 8,
            "biencoder_model": "../BLINK/output/blink_ottqa/biencoder/pytorch_model.bin",
            "biencoder_config": models_path + "biencoder_wiki_large.json",
            # "biencoder_config": "../BLINK/output/blink_ottqa/biencoder/training_params.txt",
            "entity_catalogue": "../entity_linking/data/ottqa_blink_format/ottqa_wiki_entities.jsonl",   # 6057029
            "entity_encoding": "../BLINK/output/blink_ottqa/biencoder/cand_encode_path.bin",
            "crossencoder_model": None,
            "crossencoder_config": None,
            "fast": args.fast,  # set this to be true if speed is a concern
            "output_path": os.path.dirname(args.data_output),
            "no_cuda": False,  # set this to be true if speed is a concern
        }
        if not args.fast:
            config.update({
                "crossencoder_model": "../BLINK/models/crossencoder_wiki_large.bin",
                "crossencoder_config": "../BLINK/models/crossencoder_wiki_large.json",
            })
    else:
        raise NotImplementedError("args.mode not implement:{}".format(args.mode))
    blink_args = argparse.Namespace(**config)
    models = main_dense.load_models(blink_args, logger=logger)

    if not args.do_all:
        assert (args.do_train and not args.do_val) or (not args.do_train and args.do_val)
        if args.do_train:
            dataset = LinkGenearationDataset(args.dataset, 'train', args.max_source_len)
        if args.do_val:
            dataset = LinkGenearationDataset(args.dataset, 'dev', args.max_source_len)
        data_to_link = [{"id": idx,
                         "label": item[1][0] if item[1] else "unknown",
                         # "label": "unknown",
                         "label_id": -1,
                         "context_left": item[2],
                         "mention": item[3],
                         "context_right": item[4]} for idx, item in enumerate(dataset.data)]
        print(len(data_to_link))
        a, b, c, d, e, predictions, scores, = main_dense.run(blink_args, logger, *models, test_data=data_to_link)
        print("a:{}, b:{}, c:{}, d:{}, e:{}".format(a, b, c, d, e))
        pickle_writer(dataset.data, 'tmp/{}_data.pkl'.format('train' if args.do_train else 'dev'))
        pickle_writer(predictions, 'tmp/{}_preds.pkl'.format('train' if args.do_train else 'dev'))
        pickle_writer(scores, 'tmp/{}_score.pkl'.format('train' if args.do_train else 'dev'))
        new_outputs = process_predictions(dataset.data, predictions, scores, do_evaluate=True)

        # dataset.replace_links(new_outputs, './blink_output/traindev_tables_blink.json')
        dataset.replace_links(new_outputs, args.data_output)

    if args.do_all:
        dataset = LinkGenearationDataset(args.dataset, 'all', args.max_source_len, args.shard)
        data_to_link = [{"id": idx,
                         "label": item[1][0] if item[1] else "unknown",
                         # "label": "unknown",
                         "label_id": -1,
                         "context_left": item[2],
                         "mention": item[3],
                         "context_right": item[4]} for idx, item in enumerate(dataset.data)]
        print(len(data_to_link))
        a, b, c, d, e, predictions, scores, = main_dense.run(blink_args, logger, *models, test_data=data_to_link)
        print("a:{}, b:{}, c:{}, d:{}, e:{}".format(a, b, c, d, e))
        pickle_writer(dataset.data, 'tmp/all_data_{}_{}.pkl'.format(os.path.basename(args.dataset), args.shard))
        pickle_writer(predictions, 'tmp/all_preds_{}_{}.pkl'.format(os.path.basename(args.dataset), args.shard))
        pickle_writer(scores, 'tmp/all_score_{}_{}.pkl'.format(os.path.basename(args.dataset), args.shard))
        new_outputs = process_predictions(dataset.data, predictions, scores, do_evaluate=False)
        dataset.replace_links(new_outputs, args.data_output)

