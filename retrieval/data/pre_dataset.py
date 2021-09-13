import json
import pickle
import os, sys
import random
from torch.utils.data import Dataset
import torch
import pandas as pd
from tqdm import tqdm
from .data_utils import collate_tokens
import copy
import sys
sys.path.append('../')

from utils.common import load_jsonl, load_json, load_pickle
from retrieval.utils.utils import move_to_cuda, AverageMeter, load_saved

from transformers import BatchEncoding
from transformers.models.tapas.tokenization_tapas import add_numeric_table_values, add_numeric_values_to_question

from functools import partial
# from multiprocessing import Pool, cpu_count
from pathos.multiprocessing import Pool
import logging

logger = logging.getLogger(__name__)


class DataPrefetcher():
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.stream):
            # # self.batch is a tensor
            # self.batch = self.batch.to(device=self.device, non_blocking=True)
            # self.batch is a dict
            self.batch = {key: value.to(device=self.device, non_blocking=True) for key, value in self.batch.items()}

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            #     self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch



# def convert_tb_to_string_metadata(table, passages, meta_data, cut='passage', max_length=400):
#     header = table.columns.tolist()
#     value = table.values.tolist()
#     table_str = ' [TITLE] ' + meta_data['title'] + ' [SECTITLE] ' + meta_data['section_title'] + \
#                 ' [HEADER] ' + ' [SEP] '.join(header) + ' [DATA] '+' [SEP] '.join(value[0])
#     passage_str = ' [PASSAGE] ' + ' [SEP] '.join(passages)
#     return table_str, passage_str
#
# def convert_tb_to_string_metadata_norm(table, passages, meta_data, cut='passage', max_length=400):
#     # normed text by processing table with " H1 is C1 .... "
#     header = table.columns.tolist()
#     value = table.values.tolist()
#     # table_str = ' [TITLE] ' + meta_data['title'] + ' [SECTITLE] ' + meta_data['section_title'] + \
#     #             ' [HEADER] ' + ' [SEP] '.join(header) + ' [DATA] '+' [SEP] '.join(value[0])
#     table_str = ' [TAB] ' + ' [TITLE] ' + meta_data['title']+' [SECTITLE] ' + meta_data['section_title']+ ' [DATA] ' + \
#                 ' '.join(['{} is {}.'.format(h,c) for h,c in zip(header,value[0])])
#     passage_str = ' [PASSAGE] ' + ' [SEP] '.join(passages)
#     return table_str, passage_str

# def convert_tb_to_string_metadata(table, passages, meta_data, cut='passage', max_length=400):
#     # with some new special tokens -- results indicate no use
#     header = table.columns.tolist()
#     value = table.values.tolist()
#     table_str = ' TITLE ' + meta_data['title'] + ' SECTION TITLE ' + meta_data['section_title'] + \
#                 ' HEADER|DATA </s> ' + ' </s> '.join([i+'|'+j for i,j in zip(header, value[0])])
#     passage_str = ' PASSAGE ' + '  </s> '.join(passages)
#     return table_str, passage_str


def convert_tb_to_features_bert(context, tokenizer, max_len):

    inputs = tokenizer.encode_plus(context, max_length=max_len,
                                   add_special_tokens=True, padding='max_length',
                                   return_tensors='pt', truncation=True, return_length=True)
    batch_outputs = BatchEncoding(inputs, tensor_type='pt')
    return batch_outputs


# def convert_tb_to_string(table, passages, cut='passage', max_length=400):
#     header = table.columns.tolist()
#     value = table.values.tolist()
#     table_str = ' [HEADER] ' + ' [SEP] '.join(header) + ' [SEP] '.join(value[0])
#     passage_str = ' [PASSAGE] ' + ' [SEP] '.join(passages)
#     # if cut == 'passage':
#     #     table_length = min(max_length, len(table_str.split(' ')))
#     #     doc_length = 0 if table_length >= max_length else max_length - table_length
#     # else:
#     #     doc_length = min(max_length, len(passage_str.split(' ')))
#     #     table_length = 0 if doc_length >= max_length else max_length - doc_length
#     # table_str = ' '.join(table_str.split(' ')[:table_length])
#     # passage_str = ' '.join(passage_str.split(' ')[:doc_length])
#     return table_str, passage_str
#
# def convert_tb_to_string_norm(table, passages, cut='passage', max_length=400):
#     header = table.columns.tolist()
#     value = table.values.tolist()
#     table_str = ' [TAB] ' + ' '.join(['{} is {}.'.format(h,c) for h,c in zip(header, value[0])])
#     passage_str = ' [PASSAGE] ' + ' [SEP] '.join(passages)
#     # if cut == 'passage':
#     #     table_length = min(max_length, len(table_str.split(' ')))
#     #     doc_length = 0 if table_length >= max_length else max_length - table_length
#     # else:
#     #     doc_length = min(max_length, len(passage_str.split(' ')))
#     #     table_length = 0 if doc_length >= max_length else max_length - doc_length
#     # table_str = ' '.join(table_str.split(' ')[:table_length])
#     # passage_str = ' '.join(passage_str.split(' ')[:doc_length])
#     return table_str, passage_str

# def convert_tb_to_string(table, passages, cut='passage', max_length=400):
#     header = table.columns.tolist()
#     value = table.values.tolist()
#     table_str = ' HEADER|DATA </s> ' + ' </s> '.join([i+'|'+j  for i,j in zip(header, value[0])]) + ' '
#     passage_str = ' PASSAGE ' + ' </s> '.join(passages)
#     return table_str, passage_str


# def convert_tb_to_features_bert(passages, table, tokenizer, args):
#     if table.empty:
#         inputs = tokenizer.encode_plus(passages, max_length=args.max_q_len,
#                                        add_special_tokens=True, padding='max_length',
#                                        return_tensors='pt', truncation=True, return_length=True)
#     else:
#         if args.normalize_table:
#             table_str, passage_str = convert_tb_to_string_norm(table, passages)
#         else:
#             table_str, passage_str = convert_tb_to_string(table, passages)
#         inputs = tokenizer.encode_plus(table_str, text_pair=passage_str, max_length=args.max_c_len,
#                                        add_special_tokens=True, padding='max_length',
#                                        return_tensors='pt', truncation=True, return_length=True)
#     # inputs['length'] = len(inputs['input_ids'])
#     batch_outputs = BatchEncoding(inputs, tensor_type='pt')
#     return batch_outputs


class PREDatasetNegaMeta(Dataset):

    def __init__(self, tokenizer, data_path, args, train=False):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.args = args
        self.train = train
        self.question = []
        self.table_block = []
        self.negative_table_block = []
        self.labels = []
        self.data = []

    # def processing_data(self):
    #     logger.info(f"Loading data from {self.data_path}")
    #     self.data = load_jsonl(self.data_path)
    #     # self.data = self.data[:300]
    #     logger.info(f"Total sample count {len(self.data)}")
    #
    #     if not self.args.overwrite_cache and self.train and self.args.save_tensor_path and \
    #             os.path.exists(self.args.save_tensor_path) and len(os.listdir(self.args.save_tensor_path)) == 4:
    #         self.load_tensor(self.args.save_tensor_path)
    #     else:
    #         # def running_function(js, args, tokenizer):
    #         #     output = {}
    #         #     output['question'] = convert_tb_to_features_bert(js['question'], tokenizer=tokenizer, max_len=args.max_q_len)
    #         #     output['table_block'] = convert_tb_to_features_bert(js['passages'], tokenizer=tokenizer, max_len=args.max_c_len)
    #         #     outxput['negative_table_block'] = convert_tb_to_features_bert(js['neg_passages'], tokenizer=tokenizer, max_len=args.max_c_len)
    #         #     output['labels'] = torch.tensor(js['label'])
    #         #     return output
    #         # thread = 10
    #         # with Pool(thread) as p:
    #         #     func_ = partial(running_function, args=self.args, tokenizer=self.tokenizer)
    #         #     all_results = list(tqdm(p.imap(func_, self.data, chunksize=16), total=len(self.data),
    #         #                             desc="preparing singleRetriever dataset with metadata with multiprocessing..", ))
    #         #     self.question = [res['question'] for res in all_results]
    #         #     self.table_block = [res['table_block'] for res in all_results]
    #         #     self.negative_table_block = [res['negative_table_block'] for res in all_results]
    #         #     self.labels = [res['labels'] for res in all_results]
    #
    #         self.question = []
    #         self.table_block = []
    #         self.negative_table_block = []
    #         self.labels = []
    #         for js in tqdm(self.data, desc="preparing singleRetriever dataset with serial"):
    #             self.question.append(convert_tb_to_features_bert(js['question'], tokenizer=self.tokenizer, max_len=self.args.max_q_len))
    #             self.table_block.append(convert_tb_to_features_bert(js['passages'], tokenizer=self.tokenizer, max_len=self.args.max_c_len))
    #             self.negative_table_block.append(convert_tb_to_features_bert(js['neg_passages'], tokenizer=self.tokenizer, max_len=self.args.max_c_len))
    #             self.labels.append(torch.tensor(js['label']))
    #
    #         if self.train and self.args.save_tensor_path:
    #             self.save_tensor(self.args.save_tensor_path)
    #
    # def __getitem__(self, index):
    #     return {
    #         "q": self.question[index],
    #         "tb": self.table_block[index],
    #         "neg_tb": self.negative_table_block[index],
    #         "label": self.labels[index],
    #     }

    def processing_data(self):
        logger.info(f"Loading data from {self.data_path}")
        self.data = load_jsonl(self.data_path)
        # self.data = self.data[:300]
        logger.info(f"Total sample count {len(self.data)}")

        self.question = []
        self.table_block = []
        self.negative_table_block = []
        self.labels = []

    def __getitem__(self, index):
        js = self.data[index]
        question = convert_tb_to_features_bert(js['question'], tokenizer=self.tokenizer, max_len=self.args.max_q_len)
        table_block = convert_tb_to_features_bert(js['passages'], tokenizer=self.tokenizer, max_len=self.args.max_c_len)
        negative_table_block = convert_tb_to_features_bert(js['neg_passages'], tokenizer=self.tokenizer, max_len=self.args.max_c_len)
        labels = torch.tensor(js['label'])

        return {
            "q": question,
            "tb": table_block,
            "neg_tb": negative_table_block,
            "label": labels,
        }

    def __len__(self):
        return len(self.data)

    def save_tensor(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.question, os.path.join(path, 'question.pkl'))
        logger.info("Saving training question \t tensors to {}".format(os.path.join(path, 'question.pkl')))
        torch.save(self.table_block, os.path.join(path, 'table_block.pkl'))
        logger.info("Saving training table_block \t tensors to {}".format(os.path.join(path, 'table_block.pkl')))
        torch.save(self.negative_table_block, os.path.join(path, 'negative_table_block.pkl'))
        logger.info("Saving training negative_table_block \t tensors to {}".format(
            os.path.join(path, 'negative_table_block.pkl')))
        torch.save(self.labels, os.path.join(path, 'labels.pkl'))
        logger.info("Saving training labels \t tensors to {}".format(os.path.join(path, 'labels.pkl')))
        logger.info("Saving training (q, tb, neg_tb, lab) tensors to {}".format(path))

    def load_tensor(self, path):
        self.question = torch.load(os.path.join(path, 'question.pkl'))
        logger.info("Loading training question \t tensors from {}".format(os.path.join(path, 'question.pkl')))
        self.table_block = torch.load(os.path.join(path, 'table_block.pkl'))
        logger.info("Loading training table_block \t tensors from {}".format(os.path.join(path, 'table_block.pkl')))
        self.negative_table_block = torch.load(os.path.join(path, 'negative_table_block.pkl'))
        logger.info("Loading training negative_table_block \t tensors from {}".format(
            os.path.join(path, 'negative_table_block.pkl')))
        self.labels = torch.load(os.path.join(path, 'labels.pkl'))
        logger.info("Loading training labels \t tensors from {}".format(os.path.join(path, 'labels.pkl')))
        logger.info("Loading training (q, tb, neg_tb, lab) tensors from {}".format(path))


def collate_all_tokens(input_type, inputs, *args):
    return_dict = {}
    for arg in args:
        return_dict[input_type + '_' + arg] = collate_tokens([input[input_type][arg].view(-1) for input in inputs], 0)
    return return_dict


# def tb_collate_bert(samples, pad_id=0):
#     if len(samples) == 0:
#         return {}
#
#     batch = {
#         'q_input_ids': collate_tokens([s["q"]["input_ids"].view(-1) for s in samples], pad_id),
#         'q_mask': collate_tokens([s["q"]["attention_mask"].view(-1) for s in samples], 0),
#         'c_input_ids': collate_tokens([s["tb"]["input_ids"].view(-1) for s in samples], pad_id),
#         'c_mask': collate_tokens([s["tb"]["attention_mask"].view(-1) for s in samples], 0),
#         'neg_input_ids': collate_tokens([s["neg_tb"]["input_ids"].view(-1) for s in samples], pad_id),
#         'neg_mask': collate_tokens([s["neg_tb"]["attention_mask"].view(-1) for s in samples], 0),
#     }
#
#     if "token_type_ids" in samples[0]["q"]:
#         batch.update({
#             'q_type_ids': collate_tokens([s["q"]["token_type_ids"].view(-1) for s in samples], 0),
#             'c_type_ids': collate_tokens([s["tb"]["token_type_ids"].view(-1) for s in samples], 0),
#             'neg_type_ids': collate_tokens([s["neg_tb"]["token_type_ids"].view(-1) for s in samples], 0),
#         })
#
#     return batch


def tb_collate(samples, pad_id=0):
    if len(samples) == 0:
        return {}

    batch = {
        'q_input_ids': collate_tokens([s["q"]["input_ids"].view(-1) for s in samples], pad_id),
        'q_mask': collate_tokens([s["q"]["attention_mask"].view(-1) for s in samples], 0),
        'c_input_ids': collate_tokens([s["tb"]["input_ids"].view(-1) for s in samples], pad_id),
        'c_mask': collate_tokens([s["tb"]["attention_mask"].view(-1) for s in samples], 0),
        'neg_input_ids': collate_tokens([s["neg_tb"]["input_ids"].view(-1) for s in samples], pad_id),
        'neg_mask': collate_tokens([s["neg_tb"]["attention_mask"].view(-1) for s in samples], 0),
    }

    if "part2_mask" in samples[0]["q"]:
        batch.update({
            'q_part2_mask': collate_tokens([s["q"]["part2_mask"].view(-1) for s in samples], 0),
            'c_part2_mask': collate_tokens([s["tb"]["part2_mask"].view(-1) for s in samples], 0),
            'neg_part2_mask': collate_tokens([s["neg_tb"]["part2_mask"].view(-1) for s in samples], 0),
        })

    if "part3_mask" in samples[0]["q"]:
        batch.update({
            'q_part3_mask': collate_tokens([s["q"]["part3_mask"].view(-1) for s in samples], 0),
            'c_part3_mask': collate_tokens([s["tb"]["part3_mask"].view(-1) for s in samples], 0),
            'neg_part3_mask': collate_tokens([s["neg_tb"]["part3_mask"].view(-1) for s in samples], 0),
        })

    if "token_type_ids" in samples[0]["q"]:
        batch.update({
            'q_type_ids': collate_tokens([s["q"]["token_type_ids"] for s in samples], 0, '2d'),
            'c_type_ids': collate_tokens([s["tb"]["token_type_ids"] for s in samples], 0, '2d'),
            'neg_type_ids': collate_tokens([s["neg_tb"]["token_type_ids"] for s in samples], 0, '2d'),
        })

    return batch



# def tb_collate(samples, pad_id=0):
#     if len(samples) == 0:
#         return {}
#     batch = {}
#     arg_names = ['input_ids','attention_mask','token_type_ids']
#     for input_type in ['q','tb','neg_tb']:
#         batch = batch.update(collate_all_tokens(input_type,samples,arg_names))
#     logger.info(batch)
#     input()
#     # batch = {
#     #         'q_input_ids': collate_tokens([s["q_codes"]["input_ids"].view(-1) for s in samples], 0),
#     #         'q_mask': collate_tokens([s["q_codes"]["attention_mask"].view(-1) for s in samples], 0),
#     #
#     #         'tb_input_ids': collate_tokens([s["tb_codes"]["input_ids"].view(-1) for s in samples], 0),
#     #         'tb_mask':collate_tokens([s["tb_codes"]["attention_mask"].view(-1) for s in samples], 0),
#     #
#     #         'neg_tb_input_ids': collate_tokens([s["neg_tb_codes"]["input_ids"].view(-1) for s in samples], 0),
#     #         'neg_tb_mask': collate_tokens([s["neg_tb_codes"]["attention_mask"].view(-1) for s in samples], 0),
#     #
#     #     }
#
#     # if "token_type_ids" in samples[0]["q_codes"]:
#     #     batch.update({
#     #         'q_type_ids': collate_tokens([s["q_codes"]["token_type_ids"].view(-1) for s in samples], 0),
#     #         'c1_type_ids': collate_tokens([s["start_para_codes"]["token_type_ids"] for s in samples], 0),
#     #         'c2_type_ids': collate_tokens([s["bridge_para_codes"]["token_type_ids"] for s in samples], 0),
#     #         "q_sp_type_ids": collate_tokens([s["q_sp_codes"]["token_type_ids"].view(-1) for s in samples], 0),
#     #         'neg1_type_ids': collate_tokens([s["neg_codes_1"]["token_type_ids"] for s in samples], 0),
#     #         'neg2_type_ids': collate_tokens([s["neg_codes_2"]["token_type_ids"] for s in samples], 0),
#     #     })
#     #
#     # if "sent_ids" in samples[0]["start_para_codes"]:
#     #     batch["c1_sent_target"] = collate_tokens([s["start_para_codes"]["sent_ids"] for s in samples], -1)
#     #     batch["c1_sent_offsets"] = collate_tokens([s["start_para_codes"]["sent_offsets"] for s in samples], 0),
#
#     return batch
