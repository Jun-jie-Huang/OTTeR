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
# from utils.common import convert_tb_to_string_metadata, convert_tb_to_string, get_passages
from utils.common import convert_tb_to_string, get_passages, load_jsonl

from transformers import TapasTokenizer
# from transformers.models.tapas.tokenization_tapas import BatchEncoding
from transformers import BatchEncoding
from transformers.models.tapas.tokenization_tapas import add_numeric_table_values, add_numeric_values_to_question

from functools import partial
# from multiprocessing import Pool, cpu_count
from pathos.multiprocessing import Pool
import logging

logger = logging.getLogger(__name__)


def convert_tb_to_string_metadata(table, passages, meta_data, cut='passage', max_length=400):
    header = table.columns.tolist()
    value = table.values.tolist()
    table_str = ' [TITLE] ' + meta_data['title'] + ' [SECTITLE] ' + meta_data['section_title'] + \
                ' [HEADER] ' + ' [SEP] '.join(header) + ' [DATA] '+' [SEP] '.join(value[0])
    passage_str = ' [PASSAGE] ' + ' [SEP] '.join(passages)
    return table_str, passage_str

def convert_tb_to_string_metadata_norm(table, passages, meta_data, cut='passage', max_length=400):
    # normed text by processing table with " H1 is C1 .... "
    header = table.columns.tolist()
    value = table.values.tolist()
    # table_str = ' [TITLE] ' + meta_data['title'] + ' [SECTITLE] ' + meta_data['section_title'] + \
    #             ' [HEADER] ' + ' [SEP] '.join(header) + ' [DATA] '+' [SEP] '.join(value[0])
    table_str = ' [TAB] ' + ' [TITLE] ' + meta_data['title']+' [SECTITLE] ' + meta_data['section_title']+ ' [DATA] ' + \
                ' '.join(['{} is {}.'.format(h,c) for h,c in zip(header,value[0])])
    passage_str = ' [PASSAGE] ' + ' [SEP] '.join(passages)
    return table_str, passage_str

# def convert_tb_to_string_metadata(table, passages, meta_data, cut='passage', max_length=400):
#     # with some new special tokens -- results indicate no use
#     header = table.columns.tolist()
#     value = table.values.tolist()
#     table_str = ' TITLE ' + meta_data['title'] + ' SECTION TITLE ' + meta_data['section_title'] + \
#                 ' HEADER|DATA </s> ' + ' </s> '.join([i+'|'+j for i,j in zip(header, value[0])])
#     passage_str = ' PASSAGE ' + '  </s> '.join(passages)
#     return table_str, passage_str


def convert_tb_to_features_bert_metadata(passages, table, meta_data, tokenizer, args, encode=False):
    if table.empty:
        if type(passages) == str:
            passage_str = passages  # == query
        elif type(passages) == list:
            passage_str = ' '.join(passages)
        else:
            raise TypeError ("type passages: {}, {}, {}, {}".format(type(passages), passages, table, meta_data))
        inputs = tokenizer.encode_plus(passage_str, max_length=args.max_q_len,
                                       add_special_tokens=True, padding='max_length',
                                       return_tensors='pt', truncation=True, return_length=True)
    else:
        if args.normalize_table:
            table_str, passage_str = convert_tb_to_string_metadata_norm(table, passages, meta_data)
        else:
            table_str, passage_str = convert_tb_to_string_metadata(table, passages, meta_data)
        inputs = tokenizer.encode_plus(table_str, text_pair=passage_str, max_length=args.max_c_len,
                                       add_special_tokens=True, padding='max_length',
                                       return_tensors='pt', truncation=True, return_length=True)
    # inputs['length'] = len(inputs['input_ids'])
    batch_outputs = BatchEncoding(inputs, tensor_type='pt')
    # batch_outputs = BatchEncoding(inputs, tensor_type='pt', prepend_batch_axis=True)
    return batch_outputs


def convert_tb_to_features_bert_metadata_threecat(passages, table, meta_data, tokenizer, args, encode=False):
    if table.empty:
        if type(passages) == str:
            passage_str = passages  # == query
        elif type(passages) == list:
            passage_str = ' '.join(passages)
        else:
            raise TypeError ("type passages: {}, {}, {}, {}".format(type(passages), passages, table, meta_data))
        inputs = tokenizer.encode_plus(passage_str, text_pair=passage_str, max_length=args.max_q_len*2,
                                       add_special_tokens=True, padding='max_length',
                                       return_tensors='pt', truncation=True, return_special_tokens_mask=True,
                                       return_length=True)
    else:
        if args.normalize_table:
            table_str, passage_str = convert_tb_to_string_metadata_norm(table, passages, meta_data)
        else:
            table_str, passage_str = convert_tb_to_string_metadata(table, passages, meta_data)
        inputs = tokenizer.encode_plus(table_str, text_pair=passage_str, max_length=args.max_c_len,
                                       add_special_tokens=True, padding='max_length',
                                       return_tensors='pt', truncation=True, return_special_tokens_mask=True,
                                       return_length=True)
    zero_positions = torch.where(inputs['special_tokens_mask'] == 1)[1]
    inputs['part2_mask'] = torch.tensor([[0] +
                                         [1] * (zero_positions[1] - 1) +
                                         [0] * (inputs['length'] - zero_positions[1])])
    inputs['part3_mask'] = torch.tensor([[0] * (zero_positions[2] + 1) +
                                         [1] * (zero_positions[3] - zero_positions[2] - 1) +
                                         [0] * (inputs['length'] - zero_positions[3])])
    del inputs['special_tokens_mask']
    batch_outputs = BatchEncoding(inputs, tensor_type='pt')
    # batch_outputs = BatchEncoding(inputs, tensor_type='pt', prepend_batch_axis=True)
    return batch_outputs


def convert_tb_to_features_bert_metadata_threecat_one_query(passages, table, meta_data, tokenizer, args, encode=False):
    if table.empty:
        if type(passages) == str:
            passage_str = passages  # == query
        elif type(passages) == list:
            passage_str = ' '.join(passages)
        else:
            raise TypeError ("type passages: {}, {}, {}, {}".format(type(passages), passages, table, meta_data))
        inputs = tokenizer.encode_plus(passage_str, max_length=args.max_q_len,
                                       add_special_tokens=True, padding='max_length',
                                       return_tensors='pt', truncation=True, return_special_tokens_mask=True,
                                       return_length=True)
        zero_positions = torch.where(inputs['special_tokens_mask'] == 1)[1]
        inputs['part2_mask'] = torch.tensor([[0] +
                                             [1] * (zero_positions[1] - 1) +
                                             [0] * (inputs['length'] - zero_positions[1])])
        inputs['part3_mask'] = copy.deepcopy(inputs['part2_mask'])
    else:
        if args.normalize_table:
            table_str, passage_str = convert_tb_to_string_metadata_norm(table, passages, meta_data)
        else:
            table_str, passage_str = convert_tb_to_string_metadata(table, passages, meta_data)
        inputs = tokenizer.encode_plus(table_str, text_pair=passage_str, max_length=args.max_c_len,
                                       add_special_tokens=True, padding='max_length',
                                       return_tensors='pt', truncation=True, return_special_tokens_mask=True,
                                       return_length=True)
        zero_positions = torch.where(inputs['special_tokens_mask'] == 1)[1]
        inputs['part2_mask'] = torch.tensor([[0] +
                                             [1] * (zero_positions[1] - 1) +
                                             [0] * (inputs['length'] - zero_positions[1])])
        inputs['part3_mask'] = torch.tensor([[0] * (zero_positions[2] + 1) +
                                             [1] * (zero_positions[3] - zero_positions[2] - 1) +
                                             [0] * (inputs['length'] - zero_positions[3])])

    batch_outputs = BatchEncoding(inputs, tensor_type='pt')
    # batch_outputs = BatchEncoding(inputs, tensor_type='pt', prepend_batch_axis=True)
    return batch_outputs


def convert_aug_tb_to_features_bert_threecat(passages, tokenizer, args):
    table_str, passage_str = passages.split('[PASSAGE]')
    passage_str = '[PASSAGE]' + passage_str
    inputs = tokenizer.encode_plus(table_str, text_pair=passage_str, max_length=args.max_c_len,
                                   add_special_tokens=True, padding='max_length',
                                   return_tensors='pt', truncation=True, return_special_tokens_mask=True,
                                   return_length=True)
    zero_positions = torch.where(inputs['special_tokens_mask'] == 1)[1]
    inputs['part2_mask'] = torch.tensor([[0] +
                                         [1] * (zero_positions[1] - 1) +
                                         [0] * (inputs['length'] - zero_positions[1])])
    inputs['part3_mask'] = torch.tensor([[0] * (zero_positions[2] + 1) +
                                         [1] * (zero_positions[3] - zero_positions[2] - 1) +
                                         [0] * (inputs['length'] - zero_positions[3])])

    batch_outputs = BatchEncoding(inputs, tensor_type='pt')
    # batch_outputs = BatchEncoding(inputs, tensor_type='pt', prepend_batch_axis=True)
    return batch_outputs


def convert_tb_to_string(table, passages, cut='passage', max_length=400):
    header = table.columns.tolist()
    value = table.values.tolist()
    table_str = ' [HEADER] ' + ' [SEP] '.join(header) + ' [SEP] '.join(value[0])
    passage_str = ' [PASSAGE] ' + ' [SEP] '.join(passages)
    # if cut == 'passage':
    #     table_length = min(max_length, len(table_str.split(' ')))
    #     doc_length = 0 if table_length >= max_length else max_length - table_length
    # else:
    #     doc_length = min(max_length, len(passage_str.split(' ')))
    #     table_length = 0 if doc_length >= max_length else max_length - doc_length
    # table_str = ' '.join(table_str.split(' ')[:table_length])
    # passage_str = ' '.join(passage_str.split(' ')[:doc_length])
    return table_str, passage_str

def convert_tb_to_string_norm(table, passages, cut='passage', max_length=400):
    header = table.columns.tolist()
    value = table.values.tolist()
    table_str = ' [TAB] ' + ' '.join(['{} is {}.'.format(h,c) for h,c in zip(header, value[0])])
    passage_str = ' [PASSAGE] ' + ' [SEP] '.join(passages)
    # if cut == 'passage':
    #     table_length = min(max_length, len(table_str.split(' ')))
    #     doc_length = 0 if table_length >= max_length else max_length - table_length
    # else:
    #     doc_length = min(max_length, len(passage_str.split(' ')))
    #     table_length = 0 if doc_length >= max_length else max_length - doc_length
    # table_str = ' '.join(table_str.split(' ')[:table_length])
    # passage_str = ' '.join(passage_str.split(' ')[:doc_length])
    return table_str, passage_str

# def convert_tb_to_string(table, passages, cut='passage', max_length=400):
#     header = table.columns.tolist()
#     value = table.values.tolist()
#     table_str = ' HEADER|DATA </s> ' + ' </s> '.join([i+'|'+j  for i,j in zip(header, value[0])]) + ' '
#     passage_str = ' PASSAGE ' + ' </s> '.join(passages)
#     return table_str, passage_str


def convert_tb_to_features_bert(passages, table, tokenizer, args):
    if table.empty:
        inputs = tokenizer.encode_plus(passages, max_length=args.max_q_len,
                                       add_special_tokens=True, padding='max_length',
                                       return_tensors='pt', truncation=True, return_length=True)
    else:
        if args.normalize_table:
            table_str, passage_str = convert_tb_to_string_norm(table, passages)
        else:
            table_str, passage_str = convert_tb_to_string(table, passages)
        inputs = tokenizer.encode_plus(table_str, text_pair=passage_str, max_length=args.max_c_len,
                                       add_special_tokens=True, padding='max_length',
                                       return_tensors='pt', truncation=True, return_length=True)
    # inputs['length'] = len(inputs['input_ids'])
    batch_outputs = BatchEncoding(inputs, tensor_type='pt')
    return batch_outputs


def convert_tb_to_features_tapas(passage, table, tokenizer, args):
    # tokenizer: TapasTokenizer

    passage_tokens = tokenizer.tokenize(passage)
    tokenized_table = tokenizer._tokenize_table(table)

    num_rows = tokenizer._get_num_rows(table, drop_rows_to_fit=True)
    num_columns = tokenizer._get_num_columns(table)
    _, _, num_tokens = tokenizer._get_table_boundaries(tokenized_table)

    table_data = list(tokenizer._get_table_values(tokenized_table, num_columns, num_rows, num_tokens))
    table_tokens = list(zip(*table_data))[0] if len(table_data) > 0 else list(zip(*table_data))

    passage_ids = tokenizer.convert_tokens_to_ids(passage_tokens)
    table_ids = tokenizer.convert_tokens_to_ids(table_tokens)
    # input_ids = passage_ids + table_ids
    # input_ids = tokenizer.build_inputs_with_special_tokens(passage_ids, table_ids)
    # max_c_len = 512
    if len(passage_ids + table_ids) > args.max_c_len - 2:
        length_passage = max(args.max_c_len - len(table_ids) - 2, 100)
        length_table = min(410, len(table_ids))
        input_ids = [tokenizer.cls_token_id] + passage_ids[:length_passage] + \
                    [tokenizer.sep_token_id] + table_ids[:length_table]
        input_ids = input_ids[:args.max_c_len]
        passage_ids = passage_ids[:length_passage]
        passage = tokenizer.decode(passage_ids[:length_passage])
    else:
        input_ids = [tokenizer.cls_token_id] + passage_ids + [tokenizer.sep_token_id] + table_ids
        input_ids = input_ids[:args.max_c_len]

    encoded_inputs = {}
    encoded_inputs['input_ids'] = input_ids
    segment_ids = tokenizer.create_segment_token_type_ids_from_sequences(passage_ids, table_data)
    column_ids = tokenizer.create_column_token_type_ids_from_sequences(passage_ids, table_data)
    row_ids = tokenizer.create_row_token_type_ids_from_sequences(passage_ids, table_data)
    prev_labels = [0] * len(row_ids)

    raw_table = add_numeric_table_values(table)
    raw_passage = add_numeric_values_to_question(passage)
    column_ranks, inv_column_ranks = tokenizer._get_numeric_column_ranks(column_ids, row_ids, raw_table)
    numeric_relations = tokenizer._get_numeric_relations(raw_passage, column_ids, row_ids, raw_table)
    # if answer_coordinates is not None and answer_text is not None:
    #     labels = self.get_answer_ids(column_ids, row_ids, table_data, answer_text, answer_coordinates)
    #     numeric_values = self._get_numeric_values(raw_table, column_ids, row_ids)
    #     numeric_values_scale = self._get_numeric_values_scale(raw_table, column_ids, row_ids)
    #     encoded_inputs["labels"] = labels
    #     encoded_inputs["numeric_values"] = numeric_values
    #     encoded_inputs["numeric_values_scale"] = numeric_values_scale

    token_type_ids = [
        segment_ids,
        column_ids,
        row_ids,
        prev_labels,
        column_ranks,
        inv_column_ranks,
        numeric_relations,
    ]

    token_type_ids = [list(ids) for ids in list(zip(*token_type_ids))]
    encoded_inputs["token_type_ids"] = token_type_ids

    attention_mask = tokenizer.create_attention_mask_from_sequences(passage_ids, table_data)
    encoded_inputs["attention_mask"] = attention_mask

    # Check lengths
    if len(encoded_inputs["input_ids"]) > tokenizer.model_max_length:
        logger.warning(
                f"Token indices sequence length is longer than the specified maximum sequence length "
                f"for this model ({len(encoded_inputs['input_ids'])} > {tokenizer.model_max_length}). Running this "
                "sequence through the model will result in indexing errors.")

    # Padding
    encoded_inputs = tokenizer.pad(encoded_inputs, max_length=args.max_c_len, padding='max_length',
                                   return_attention_mask=True)
    encoded_inputs["length"] = len(encoded_inputs["input_ids"])

    batch_outputs = BatchEncoding(encoded_inputs, tensor_type='pt', prepend_batch_axis=True)

    return batch_outputs


def convert_tb_to_features_tapas_metadata(passage, table, meta_data, tokenizer, args):
    # tokenizer: TapasTokenizer
    passage = ' [TITLE] ' + meta_data['title'] + ' [SECTITLE] ' + meta_data['section_title'] + ' [PASSAGE] ' + passage

    passage_tokens = tokenizer.tokenize(passage)
    tokenized_table = tokenizer._tokenize_table(table)

    num_rows = tokenizer._get_num_rows(table, drop_rows_to_fit=True)
    num_columns = tokenizer._get_num_columns(table)
    _, _, num_tokens = tokenizer._get_table_boundaries(tokenized_table)

    table_data = list(tokenizer._get_table_values(tokenized_table, num_columns, num_rows, num_tokens))
    table_tokens = list(zip(*table_data))[0] if len(table_data) > 0 else list(zip(*table_data))

    passage_ids = tokenizer.convert_tokens_to_ids(passage_tokens)
    table_ids = tokenizer.convert_tokens_to_ids(table_tokens)
    # input_ids = passage_ids + table_ids
    # input_ids = tokenizer.build_inputs_with_special_tokens(passage_ids, table_ids)
    # max_c_len = 512
    if len(passage_ids + table_ids) > args.max_c_len - 2:
        length_passage = max(args.max_c_len - len(table_ids) - 2, 100)
        length_table = min(410, len(table_ids))
        input_ids = [tokenizer.cls_token_id] + passage_ids[:length_passage] + \
                    [tokenizer.sep_token_id] + table_ids[:length_table]
        input_ids = input_ids[:args.max_c_len]
        passage_ids = passage_ids[:length_passage]
        passage = tokenizer.decode(passage_ids[:length_passage])
    else:
        input_ids = [tokenizer.cls_token_id] + passage_ids + [tokenizer.sep_token_id] + table_ids
        input_ids = input_ids[:args.max_c_len]

    encoded_inputs = {}
    encoded_inputs['input_ids'] = input_ids
    segment_ids = tokenizer.create_segment_token_type_ids_from_sequences(passage_ids, table_data)
    column_ids = tokenizer.create_column_token_type_ids_from_sequences(passage_ids, table_data)
    row_ids = tokenizer.create_row_token_type_ids_from_sequences(passage_ids, table_data)
    prev_labels = [0] * len(row_ids)

    raw_table = add_numeric_table_values(table)
    raw_passage = add_numeric_values_to_question(passage)
    column_ranks, inv_column_ranks = tokenizer._get_numeric_column_ranks(column_ids, row_ids, raw_table)
    numeric_relations = tokenizer._get_numeric_relations(raw_passage, column_ids, row_ids, raw_table)

    token_type_ids = [
        segment_ids,
        column_ids,
        row_ids,
        prev_labels,
        column_ranks,
        inv_column_ranks,
        numeric_relations,
    ]

    token_type_ids = [list(ids) for ids in list(zip(*token_type_ids))]
    encoded_inputs["token_type_ids"] = token_type_ids

    attention_mask = tokenizer.create_attention_mask_from_sequences(passage_ids, table_data)
    encoded_inputs["attention_mask"] = attention_mask

    # Check lengths
    if len(encoded_inputs["input_ids"]) > tokenizer.model_max_length:
        logger.warning(
                f"Token indices sequence length is longer than the specified maximum sequence length "
                f"for this model ({len(encoded_inputs['input_ids'])} > {tokenizer.model_max_length}). Running this "
                "sequence through the model will result in indexing errors.")

    # Padding
    encoded_inputs = tokenizer.pad(encoded_inputs, max_length=args.max_c_len, padding='max_length',
                                   return_attention_mask=True)
    encoded_inputs["length"] = len(encoded_inputs["input_ids"])

    batch_outputs = BatchEncoding(encoded_inputs, tensor_type='pt', prepend_batch_axis=True)

    return batch_outputs


class TRDataset(Dataset):

    def __init__(self,
                 tokenizer,
                 data_path,
                 args,
                 train=False,
                 ):
        super().__init__()
        # self.tokenizer = tokenizer
        # self.args = args
        self.train = train
        self.question = []
        self.table_block = []
        self.labels = []

        logger.info(f"Loading data from {data_path}")
        self.data = []
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        logger.info(f"Total sample count {len(self.data)}")

        for js in tqdm(self.data, desc='preparing dataset..'):
            question = js['question']
            if question.endswith("?"):
                question = question[:-1]
            # self.table_block.append(convert_tb_to_features_tapas(' '.join(js['passages']), js['table'], tokenizer=tokenizer, args=args))
            # self.question.append(convert_tb_to_features_tapas(question, pd.DataFrame([]), tokenizer=tokenizer, args=args))
            # self.table_block.append(convert_tb_to_features(' '.join(js['passages']), js['table'], tokenizer=tokenizer, args=args))
            # self.question.append(convert_tb_to_features(question, pd.DataFrame([]), tokenizer=tokenizer, args=args))
            self.labels.append(torch.tensor(js['label']))

    def __getitem__(self, index):
        return {
            "q": self.question[index],
            "tb": self.table_block[index],
            "label": self.labels[index],
        }

    def __len__(self):
        return len(self.data)


class TRDatasetNega(Dataset):

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

    def processing_data(self):
        logger.info(f"Loading data from {self.data_path}")
        with open(self.data_path, 'rb') as f:
            self.data = pickle.load(f)
        # self.data = self.data[:300]
        logger.info(f"Total sample count {len(self.data)}")

        if not self.args.overwrite_cache and self.train and self.args.save_tensor_path and \
                os.path.exists(self.args.save_tensor_path) and len(os.listdir(self.args.save_tensor_path)) == 4:
            self.load_tensor(self.args.save_tensor_path)
        else:
            for js in tqdm(self.data, desc='preparing singleRetriever dataset..'):
                question = js['question']
                if question.endswith("?"):
                    question = question[:-1]
                if 'tapas' in self.args.model_name:
                    self.question.append(convert_tb_to_features_tapas(question, pd.DataFrame([]), tokenizer=self.tokenizer, args=self.args))
                    self.table_block.append(convert_tb_to_features_tapas(' '.join(js['passages']), js['table'], tokenizer=self.tokenizer, args=self.args))
                    self.negative_table_block.append(convert_tb_to_features_tapas(' '.join(js['neg_passages']), js['neg_table'], tokenizer=self.tokenizer, args=self.args))
                else:
                    self.question.append(convert_tb_to_features_bert(question, pd.DataFrame([]), tokenizer=self.tokenizer, args=self.args))
                    # self.question.append(tokenizer.encode_plus(question, max_length=args.max_q_len,
                    #                                            return_tensors='pt', truncation=True, padding=True))
                    self.table_block.append(convert_tb_to_features_bert(js['passages'], js['table'],
                                                                        tokenizer=self.tokenizer, args=self.args))
                    self.negative_table_block.append(convert_tb_to_features_bert(js['neg_passages'], js['neg_table'],
                                                                                 tokenizer=self.tokenizer, args=self.args))

                self.labels.append(torch.tensor(js['label']))
            if self.train and self.args.save_tensor_path:
                self.save_tensor(self.args.save_tensor_path)

    def __getitem__(self, index):
        return {
            "q": self.question[index],
            "tb": self.table_block[index],
            "neg_tb": self.negative_table_block[index],
            "label": self.labels[index],
        }

    def __len__(self):
        return len(self.question)

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


class TRDatasetNegaMeta(TRDatasetNega):

    def __init__(self, tokenizer, data_path, args, train=False):
        super(TRDatasetNegaMeta, self).__init__(tokenizer, data_path, args, train)

    def processing_data(self):
        logger.info(f"Loading data from {self.data_path}")
        with open(self.data_path, 'rb') as f:
            self.data = list(pickle.load(f))
        # self.data = self.data[:300]
        logger.info(f"Total sample count {len(self.data)}")

        if not self.args.overwrite_cache and self.train and self.args.save_tensor_path and \
                os.path.exists(self.args.save_tensor_path) and len(os.listdir(self.args.save_tensor_path)) == 4:
            self.load_tensor(self.args.save_tensor_path)
        else:
            # for js in tqdm(self.data, desc='preparing singleRetriever dataset with metadata..'):
            #     question = js['question']
            #     if question.endswith("?"):
            #         question = question[:-1]
            #     if 'tapas' in self.args.model_name:
            #         self.question.append(convert_tb_to_features_tapas_metadata(question, pd.DataFrame([]), js['meta_data'], tokenizer=self.tokenizer, args=self.args))
            #         psg = ' '.join(get_passages(js, self.args.psg_mode, neg=False))
            #         self.table_block.append(convert_tb_to_features_tapas_metadata(psg, js['table'], js['meta_data'], tokenizer=self.tokenizer, args=self.args))
            #         psg = ' '.join(get_passages(js, self.args.psg_mode, neg=True))
            #         self.negative_table_block.append(convert_tb_to_features_tapas_metadata(psg, js['neg_table'], js['meta_data'], tokenizer=self.tokenizer, args=self.args))
            #     else:
            #         self.question.append(convert_tb_to_features_bert_metadata(question, pd.DataFrame([]), js['meta_data'],
            #                                                                   tokenizer=self.tokenizer, args=self.args))
            #         # self.question.append(tokenizer.encode_plus(question, max_length=args.max_q_len,
            #         #                                            return_tensors='pt', truncation=True, padding=True))
            #         psg = ' '.join(get_passages(js, self.args.psg_mode, neg=False))
            #         self.table_block.append(convert_tb_to_features_bert_metadata(psg, js['table'], js['meta_data'],
            #                                                             tokenizer=self.tokenizer, args=self.args))
            #         psg = ' '.join(get_passages(js, self.args.psg_mode, neg=True))
            #         self.negative_table_block.append(convert_tb_to_features_bert_metadata(psg, js['neg_table'], js['meta_data'],
            #                                                                      tokenizer=self.tokenizer, args=self.args))
            #     self.labels.append(torch.tensor(js['label']))
            def running_function(js, args, tokenizer):
                output = {}
                question = js['question']
                if question.endswith("?"):
                    question = question[:-1]
                if 'tapas' in args.model_name:
                    output['question'] = convert_tb_to_features_tapas_metadata(question, pd.DataFrame([]), js['meta_data'], tokenizer=tokenizer, args=args)
                    psg = get_passages(js, args.psg_mode, neg=False)[:8]
                    output['table_block'] = convert_tb_to_features_tapas_metadata(psg, js['table'], js['meta_data'], tokenizer=tokenizer, args=args)
                    psg = get_passages(js, args.psg_mode, neg=True)[:8]
                    output['negative_table_block'] = convert_tb_to_features_tapas_metadata(psg, js['neg_table'], js['meta_data'], tokenizer=tokenizer, args=args)
                else:
                    output['question'] = convert_tb_to_features_bert_metadata(question, pd.DataFrame([]), js['meta_data'],
                                                                              tokenizer=tokenizer, args=args)
                    # output['question'] = tokenizer.encode_plus(question, max_length=args.max_q_len,
                    #                                            return_tensors='pt', truncation=True, padding=True)
                    psg = get_passages(js, args.psg_mode, neg=False)[:8]
                    output['table_block'] = convert_tb_to_features_bert_metadata(psg, js['table'], js['meta_data'],
                                                                        tokenizer=tokenizer, args=args)
                    psg = get_passages(js, args.psg_mode, neg=True)[:8]
                    output['negative_table_block'] = convert_tb_to_features_bert_metadata(psg, js['neg_table'], js['meta_data'],
                                                                                 tokenizer=tokenizer, args=args)
                output['labels'] = torch.tensor(js['label'])
                return output
            logger.info("normalize table text")
            # with Pool(self.args.num_workers) as p:
            #     func_ = partial(running_function, args=self.args, tokenizer=self.tokenizer)
            #     all_results = list(tqdm(p.imap(func_, self.data, chunksize=16), total=len(self.data),
            #                             desc="preparing singleRetriever dataset with metadata with multiprocessing..", ))
            #     self.question = [res['question'] for res in all_results]
            #     self.table_block = [res['table_block'] for res in all_results]
            #     self.negative_table_block = [res['negative_table_block'] for res in all_results]
            #     self.labels = [res['labels'] for res in all_results]
            for js in tqdm(self.data, total=len(self.data), desc="preparing singleRetriever dataset with metadata..", ):
                res = running_function(js, self.args, self.tokenizer)
                self.question.append(res['question'])
                self.table_block.append(res['table_block'])
                self.negative_table_block.append(res['negative_table_block'])
                self.labels.append(res['labels'])

            if self.train and self.args.save_tensor_path:
                self.save_tensor(self.args.save_tensor_path)

    # def add_augment_data(self, file):
    #     augment_data = load_jsonl(file)
    #     # augment_data = augment_data[:10000]
    #
    #     def running_function(js, args, tokenizer):
    #         output = {}
    #         question = js['question']
    #         if question.endswith("?"):
    #             question = question[:-1]
    #         output['question'] = convert_tb_to_features_bert_metadata(question, pd.DataFrame([]), None, tokenizer=tokenizer, args=args)
    #         output['table_block'] = convert_tb_to_features_bert_metadata(js['passages'], pd.DataFrame([]), None, tokenizer=tokenizer, args=args)
    #         output['negative_table_block'] = convert_tb_to_features_bert_metadata(js['neg_passages'], pd.DataFrame([]), None, tokenizer=tokenizer, args=args)
    #         output['labels'] = torch.tensor(js['label'])
    #         return output
    #     # with Pool(self.args.num_workers) as p:
    #     #     func_ = partial(running_function, args=self.args, tokenizer=self.tokenizer)
    #     #     all_results = list(tqdm(p.imap(func_, augment_data, chunksize=16), total=len(augment_data),
    #     #                             desc="preparing augmented dataset with multiprocessing..", ))
    #     #     self.question.extend([res['question'] for res in all_results])
    #     #     self.table_block.extend([res['table_block'] for res in all_results])
    #     #     self.negative_table_block.extend([res['negative_table_block'] for res in all_results])
    #     #     self.labels.extend([res['labels'] for res in all_results])
    #     for js in tqdm(augment_data, total=len(augment_data),
    #                    desc="preparing augment_data dataset  with metadata..", ):
    #         res = running_function(js, self.args, self.tokenizer)
    #         self.question.append(res['question'])
    #         self.table_block.append(res['table_block'])
    #         self.negative_table_block.append(res['negative_table_block'])
    #         self.labels.append(res['labels'])
    #
    #     logger.info("after augmentation: num: {}".format(len(self.question)))


class TRDatasetNegaMetaThreeCat(TRDatasetNega):

    def __init__(self, tokenizer, data_path, args, train=False):
        super(TRDatasetNegaMetaThreeCat, self).__init__(tokenizer, data_path, args, train)

    def processing_data(self):
        logger.info(f"Loading data from {self.data_path}")
        with open(self.data_path, 'rb') as f:
            self.data = list(pickle.load(f))
        # self.data = self.data[:500]
        logger.info(f"Total sample count {len(self.data)}")

        if not self.args.overwrite_cache and self.train and self.args.save_tensor_path and \
                os.path.exists(self.args.save_tensor_path) and len(os.listdir(self.args.save_tensor_path)) == 4:
            self.load_tensor(self.args.save_tensor_path)
        else:
            def running_function(js, args, tokenizer):
                output = {}
                question = js['question']
                if question.endswith("?"):
                    question = question[:-1]
                if 'tapas' in args.model_name:
                    raise Exception ('did not implement tapas with threecat setting')
                    # output['question'] = convert_tb_to_features_tapas_metadata(question, pd.DataFrame([]), js['meta_data'], tokenizer=tokenizer, args=args)
                    # psg = get_passages(js, args.psg_mode, neg=False)[:8]
                    # output['table_block'] = convert_tb_to_features_tapas_metadata(psg, js['table'], js['meta_data'], tokenizer=tokenizer, args=args)
                    # psg = get_passages(js, args.psg_mode, neg=True)[:8]
                    # output['negative_table_block'] = convert_tb_to_features_tapas_metadata(psg, js['neg_table'], js['meta_data'], tokenizer=tokenizer, args=args)
                else:
                    if args.one_query:
                        function = convert_tb_to_features_bert_metadata_threecat_one_query
                    else:
                        function = convert_tb_to_features_bert_metadata_threecat
                    output['question'] = function(question, pd.DataFrame([]), js['meta_data'], tokenizer=tokenizer, args=args)
                    # output['question'] = tokenizer.encode_plus(question, max_length=args.max_q_len,
                    #                                            return_tensors='pt', truncation=True, padding=True)
                    psg = get_passages(js, args.psg_mode, neg=False)[:8]
                    output['table_block'] = function(psg, js['table'], js['meta_data'], tokenizer=tokenizer, args=args)
                    psg = get_passages(js, args.psg_mode, neg=True)[:8]
                    output['negative_table_block'] = function(psg, js['neg_table'], js['meta_data'], tokenizer=tokenizer, args=args)
                output['labels'] = torch.tensor(js['label'])
                return output

            logger.info("normalize table text")
            # with Pool(self.args.num_workers) as p:
            #     func_ = partial(running_function, args=self.args, tokenizer=self.tokenizer)
            #     all_results = list(tqdm(p.imap(func_, self.data, chunksize=16), total=len(self.data),
            #                             desc="preparing singleRetriever ThreeCat dataset with metadata with multiprocessing..", ))
            #     self.question = [res['question'] for res in all_results]
            #     self.table_block = [res['table_block'] for res in all_results]
            #     self.negative_table_block = [res['negative_table_block'] for res in all_results]
            #     self.labels = [res['labels'] for res in all_results]
            for js in tqdm(self.data, total=len(self.data), desc="preparing singleRetriever ThreeCat dataset with metadata..", ):
                res = running_function(js, self.args, self.tokenizer)
                self.question.append(res['question'])
                self.table_block.append(res['table_block'])
                self.negative_table_block.append(res['negative_table_block'])
                self.labels.append(res['labels'])

            if self.train and self.args.save_tensor_path:
                self.save_tensor(self.args.save_tensor_path)

    def add_augment_data(self, file):
        augment_data = load_jsonl(file)
        # augment_data = augment_data[:500]

        def running_function(js, args, tokenizer):
            output = {}
            question = js['question']
            if question.endswith("?"):
                question = question[:-1]

            if args.one_query:
                function = convert_tb_to_features_bert_metadata_threecat_one_query
            else:
                function = convert_tb_to_features_bert_metadata_threecat
            output['question'] = function(question, pd.DataFrame([]), None, tokenizer=tokenizer, args=args)
            output['table_block'] = convert_aug_tb_to_features_bert_threecat(js['passages'], tokenizer=tokenizer, args=args)
            output['negative_table_block'] = convert_aug_tb_to_features_bert_threecat(js['neg_passages'], tokenizer=tokenizer, args=args)
            output['labels'] = torch.tensor(js['label'])
            return output
        # with Pool(self.args.num_workers) as p:
        #     func_ = partial(running_function, args=self.args, tokenizer=self.tokenizer)
        #     all_results = list(tqdm(p.imap(func_, augment_data, chunksize=16), total=len(augment_data),
        #                             desc="preparing augmented dataset with multiprocessing..", ))
        #     self.question.extend([res['question'] for res in all_results])
        #     self.table_block.extend([res['table_block'] for res in all_results])
        #     self.negative_table_block.extend([res['negative_table_block'] for res in all_results])
        #     self.labels.extend([res['labels'] for res in all_results])
        for js in tqdm(augment_data, total=len(augment_data),
                       desc="preparing augment_data dataset threecat with metadata..", ):
            res = running_function(js, self.args, self.tokenizer)
            self.question.append(res['question'])
            self.table_block.append(res['table_block'])
            self.negative_table_block.append(res['negative_table_block'])
            self.labels.append(res['labels'])

        logger.info("after augmentation: num: {}".format(len(self.question)))

        # if self.train and self.args.save_tensor_path:
        #     self.save_tensor(self.args.save_tensor_path)



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


def tb_collate_tapas(samples, pad_id=0):
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
            'q_type_ids': collate_tokens([s["q"]["token_type_ids"] for s in samples], 0,'2d'),
            'c_type_ids': collate_tokens([s["tb"]["token_type_ids"] for s in samples], 0,'2d'),
            'neg_type_ids': collate_tokens([s["neg_tb"]["token_type_ids"] for s in samples], 0,'2d'),
        })

    return batch


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
