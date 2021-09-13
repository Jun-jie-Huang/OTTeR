# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import collections
import json
import random
import pandas as pd
import numpy as np
from transformers.data.processors.utils import DataProcessor
from transformers.models.bert.tokenization_bert import whitespace_tokenize
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase, TruncationStrategy
from torch.utils.data import TensorDataset
import torch
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm
import os, pickle
from multiprocessing import Pool, cpu_count
from functools import partial
from collections import Counter

import logging
logger = logging.getLogger(__name__)

from qa.qa_dataset import _is_whitespace, ottqa_convert_examples_to_features, OTTQAExample
from qa.utils import read_jsonl
import sys
sys.path.append('../')
from utils.common import convert_tb_to_string_metadata, convert_tb_to_string, get_passages, convert_table_to_string



class OTTQAProcessor(DataProcessor):
    def __init__(self, args):
        super(OTTQAProcessor, self).__init__()
        self.args = args

    def get_qa_data(self, input_data, filename):
        qa_data_path = os.path.join(self.args.data_dir, filename + '_qa')
        if (not os.path.exists(qa_data_path)) or self.args.repreprocess:
            if self.args.append_table_to_question:
                qa_data = prepare_qa_data_newquestion(input_data, self.args.psg_mode)
                logger.info("OTTQAProcessor prepare_qa_data_newquestion")
            else:
                qa_data = prepare_qa_data(input_data, self.args.psg_mode)

            with open(qa_data_path, 'w', encoding='utf8') as outf:
                json.dump(qa_data, outf)
            logger.info("OTTQAProcessor dumping qa data to {}".format(qa_data_path))
        else:
            qa_data = json.load(open(qa_data_path, 'r', encoding='utf8'))
            logger.info("OTTQAProcessor loading qa data from {}".format(qa_data_path))
        return qa_data

    # def get_train_examples(self, filename=None):
    #     logger.info("loading input data from {}".format(os.path.join(self.args.data_dir, filename)))
    #     # input_data = readGZip(filename)
    #     with open(os.path.join(self.args.data_dir, filename), "r", encoding="utf-8") as reader:
    #         input_data = json.load(reader)
    #     qa_data = self.get_qa_data(input_data, filename)
    #     logger.info("{}:length {}/{}".format(filename, len(input_data), len(qa_data)))
    #     return self._create_examples(qa_data, "train")
    #
    # def get_dev_examples(self, filename=None):
    #     # input_data = readGZip(filename)
    #     with open(os.path.join(self.args.data_dir, filename), "r", encoding="utf-8") as reader:
    #         input_data = json.load(reader)
    #     qa_data = self.get_qa_data(input_data, filename)
    #     logger.info("{}:length {}/{}".format(filename, len(input_data), len(qa_data)))
    #     return self._create_examples(qa_data, "dev")

    def get_eval_examples(self, filename=None, split="dev"):
        # input_data = readGZip(filename)
        input_data = read_jsonl(os.path.join(self.args.data_dir, filename))
        # with open(os.path.join(self.args.data_dir, filename), "r", encoding="utf-8") as reader:
        #     input_data = json.load(reader)
        qa_data = self.get_qa_data(input_data, filename)
        logger.info("{}:length {}/{}".format(filename, len(input_data), len(qa_data)))
        return self._create_examples(qa_data, split)

    def _create_examples(self, qa_data, set_type):

        # is_training = set_type == "train"
        examples = []
        for entry in tqdm(qa_data, desc='   Creating examples: '):

            # import pdb
            # pdb.set_trace()
            title = entry["title"]
            context_text = entry["context"]
            qas_id = entry["question_id"]
            question_text = entry["question"]
            start_position_character = None
            answer_text = None
            answers = []

            if "is_impossible" in entry:
                is_impossible = entry["is_impossible"]
            else:
                is_impossible = False

            if not is_impossible:
                answers = entry["answers"]

            example = OTTQAExample(
                qas_id=qas_id,
                question_text=question_text,
                context_text=context_text,
                answer_text=answer_text,
                start_position_character=start_position_character,
                title=title,
                is_impossible=is_impossible,
                answers=answers,
            )

            examples.append(example)

        return examples


class OTTQAProcessorConcat(OTTQAProcessor):
    def __init__(self,args):
        super(OTTQAProcessorConcat, self).__init__(args)

    def get_qa_data(self, input_data, filename):
        qa_data_path = os.path.join(self.args.data_dir, filename + '_qa')
        if (not os.path.exists(qa_data_path)) or self.args.repreprocess:
            qa_data = prepare_qa_data_tbconcat(input_data, self.args.psg_mode, self.args.qa_topk_concat)
            with open(qa_data_path, 'w', encoding='utf8') as outf:
                json.dump(qa_data, outf)
            logger.info("OTTQAProcessorConcat dumping qa data to {}".format(qa_data_path))
        else:
            qa_data = json.load(open(qa_data_path, 'r', encoding='utf8'))
            logger.info("OTTQAProcessorConcat loading qa data from {}".format(qa_data_path))
        return qa_data


class OTTQAFeatures:
    """
    Single squad example features to be fed to a model. Those features are model-specific and can be crafted from
    :class:`~transformers.data.processors.squad.SquadExample` using the
    :method:`~transformers.data.processors.squad.squad_convert_examples_to_features` method.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        cls_index: the index of the CLS token.
        p_mask: Mask identifying tokens that can be answers vs. tokens that cannot.
            Mask with 1 for tokens than cannot be in the answer and 0 for token that can be in an answer
        example_index: the index of the example
        unique_id: The unique Feature identifier
        paragraph_len: The length of the context
        token_is_max_context: List of booleans identifying which tokens have their maximum context in this feature object.
            If a token does not have their maximum context in this feature object, it means that another feature object
            has more information related to that token and should be prioritized over this feature for that token.
        tokens: list of tokens corresponding to the input ids
        token_to_orig_map: mapping between the tokens and the original text, needed in order to identify the answer.
        start_position: start of the answer token index
        end_position: end of the answer token index
        encoding: optionally store the BatchEncoding with the fast-tokenizer alignment methods.
    """

    def __init__(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
            cls_index,
            p_mask,
            example_index,
            unique_id,
            paragraph_len,
            token_is_max_context,
            tokens,
            token_to_orig_map,
            start_position,
            end_position,
            is_impossible,
            qas_id: str = None,
            encoding: BatchEncoding = None,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.cls_index = cls_index
        self.p_mask = p_mask

        self.example_index = example_index
        self.unique_id = unique_id
        self.paragraph_len = paragraph_len
        self.token_is_max_context = token_is_max_context
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map

        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.qas_id = qas_id

        self.encoding = encoding


def get_table_block(node, documents):
    table_block = documents[node[2]]
    table_block += 'Title : {} . '.format(node[0]) + table_block
    return table_block


# def get_test_data(args):
#     with open(args.request_path, 'r') as f:
#         requests = json.load(f)
#
#     # evaluate(args, model, tokenizer, prefix=global_step)
#     with open(args.predict_file, 'r') as f:
#         data = json.load(f)
#
#     full_split = []
#     key2idx = {}
#     for step, d in enumerate(data):
#         if isinstance(d['pred'], str):
#             continue
#         table_id = d['table_id']
#         node = d['pred']
#         context = 'Title : {} . {}'.format(node[0], requests[node[1]])
#         context = get_table_block(node[0], requests[node[1]])
#         full_split.append({'context': context, 'title': table_id,
#                            'question': d['question'], 'question_id': d['question_id'],
#                            'answers': [{'answer_start': None, 'text': None}]})
#         key2idx[d['question_id']] = step
#
#     processor = OTTQAProcessor(args)
#     examples = processor._create_examples(full_split, 'dev')
#
#     # logger.info("Preprocessing {} examples".format(len(examples)))
#     features, dataset = ottqa_convert_examples_to_features(
#         examples=examples,
#         tokenizer=tokenizer,
#         max_seq_length=args.max_seq_length,
#         doc_stride=args.doc_stride,
#         max_query_length=args.max_query_length,
#         is_training=False,
#         threads=args.threads,
#     )
#     return dataset, examples, features, key2idx


def prepare_qa_data(data, psg_mode='ori', max_context_length=460):
    '''
        tokenizer-free, 后面要改输入输出就改这里就行了
        OverLoading...
    Input file format
    [{
        'question_id': str
        'question': str
        'answer-text': answer-text  ???test
        'answer-nodes': []
        'top_{}".format(args.beam_size)': [{}]

        # 'positive_table_blocks‘: {'table_id','row_id','table_segment':,'passages':[passage text]}
        # 'passages': [{'index':str , 'position':[row_id,col_id], 'passage':str}]
        # 'table': table
     }, {}]

    Output data format
    [{
        'context': str,
        'title': str,
        'question': str,
        'question_id': str,

        'answers': [{ 'answer_start': int,
                      'text': int
                      }]
     }, {}]
    '''

    TOPN = [item for item in list(data[0].keys()) if 'top_' in item][0]
    split = []
    all_question_ids = Counter()
    for d in tqdm(data, desc='   Preparing QA data, do not find answer span: '):

        for block in d[TOPN]:
            header = block['table'][0]
            table = block['table'][1]
            table_segment = pd.DataFrame(data=table, columns=header)
            # passage = [random.choice(block['passages'])] if block['passages'] else ['']
            passage = get_passages(block, psg_mode, neg=False)
            # passage = ' '.join(block['passages'])
            context = convert_tb_to_string(table_segment, passage, cut='passage')

            all_question_ids[d['question_id']] += 1
            question_id = d['question_id'] + '-{}'.format(all_question_ids[d['question_id']])

            answer = d.get('answer-text', '[HEADER]')
            gold_title = d.get('table_id', '[NONE]')
            title = block['table_id']
            # split.append({'context': context, 'title': d['table_id'],
            split.append({'context': context, 'title': title, 'gold_title': gold_title,
                          'question': d['question'], 'question_id': question_id,
                          'answers': [{'answer_start': 0, 'text': answer}], })

    return split


def prepare_qa_data_newquestion(data, psg_mode='ori', max_context_length=460):

    TOPN = [item for item in list(data[0].keys()) if 'top_' in item][0]
    split = []
    all_question_ids = Counter()
    # # for d in tqdm(data, desc='   Preparing QA data, append table into question do not find answer span: '):
    # threads = min(40, cpu_count())
    # with Pool(threads) as p:
    #     todo_ = partial(_process, psg_mode=psg_mode)
    #     qa_data = list(tqdm(p.imap(todo_, input_data, chunksize=32), total=len(input_data),
    #                         desc="   Convert dev output to qa data"))
    #
    # def _process(d, psg_mode):
    #     new_qa_data =[]
    #     for block in d[TOPN]:
    #         header = block['table'][0]
    #         table = block['table'][1]
    #         table_segment = pd.DataFrame(data=table, columns=header)
    #         table_str = convert_table_to_string(table_segment)
    #         tabled_question = "{} {}".format(d['question'], table_str)
    #         # passage = [random.choice(block['passages'])] if block['passages'] else ['']
    #         passages = get_passages(block, psg_mode, neg=False)
    #         passages_str = ' [PASSAGE] ' + ' [SEP] '.join(passages)
    #
    #         all_question_ids[d['question_id']] += 1
    #         question_id = d['question_id'] + '-{}'.format(all_question_ids[d['question_id']])
    #
    #         answer = d.get('answer-text', '[HEADER]')
    #         gold_title = d.get('table_id', '[NONE]')
    #         title = block['table_id']
    #
    #
    #         new_qa_data.append({'context': passages_str, 'title': title, 'gold_title': gold_title,
    #                             'question': tabled_question, 'question_id': question_id,
    #                             'answers': [{'answer_start': 0, 'text': answer}], })
    for d in tqdm(data, desc='   Preparing QA data, append table into question do not find answer span: '):

        for block in d[TOPN]:
            header = block['table'][0]
            table = block['table'][1]
            table_segment = pd.DataFrame(data=table, columns=header)
            table_str = convert_table_to_string(table_segment)
            tabled_question = "{} {}".format(d['question'], table_str)
            # passage = [random.choice(block['passages'])] if block['passages'] else ['']
            passages = get_passages(block, psg_mode, neg=False)
            passages_str = ' [PASSAGE] ' + ' [SEP] '.join(passages)

            all_question_ids[d['question_id']] += 1
            question_id = d['question_id'] + '-{}'.format(all_question_ids[d['question_id']])

            answer = d.get('answer-text', '[HEADER]')
            gold_title = d.get('table_id', '[NONE]')
            title = block['table_id']
            split.append({'context': passages_str, 'title': title, 'gold_title': gold_title,
                          'question': tabled_question, 'question_id': question_id,
                          'answers': [{'answer_start': 0, 'text': answer}], })

    return split


def prepare_qa_data_tbconcat(data, psg_mode='ori', topk_tbs=5, max_context_length=400, add_ground_truth_table=False):

    TOPN = [item for item in list(data[0].keys()) if 'top_' in item][0]
    split = []

    for d in tqdm(data, desc='   Preparing QA data with {} tbs, do not find answer span: '.format(topk_tbs)):
        all_context = []
        for block in d[TOPN][:topk_tbs]:
            header = block['table'][0]
            table = block['table'][1]
            table_segment = pd.DataFrame(data=table, columns=header)
            passages = get_passages(block, psg_mode, neg=False)
            # passage = random.choice(block['passages']) if block['passages'] else ''
            # passage = ' '.join(block['passages'])
            context = convert_tb_to_string(table_segment, passages, cut='passage')
            title = block['table_id']
            all_context.append({'context': context, 'title': title})

        question_str = d['question']
        question_id = d['question_id']
        answer = d.get('answer-text', '[HEADER]')
        gold_title = d.get('table_id', '[NONE]')
        selected_context = all_context[:topk_tbs]
        context_str = ' [SEP] '.join([c['context'] for c in selected_context])

        split.append({'context': context_str, 'title': [c['title'] for c in selected_context],
                      'gold_title': gold_title,
                      'question': question_str, 'question_id': question_id,
                      'answers': [{'answer_start': 0, 'text': answer}], })
    return split

# def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
#     """Returns tokenized answer spans that better match the annotated answer."""
#     tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))
#
#     for new_start in range(input_start, input_end + 1):
#         for new_end in range(input_end, new_start - 1, -1):
#             text_span = " ".join(doc_tokens[new_start: (new_end + 1)])
#             if text_span == tok_answer_text:
#                 return (new_start, new_end)
#
#     return (input_start, input_end)


# def _new_check_is_max_context(doc_spans, cur_span_index, position):
#     """Check if this is the 'max context' doc span for the token."""
#     # if len(doc_spans) == 1:
#     # return True
#     best_score = None
#     best_span_index = None
#     for (span_index, doc_span) in enumerate(doc_spans):
#         end = doc_span["start"] + doc_span["length"] - 1
#         if position < doc_span["start"]:
#             continue
#         if position > end:
#             continue
#         num_left_context = position - doc_span["start"]
#         num_right_context = end - position
#         score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
#         if best_score is None or score > best_score:
#             best_score = score
#             best_span_index = span_index
#
#     return cur_span_index == best_span_index


def load_and_cache_examples(args, tokenizer, test=False, output_examples=True, cache=True):
    # OverLoading...

    # Load data features from cache or dataset file
    input_dir = args.data_dir if args.data_dir else "."
    cached_features_file = os.path.join(
        input_dir,
        "cached_eval_{}_{}_{}".format(
            "test" if test else "dev",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
        ),
    )

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        logger.info("Creating features from dataset file at %s to %s", input_dir, cached_features_file)

        if args.qa_topk_concat == 0:
            processor = OTTQAProcessor(args)
        else:
            processor = OTTQAProcessorConcat(args)
        if test:
            examples = processor.get_eval_examples(args.test_file, "test")
        else:
            examples = processor.get_eval_examples(args.dev_file, "dev")

        features, dataset = ottqa_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=False,
            threads=args.threads,
            padding_strategy="max_length"
        ) # TODO
        if args.local_rank in [-1, 0] and args.dont_save_cache:
        # if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)
            logger.info("Saving features finished")
        else:
            logger.info("dont save features")

    if output_examples:
        return dataset, examples, features

    return dataset


# MULTI_SEP_TOKENS_TOKENIZERS_SET = {"roberta", "camembert", "bart", "mpnet"}


# def ottqa_convert_example_to_features(
#         example, max_seq_length, doc_stride, max_query_length, padding_strategy, is_training
# ):
#     features = []
#     if is_training and not example.is_impossible:
#         # Get start and end position
#         start_position = example.start_position
#         end_position = example.end_position
#
#         # If the answer cannot be found in the text, then skip this example.
#         actual_text = " ".join(example.doc_tokens[start_position: (end_position + 1)])
#         cleaned_answer_text = " ".join(whitespace_tokenize(example.answer_text))
#         if actual_text.find(cleaned_answer_text) == -1:
#             logger.warning(f"Could not find answer: '{actual_text}' vs. '{cleaned_answer_text}'")
#             return []
#
#     tok_to_orig_index = []
#     orig_to_tok_index = []
#     all_doc_tokens = []
#     for (i, token) in enumerate(example.doc_tokens):
#         orig_to_tok_index.append(len(all_doc_tokens))
#         if tokenizer.__class__.__name__ in [
#             "RobertaTokenizer",
#             "LongformerTokenizer",
#             "BartTokenizer",
#             "RobertaTokenizerFast",
#             "LongformerTokenizerFast",
#             "BartTokenizerFast",
#         ]:
#             sub_tokens = tokenizer.tokenize(token, add_prefix_space=True)
#         else:
#             sub_tokens = tokenizer.tokenize(token)
#         for sub_token in sub_tokens:
#             tok_to_orig_index.append(i)
#             all_doc_tokens.append(sub_token)
#
#     if is_training and not example.is_impossible:
#         tok_start_position = orig_to_tok_index[example.start_position]
#         if example.end_position < len(example.doc_tokens) - 1:
#             tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
#         else:
#             tok_end_position = len(all_doc_tokens) - 1
#
#         (tok_start_position, tok_end_position) = _improve_answer_span(
#             all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.answer_text
#         )
#
#     spans = []
#
#     truncated_query = tokenizer.encode(
#         example.question_text, add_special_tokens=False, truncation=True, max_length=max_query_length
#     )
#
#     # Tokenizers who insert 2 SEP tokens in-between <context> & <question> need to have special handling
#     # in the way they compute mask of added tokens.
#     tokenizer_type = type(tokenizer).__name__.replace("Tokenizer", "").lower()
#     sequence_added_tokens = (
#         tokenizer.model_max_length - tokenizer.max_len_single_sentence + 1
#         if tokenizer_type in MULTI_SEP_TOKENS_TOKENIZERS_SET
#         else tokenizer.model_max_length - tokenizer.max_len_single_sentence
#     )
#     sequence_pair_added_tokens = tokenizer.model_max_length - tokenizer.max_len_sentences_pair
#
#     span_doc_tokens = all_doc_tokens
#     while len(spans) * doc_stride < len(all_doc_tokens):
#
#         # Define the side we want to truncate / pad and the text/pair sorting
#         if tokenizer.padding_side == "right":
#             texts = truncated_query
#             pairs = span_doc_tokens
#             truncation = TruncationStrategy.ONLY_SECOND.value
#         else:
#             texts = span_doc_tokens
#             pairs = truncated_query
#             truncation = TruncationStrategy.ONLY_FIRST.value
#
#         encoded_dict = tokenizer.encode_plus(  # TODO(thom) update this logic
#             texts,
#             pairs,
#             truncation=truncation,
#             padding=padding_strategy,
#             max_length=max_seq_length,
#             return_overflowing_tokens=True,
#             stride=max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens,
#             return_token_type_ids=True,
#         )
#
#         paragraph_len = min(
#             len(all_doc_tokens) - len(spans) * doc_stride,
#             max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
#         )
#
#         if tokenizer.pad_token_id in encoded_dict["input_ids"]:
#             if tokenizer.padding_side == "right":
#                 non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
#             else:
#                 last_padding_id_position = (
#                         len(encoded_dict["input_ids"]) - 1 - encoded_dict["input_ids"][::-1].index(
#                     tokenizer.pad_token_id)
#                 )
#                 non_padded_ids = encoded_dict["input_ids"][last_padding_id_position + 1:]
#
#         else:
#             non_padded_ids = encoded_dict["input_ids"]
#
#         tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)
#
#         token_to_orig_map = {}
#         for i in range(paragraph_len):
#             index = len(truncated_query) + sequence_added_tokens + i if tokenizer.padding_side == "right" else i
#             token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]
#
#         encoded_dict["paragraph_len"] = paragraph_len
#         encoded_dict["tokens"] = tokens
#         encoded_dict["token_to_orig_map"] = token_to_orig_map
#         encoded_dict["truncated_query_with_special_tokens_length"] = len(truncated_query) + sequence_added_tokens
#         encoded_dict["token_is_max_context"] = {}
#         encoded_dict["start"] = len(spans) * doc_stride
#         encoded_dict["length"] = paragraph_len
#
#         spans.append(encoded_dict)
#
#         if "overflowing_tokens" not in encoded_dict or (
#                 "overflowing_tokens" in encoded_dict and len(encoded_dict["overflowing_tokens"]) == 0
#         ):
#             break
#         span_doc_tokens = encoded_dict["overflowing_tokens"]
#
#     for doc_span_index in range(len(spans)):
#         for j in range(spans[doc_span_index]["paragraph_len"]):
#             is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
#             index = (
#                 j
#                 if tokenizer.padding_side == "left"
#                 else spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j
#             )
#             spans[doc_span_index]["token_is_max_context"][index] = is_max_context
#
#     for span in spans:
#         # Identify the position of the CLS token
#         cls_index = span["input_ids"].index(tokenizer.cls_token_id)
#
#         # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
#         # Original TF implem also keep the classification token (set to 0)
#         p_mask = np.ones_like(span["token_type_ids"])
#         if tokenizer.padding_side == "right":
#             p_mask[len(truncated_query) + sequence_added_tokens:] = 0
#         else:
#             p_mask[-len(span["tokens"]): -(len(truncated_query) + sequence_added_tokens)] = 0
#
#         pad_token_indices = np.where(span["input_ids"] == tokenizer.pad_token_id)
#         special_token_indices = np.asarray(
#             tokenizer.get_special_tokens_mask(span["input_ids"], already_has_special_tokens=True)
#         ).nonzero()
#
#         p_mask[pad_token_indices] = 1
#         p_mask[special_token_indices] = 1
#
#         # Set the cls index to 0: the CLS index can be used for impossible answers
#         p_mask[cls_index] = 0
#
#         span_is_impossible = example.is_impossible
#         start_position = 0
#         end_position = 0
#         if is_training and not span_is_impossible:
#             # For training, if our document chunk does not contain an annotation
#             # we throw it out, since there is nothing to predict.
#             doc_start = span["start"]
#             doc_end = span["start"] + span["length"] - 1
#             out_of_span = False
#
#             if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
#                 out_of_span = True
#
#             if out_of_span:
#                 start_position = cls_index
#                 end_position = cls_index
#                 span_is_impossible = True
#             else:
#                 if tokenizer.padding_side == "left":
#                     doc_offset = 0
#                 else:
#                     doc_offset = len(truncated_query) + sequence_added_tokens
#
#                 start_position = tok_start_position - doc_start + doc_offset
#                 end_position = tok_end_position - doc_start + doc_offset
#
#         features.append(
#             OTTQAFeatures(
#                 span["input_ids"],
#                 span["attention_mask"],
#                 span["token_type_ids"],
#                 cls_index,
#                 p_mask.tolist(),
#                 example_index=0,
#                 # Can not set unique_id and example_index here. They will be set after multiple processing.
#                 unique_id=0,
#                 paragraph_len=span["paragraph_len"],
#                 token_is_max_context=span["token_is_max_context"],
#                 tokens=span["tokens"],
#                 token_to_orig_map=span["token_to_orig_map"],
#                 start_position=start_position,
#                 end_position=end_position,
#                 is_impossible=span_is_impossible,
#                 qas_id=example.qas_id,
#             )
#         )
#     return features


# def ottqa_convert_example_to_features_init(tokenizer_for_convert):
#     global tokenizer
#     tokenizer = tokenizer_for_convert
#
#
# def ottqa_convert_examples_to_features(examples, tokenizer, max_seq_length, doc_stride, max_query_length,
#                                        is_training, threads=1, padding_strategy="max_length"):
#     # Defining helper methods
#     features = []
#     threads = min(threads, cpu_count())
#     with Pool(threads, initializer=ottqa_convert_example_to_features_init, initargs=(tokenizer, )) as p:
#         annotate_ = partial(
#             ottqa_convert_example_to_features,
#             max_seq_length=max_seq_length,
#             doc_stride=doc_stride,
#             max_query_length=max_query_length,
#             is_training=is_training,
#             padding_strategy=padding_strategy
#         )
#         features = list(
#             tqdm(
#                 p.imap(annotate_, examples, chunksize=32),
#                 total=len(examples),
#                 desc="   Convert ottqa examples to index features",
#             )
#         )
#
#     new_features = []
#     unique_id = 1000000000
#     example_index = 0
#     for example_features in tqdm(features, total=len(features), desc="   Add example index and unique id"):
#         if not example_features:
#             continue
#         for example_feature in example_features:
#             example_feature.example_index = example_index
#             example_feature.unique_id = unique_id
#             new_features.append(example_feature)
#             unique_id += 1
#         example_index += 1
#     features = new_features
#     del new_features
#
#     # Convert to Tensors and build dataset
#     all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
#     all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
#     all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
#     all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
#     all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
#     all_is_impossible = torch.tensor([f.is_impossible for f in features], dtype=torch.float)
#
#     if not is_training:
#         all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
#         dataset = TensorDataset(
#             all_input_ids, all_attention_masks, all_token_type_ids, all_example_index, all_cls_index, all_p_mask
#         )
#     else:
#         all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
#         all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
#         dataset = TensorDataset(
#             all_input_ids,
#             all_attention_masks,
#             all_token_type_ids,
#             all_start_positions,
#             all_end_positions,
#             all_cls_index,
#             all_p_mask,
#             all_is_impossible,
#         )
#
#     return features, dataset
