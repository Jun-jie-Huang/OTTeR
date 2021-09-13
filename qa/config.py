# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.
import argparse
from ast import parse
from typing import NamedTuple
from torch.nn import parallel

import logging

logger = logging.getLogger(__name__)


class ClusterConfig(NamedTuple):
    dist_backend: str
    dist_url: str


def qa_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--model_type", default='bert', type=str)
    parser.add_argument("--model_name_or_path", default="bert-base-uncased", type=str, help="Path to pre-trained model or shortcut name selected in the list: ")
    parser.add_argument("--output_dir", default='qa_model', type=str, help="The output directory where the model checkpoints and predictions will be written.",)
    parser.add_argument(
            "--train_file",
            default=None,
            type=str,
            help="The input training file. If a data dir is specified, will look for the file there"
                 + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
            "--dev_file",
            default=None,
            type=str,
            help="The input development file. If a data dir is specified, will look for the file there"
                 + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
            "--resource_dir",
            type=str,
            default='data/',
            help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
            "--data_dir",
            type=str,
            default='data/',
            help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
            "--predict_file",
            default=None,
            type=str,
            help="The input evaluation file. If a data dir is specified, will look for the file there"
                 + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
            "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
            "--tokenizer_name",
            default="",
            type=str,
            help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
            "--cache_dir",
            default="/tmp/",
            type=str,
            help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
            "--version_2_with_negative",
            action="store_true",
            help="If true, the SQuAD examples contain some that do not have an answer.",
    )
    parser.add_argument(
            "--null_score_diff_threshold",
            type=float,
            default=0.0,
            help="If null_score - best_non_null is greater than the threshold predict null.",
    )
    parser.add_argument(
            "--max_seq_length",
            default=384,
            type=int,
            help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                 "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
            "--doc_stride",
            default=128,
            type=int,
            help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument(
            "--max_query_length",
            default=64,
            type=int,
            help="The maximum number of tokens for the question. Questions longer than this will "
                 "be truncated to this length.",
    )
    parser.add_argument(
            "--num_tokenizer_vocab",
            default=50271,
            type=int,
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run eval on the dev set.")

    parser.add_argument("--append_table_to_question", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--repreprocess", action="store_true", help="Whether to re-prepare the qa data.")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--dont_save_cache", action="store_false", help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--qa_topk_concat", type=int, default=0, help="number of table blocks used for qa evaluation, if 0 then dont concat")
    parser.add_argument("--metadata", action="store_true", help="whether to add meta data, True(use) if call")
    parser.add_argument("--add_special_tokens", action="store_true", help="whether to add special tokens")
    parser.add_argument("--topk_tbs", type=int, default=10, help="multiple threads for converting example to features")
    parser.add_argument("--prefix", type=str, default='', help="prefix for saving cached file")
    parser.add_argument("--predict_output_file", type=str, default='', help="file to save predictions")
    parser.add_argument("--pred_model_dir", type=str, default='', help="file to save predictions")

    parser.add_argument("--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--n_best_size", default=20, type=int, help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
    parser.add_argument("--max_answer_length", default=30, type=int, help="The maximum length of an answer that can be generated. This is needed because the start and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", action="store_true",
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--lang_id", default=0, type=int,
                        help="language id of input for language-specific xlm models (see tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)")
    parser.add_argument("--request_path", type=str, default='request_tok', help="Request directory.")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--fp16", action="store_true",help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",)
    parser.add_argument("--evaluate_during_training", action="store_true", help="Whether to evaluate during training",)
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. "
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--threads", type=int, default=40, help="multiple threads for converting example to features")
    args = parser.parse_args()
    return args
