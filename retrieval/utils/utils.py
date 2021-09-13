# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.
import torch
import sqlite3
import unicodedata
import numpy as np


def load_saved(model, path, exact=True):
    try:
        state_dict = torch.load(path)
    except:
        state_dict = torch.load(path, map_location=torch.device('cpu'))

    def filter(x):
        return x[7:] if x.startswith('module.') else x

    if exact:
        state_dict = {filter(k): v for (k, v) in state_dict.items()}
    else:
        state_dict = {filter(k): v for (k, v) in state_dict.items() if filter(k) in model.state_dict()}

    # model.load_state_dict(state_dict)
    model.load_state_dict(state_dict, False)
    return model


def move_to_cuda(sample):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda()
        elif isinstance(maybe_tensor, dict):
            return {
                key: _move_to_cuda(value)
                for key, value in maybe_tensor.items()
            }
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        else:
            return maybe_tensor

    return _move_to_cuda(sample)


def convert_to_half(sample):
    if len(sample) == 0:
        return {}

    def _convert_to_half(maybe_floatTensor):
        if torch.is_tensor(maybe_floatTensor) and maybe_floatTensor.type() == "torch.FloatTensor":
            return maybe_floatTensor.half()
        elif isinstance(maybe_floatTensor, dict):
            return {
                key: _convert_to_half(value)
                for key, value in maybe_floatTensor.items()
            }
        elif isinstance(maybe_floatTensor, list):
            return [_convert_to_half(x) for x in maybe_floatTensor]
        else:
            return maybe_floatTensor

    return _convert_to_half(sample)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)


class DocDB(object):
    """Sqlite backed document storage.

    Implements get_doc_text(doc_id).
    """

    def __init__(self, db_path=None):
        self.path = db_path
        self.connection = sqlite3.connect(self.path, check_same_thread=False)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        """Close the connection to the database."""
        self.connection.close()

    def get_doc_ids(self):
        """Fetch all ids of docs stored in the db."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT id FROM documents")
        results = [r[0] for r in cursor.fetchall()]
        cursor.close()
        return results

    def get_doc_text(self, doc_id):
        """Fetch the raw text of the doc for 'doc_id'."""
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT text FROM documents WHERE id = ?",
            (normalize(doc_id),)
        )
        result = cursor.fetchone()
        cursor.close()
        return result if result is None else result[0]


def para_has_answer(answer_text, table_blocks, tokenizer):
    # assert isinstance(answer, list)
    # answer = [answer]
    answer_text = normalize(answer_text)
    tokens = tokenizer.tokenize(answer_text)
    answer_text = tokens.words(uncased=True)
    assert len(answer_text) == len(tokens)
    for single_prediction in table_blocks:
        single_prediction = ' '.join(single_prediction['passages']) + \
                            ' '.join(single_prediction['table'][0]) + \
                            ' '.join(single_prediction['table'][1][0])

        single_prediction = normalize(single_prediction)
        single_prediction = tokenizer.tokenize(single_prediction)
        single_prediction = single_prediction.words(uncased=True)
        for i in range(0, len(single_prediction) - len(answer_text) + 1):
            if answer_text == single_prediction[i: i + len(answer_text)]:
                return True
    return False


# def para_has_answer(answer, para, tokenizer):
#     # assert isinstance(answer, list)
#     answer = [answer]
#     text = normalize(para)
#     tokens = tokenizer.tokenize(text)
#     text = tokens.words(uncased=True)
#     assert len(text) == len(tokens)
#     for single_answer in answer:
#         single_answer = ' '.join(single_answer['passages']) + single_answer['table']
#         single_answer = normalize(single_answer)
#         single_answer = tokenizer.tokenize(single_answer)
#         single_answer = single_answer.words(uncased=True)
#         for i in range(0, len(text) - len(single_answer) + 1):
#             if single_answer == text[i: i + len(single_answer)]:
#                 return True
#     return False
class CellHelper(object):
    """Cell Helper to detect the cell type."""

    @staticmethod
    def is_unit(string):
        """Is the input a unit."""
        return re.search(r'\b(kg|m|cm|lb|hz|million)\b', string.lower())

    @staticmethod
    def is_score(string):
        """Is the input a score between two things."""
        if re.search(r'[0-9]+ - [0-9]+', string):
            return True
        elif re.search(r'[0-9]+-[0-9]+', string):
            return True
        else:
            return False

    @staticmethod
    def is_date(string, fuzzy=False):
        """Is the input a date."""
        try:
            parse(string, fuzzy=fuzzy)
            return True
        except Exception:  # pylint: disable=broad-except
            return False

    @staticmethod
    def is_bool(string):
        if string.lower() in ['yes', 'no']:
            return True
        else:
            return False

    @staticmethod
    def is_float(string):
        if '.' in string:
            try:
                float(string)
                return True
            except Exception:
                return False
        else:
            return False

    @staticmethod
    def is_normal_word(string):
        if ' ' not in string:
            return string.islower()
        else:
            return False


def whitelist(string):
    """Is the input a whitelist string."""
    string = string.strip()
    if len(string) < 2:
        return False
    elif string.isdigit():
        if len(string) == 4:
            return True
        else:
            return False
    elif string.replace(',', '').isdigit():
        return False
    elif CellHelper.is_float(string):
        return False
    elif '#' in string or '%' in string or '+' in string or '$' in string:
        return False
    elif CellHelper.is_normal_word(string):
        return False
    elif CellHelper.is_bool(string):
        return False
    elif CellHelper.is_score(string):
        return False
    elif CellHelper.is_unit(string):
        return False
    elif CellHelper.is_date(string):
        return False
    return True


def is_year(string):
    if len(string) == 4 and string.isdigit():
        return True
    else:
        return False


def found_table(answer, tbs):
    answer = answer.split(',')
    assert isinstance(answer, list)
    for g_tid in answer:
        for tb in tbs:
            if g_tid in tb['table_id']:
                return True
    return False


def complex_ans_recall():
    """
    calculate retrieval metrics for complexwebQ
    """
    import json
    import numpy as np
    from basic_tokenizer import SimpleTokenizer
    tok = SimpleTokenizer()

    predictions = json.load(
        open("/private/home/xwhan/code/learning_to_retrieve_reasoning_paths/results/complexwebq_retrieval_res.json"))
    raw_dev = [json.loads(l) for l in open("/private/home/xwhan/data/ComplexWebQ/complexwebq_dev_qas.txt").readlines()]
    id2qas = {_["id"]: _ for _ in raw_dev}

    assert len(predictions) == len(raw_dev)
    answer_recalls = []
    for item in predictions:
        qid = item["q_id"]
        title2passage = item["context"]
        gold_answers = id2qas[qid]["answer"]

        chain_coverage = []
        for chain in item["topk_titles"]:
            chain_text = " ".join([title2passage[_] for _ in chain])
            chain_coverage.append(para_has_answer(gold_answers, chain_text, tok))
        answer_recalls.append(np.sum(chain_coverage) > 0)
    print(len(answer_recalls))
    print(np.mean(answer_recalls))


def compute_kernel_bias(vecs):
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))
    return W, -mu


def transform_and_normalize(vecs, kernel=None, bias=None):
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    # return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5
    return vecs


def whitening(embedding1, embedding2):
    e1_type = embedding1.dtype
    e2_type = embedding2.dtype
    kernel, bias = compute_kernel_bias(np.concatenate([embedding1, embedding2], axis=0))
    embedding1 = transform_and_normalize(embedding1, kernel, bias)
    embedding2 = transform_and_normalize(embedding2, kernel, bias)
    embedding1 = embedding1.astype(e1_type)
    embedding2 = embedding2.astype(e2_type)
    return embedding1, embedding2


if __name__ == "__main__":
    complex_ans_recall()
