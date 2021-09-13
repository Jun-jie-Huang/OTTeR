from torch.utils.data import Dataset
import json
import pickle
import random
import torch

from .data_utils import collate_tokens


class TRDataset(Dataset):

    def __init__(self,
                 tokenizer,
                 data_path,
                 max_q_len,
                 max_q_sp_len,
                 max_c_len,
                 train=False,
                 ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_q_len = max_q_len
        self.max_c_len = max_c_len
        self.max_q_sp_len = max_q_sp_len
        self.train = train
        print(f"Loading data from {data_path}")
        self.data = [json.loads(line) for line in open(data_path).readlines()]
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        print(f"Total sample count {len(self.data)}")

    def encode_tb(self, passages, table, max_len):
        return self.tokenizer(table=table, queries=passages, max_length=max_len, return_tensors="pt")

    def __getitem__(self, index):
        sample = self.data[index]
        question = sample['question']
        if question.endswith("?"):
            question = question[:-1]

        table = sample['table']
        passages = ' '.join(sample['passages'])
        tb_codes = self.encode_tb(passages, table, self.max_c_len)

        q_codes = self.tokenizer.encode_plus(question, max_length=self.max_q_len, return_tensors="pt")
        label = torch.tensor(sample['label'])
        return {
                "q_codes": q_codes,
                "tb_codes": tb_codes,
                "label": label,
                }

    def __len__(self):
        return len(self.data)


def tb_collate(samples, pad_id=0):
    if len(samples) == 0:
        return {}

    batch = {
            'q_input_ids': collate_tokens([s["q_codes"]["input_ids"].view(-1) for s in samples], 0),
            'q_mask': collate_tokens([s["q_codes"]["attention_mask"].view(-1) for s in samples], 0),

            'tb_input_ids': collate_tokens([s["tb_codes"]["input_ids"].view(-1) for s in samples], 0),
            'tb_mask':collate_tokens([s["tb_codes"]["attention_mask"].view(-1) for s in samples], 0),

        }

    # if "token_type_ids" in samples[0]["q_codes"]:
    #     batch.update({
    #         'q_type_ids': collate_tokens([s["q_codes"]["token_type_ids"].view(-1) for s in samples], 0),
    #         'c1_type_ids': collate_tokens([s["start_para_codes"]["token_type_ids"] for s in samples], 0),
    #         'c2_type_ids': collate_tokens([s["bridge_para_codes"]["token_type_ids"] for s in samples], 0),
    #         "q_sp_type_ids": collate_tokens([s["q_sp_codes"]["token_type_ids"].view(-1) for s in samples], 0),
    #         'neg1_type_ids': collate_tokens([s["neg_codes_1"]["token_type_ids"] for s in samples], 0),
    #         'neg2_type_ids': collate_tokens([s["neg_codes_2"]["token_type_ids"] for s in samples], 0),
    #     })
    #
    # if "sent_ids" in samples[0]["start_para_codes"]:
    #     batch["c1_sent_target"] = collate_tokens([s["start_para_codes"]["sent_ids"] for s in samples], -1)
    #     batch["c1_sent_offsets"] = collate_tokens([s["start_para_codes"]["sent_offsets"] for s in samples], 0),

    return batch
