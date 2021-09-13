# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.

"""
single hop retrieval models
"""
from transformers import AutoModel
import torch.nn as nn
import torch
from transformers import TapasConfig, TapasModel, TapasForSequenceClassification

import logging
logger = logging.getLogger(__name__)
from transformers import (AutoConfig, AutoModel, AutoTokenizer,
                          BertConfig, BertModel, BertTokenizer,
                          XLNetConfig, XLNetModel, XLNetTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          ElectraConfig, ElectraModel, ElectraTokenizer,
                          AlbertConfig, AlbertModel, AlbertTokenizer,
                          LongformerConfig, LongformerModel, LongformerTokenizer)


MODEL_CLASSES = {
    "bert": (BertConfig, BertModel, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetModel, XLNetTokenizer),
    "roberta": (RobertaConfig, RobertaModel, RobertaTokenizer),
    "electra": (ElectraConfig, ElectraModel, ElectraTokenizer),
    "albert": (AlbertConfig, AlbertModel, AlbertTokenizer),
    'longformer': (LongformerConfig, LongformerModel, LongformerTokenizer)
}


class SingleRetriever(nn.Module):
    def __init__(self, config, args):
        super(SingleRetriever, self).__init__()
        self.shared_encoder = args.shared_encoder
        self.no_proj = args.no_proj
        self.encoder = AutoModel.from_pretrained(args.model_name)
        self.hidden_size = config.hidden_size
        if not self.shared_encoder:
            self.encoder_q = AutoModel.from_pretrained(args.model_name)
        self.project = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                     nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps))

    def forward(self, batch):
        c_cls = self.encode_seq(batch['c_input_ids'], batch['c_mask'], batch['c_type_ids'])
        neg_c_cls = self.encode_seq(batch['neg_input_ids'], batch['neg_mask'], batch['neg_type_ids'])
        q_cls = self.encode_q(batch['q_input_ids'], batch['q_mask'], batch['q_type_ids'])
        return {'q': q_cls, 'c': c_cls, 'neg_c': neg_c_cls}

    def encode_seq(self, input_ids=None,
                   attention_mask=None,
                   token_type_ids=None,
                   position_ids=None,
                   head_mask=None,
                   output_hidden_states=True, ):
        cls_rep = self.encoder(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask,
                               output_hidden_states=output_hidden_states)[0][:, 0, :]
        # pooled_output = self.dropout(cls_rep[1])
        if self.no_proj:
            return cls_rep
        vector = self.project(cls_rep)
        return vector

    def encode_q(self, input_ids=None,
                 attention_mask=None,
                 token_type_ids=None,
                 position_ids=None,
                 head_mask=None,
                 output_hidden_states=True, ):
        if self.shared_encoder:
            cls_rep = self.encoder(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   position_ids=position_ids,
                                   head_mask=head_mask,
                                   output_hidden_states=output_hidden_states)[0][:, 0, :]
        else:
            cls_rep = self.encoder_q(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids,
                                     position_ids=position_ids,
                                     head_mask=head_mask,
                                     output_hidden_states=output_hidden_states)[0][:, 0, :]
        if self.no_proj:
            return cls_rep
        vector = self.project(cls_rep)
        return vector

    def evaluate_encode_tb(self, batch):
        c_cls = self.encode_seq(batch['input_ids'], batch['input_mask'], batch['input_type_ids'])
        # (Batch, dim)
        # logger.info("vector.shape:{}".format(cls_rep[0].shape))
        return {'embed': c_cls}

    def evaluate_encode_que(self, batch):
        q_cls = self.encode_q(batch['input_ids'], batch['input_mask'], batch['input_type_ids'])
        # (Batch, dim)
        # logger.info("vector.shape:{}".format(cls_rep[0].shape))
        return {'embed': q_cls}


class SingleEncoder(nn.Module):
    def __init__(self, config, args):
        super(SingleEncoder, self).__init__()
        self.config = config
        self.args = args
        self.shared_encoder = args.shared_encoder
        self.no_proj = args.no_proj

        self.encoder = AutoModel.from_pretrained(args.model_name)
        if not self.shared_encoder:
            self.encoder_q = AutoModel.from_pretrained(args.model_name)
        self.project = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                     nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps))

    def forward(self, batch):
        if self.encode_table:
            cls = self.encode_seq(batch['input_ids'], batch['input_mask'], batch['input_type_ids'])
        else:
            cls = self.encode_q(batch['input_ids'], batch['input_mask'], batch['input_type_ids'])
        return {'embed': cls}

    def encode_seq(self, input_ids=None,
                   attention_mask=None,
                   token_type_ids=None,
                   position_ids=None,
                   head_mask=None,
                   output_hidden_states=True, ):
        cls_rep = self.encoder(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask,
                               output_hidden_states=output_hidden_states)[0][:, 0, :]
        # pooled_output = self.dropout(cls_rep[1])
        if self.no_proj:
            return cls_rep
        vector = self.project(cls_rep)
        return vector

    def encode_q(self, input_ids=None,
                 attention_mask=None,
                 token_type_ids=None,
                 position_ids=None,
                 head_mask=None,
                 output_hidden_states=True, ):
        if self.shared_encoder:
            cls_rep = self.encoder(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   position_ids=position_ids,
                                   head_mask=head_mask,
                                   output_hidden_states=output_hidden_states)[0][:, 0, :]
        else:
            cls_rep = self.encoder_q(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids,
                                     position_ids=position_ids,
                                     head_mask=head_mask,
                                     output_hidden_states=output_hidden_states)[0][:, 0, :]
        if self.no_proj:
            return cls_rep
        vector = self.project(cls_rep)
        return vector


class RobertaSingleRetriever(SingleRetriever):

    def __init__(self, config, args):
        super(RobertaSingleRetriever, self).__init__(config, args)

    def forward(self, batch):
        c_cls = self.encode_seq(batch['c_input_ids'], batch['c_mask'])
        neg_c_cls = self.encode_seq(batch['neg_input_ids'], batch['neg_mask'])
        q_cls = self.encode_q(batch['q_input_ids'], batch['q_mask'])
        return {'q': q_cls, 'c': c_cls, 'neg_c': neg_c_cls}

    def evaluate_encode_tb(self, batch):
        c_cls = self.encode_seq(batch['input_ids'], batch['input_mask'])
        return {'embed': c_cls}

    def evaluate_encode_que(self, batch):
        q_cls = self.encode_q(batch['input_ids'], batch['input_mask'])
        return {'embed': q_cls}


class RobertaSingleEncoder(SingleEncoder):

    def __init__(self, config, args):
        super(RobertaSingleEncoder, self).__init__(config, args)
        self.encode_table = args.encode_table

    def forward(self, batch):
        if self.encode_table:
            cls = self.encode_seq(batch['input_ids'], batch['input_mask'])
        else:
            cls = self.encode_q(batch['input_ids'], batch['input_mask'])
        return {'embed': cls}


class MomentumRetriever(nn.Module):

    def __init__(self, config, args):
        super().__init__()
        self.config = config
        self.args = args
        # shared encoder for everything
        self.encoder = AutoModel.from_pretrained(args.model_name)
        self.max_c_len = args.max_c_len

        # queue of context token ids
        self.k = args.k  # queue size
        self.register_buffer("queue", torch.zeros(self.k, args.max_c_len * 3, dtype=torch.long))  #
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def forward(self, batch):
        q_cls = self.encoder(batch['q_input_ids'], batch['q_mask'], batch.get('q_type_ids', None))[0][:, 0, :]
        c_cls = self.encoder(batch['c_input_ids'], batch['c_mask'], batch.get('c_type_ids', None))[0][:, 0, :]
        neg = self.encoder(batch['neg_input_ids'], batch['neg_mask'], batch.get('neg_type_ids', None))[0][:, 0, :]

        return {'q': q_cls, 'c': c_cls, 'neg_c': neg}

    def encode_q(self, input_ids, q_mask, q_type_ids):
        return self.encoder(input_ids, q_mask, q_type_ids)[0][:, 0, :]

    @torch.no_grad()
    def encode_queue_ctx(self):
        queue = self.queue.clone().detach()
        input_ids = queue[:, :self.max_c_len]
        input_masks = queue[:, self.max_c_len:2 * self.max_c_len]
        type_ids = queue[:, self.max_c_len * 2:]

        queue_c_clss = []
        self.encoder.eval()
        with torch.no_grad():
            for batch_start in range(0, self.k, 100):
                queue_c_cls = \
                self.encoder(input_ids[batch_start:batch_start + 100], input_masks[batch_start:batch_start + 100],
                             type_ids[batch_start:batch_start + 100])[0][:, 0, :]
                queue_c_clss.append(queue_c_cls)
        self.encoder.train()

        return torch.cat(queue_c_clss, dim=0)

    @torch.no_grad()
    def dequeue_and_enqueue(self, batch):
        """
        memory bank of previous contexts
        """

        # gather keys before updating queue
        batch_size = batch["c_input_ids"].shape[0]
        ptr = int(self.queue_ptr)
        if ptr + batch_size > self.k:
            batch_size = self.k - ptr
            batch["c_input_ids"] = batch["c_input_ids"][:batch_size]
            batch["c_mask"] = batch["c_mask"][:batch_size]
            batch["c_type_ids"] = batch["c_type_ids"][:batch_size]
        batch_seq_len = batch["c_input_ids"].size(1)

        # if self.k % batch_size != 0:
        #     return
        # assert self.k % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size, :batch_seq_len] = batch["c_input_ids"]
        self.queue[ptr:ptr + batch_size, self.max_c_len:self.max_c_len + batch_seq_len] = batch["c_mask"]
        self.queue[ptr:ptr + batch_size, self.max_c_len * 2:self.max_c_len * 2 + batch_seq_len] = batch["c_type_ids"]

        ptr = (ptr + batch_size) % self.k  # move pointer
        self.queue_ptr[0] = ptr
        return


class RobertaMomentumRetriever(nn.Module):

    def __init__(self, config, args):
        super().__init__()
        self.config = config
        self.args = args
        # shared encoder for everything
        self.encoder = AutoModel.from_pretrained(args.model_name)
        self.max_c_len = args.max_c_len

        # queue of context token ids
        self.k = args.k  # queue size
        self.register_buffer("queue", torch.zeros(self.k, args.max_c_len * 2, dtype=torch.long))  #
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def forward(self, batch):
        q_cls = self.encoder(batch['q_input_ids'], batch['q_mask'])[0][:, 0, :]
        c_cls = self.encoder(batch['c_input_ids'], batch['c_mask'])[0][:, 0, :]
        neg = self.encoder(batch['neg_input_ids'], batch['neg_mask'])[0][:, 0, :]
        return {'q': q_cls, 'c': c_cls, 'neg_c': neg}

    def encode_q(self, input_ids, q_mask):
        return self.encoder(input_ids, q_mask)[0][:, 0, :]

    @torch.no_grad()
    def encode_queue_ctx(self):
        queue = self.queue.clone().detach()
        input_ids = queue[:, :self.max_c_len]
        input_masks = queue[:, self.max_c_len:2 * self.max_c_len]
        # type_ids = queue[:, self.max_c_len * 2:]

        queue_c_clss = []
        self.encoder.eval()
        with torch.no_grad():
            for batch_start in range(0, self.k, 100):
                queue_c_cls = \
                self.encoder(input_ids[batch_start:batch_start + 100], input_masks[batch_start:batch_start + 100])[0][:, 0, :]
                queue_c_clss.append(queue_c_cls)
        self.encoder.train()

        return torch.cat(queue_c_clss, dim=0)

    @torch.no_grad()
    def dequeue_and_enqueue(self, batch):
        """
        memory bank of previous contexts
        """

        # gather keys before updating queue
        batch_size = batch["c_input_ids"].shape[0]
        ptr = int(self.queue_ptr)
        if ptr + batch_size > self.k:
            batch_size = self.k - ptr
            batch["c_input_ids"] = batch["c_input_ids"][:batch_size]
            batch["c_mask"] = batch["c_mask"][:batch_size]
        batch_seq_len = batch["c_input_ids"].size(1)

        # if self.k % batch_size != 0:
        #     return
        # assert self.k % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size, :batch_seq_len] = batch["c_input_ids"]
        self.queue[ptr:ptr + batch_size, self.max_c_len:self.max_c_len + batch_seq_len] = batch["c_mask"]

        ptr = (ptr + batch_size) % self.k  # move pointer
        self.queue_ptr[0] = ptr
        return


class MomentumEncoder(nn.Module):

    def __init__(self, config, args):
        super().__init__()
        self.config = config
        self.args = args
        # shared encoder for everything
        self.encoder = AutoModel.from_pretrained(args.model_name)
        self.max_c_len = args.max_c_len
        self.encode_table = args.encode_table

    def forward(self, batch):
        cls = self.encoder(batch['input_ids'], batch['input_mask'], batch.get('input_type_ids', None))[0][:, 0, :]
        return {'embed': cls}


"""
The following are models used to encode the corpus
"""


class CtxEncoder(nn.Module):

    def __init__(self,
                 config,
                 args
                 ):
        super().__init__()
        self.config = config
        self.args = args
        self.encoder_c = AutoModel.from_pretrained(args.model_name)
        self.multi_vector = args.multi_vector
        self.scheme = args.scheme
        if self.scheme == "layerwise":
            self.encoder_c.encoder.output_hidden_states = True

    def forward(self, batch):
        input_ids, attention_mask, type_ids = batch["input_ids"], batch["input_mask"], batch.get("input_type_ids", None)

        if self.multi_vector > 1:
            if self.scheme == "layerwise":
                c_hiddens = self.encoder(batch['input_ids'], batch['input_mask'], batch.get('input_type_ids', None))[2][
                            ::-1]
                c_cls = torch.cat([hidden[:, 0, :].unsqueeze(1) for hidden in c_hiddens[:self.multi_vector]], dim=1)
            elif self.scheme == "tokenwise":
                c_cls = self.encoder(batch['input_ids'], batch['input_mask'], batch.get('input_type_ids', None))[0][:,
                        :self.multi_vector, :]
            else:
                assert False
            c_cls = c_cls.view(-1, c_cls.size(-1))
        else:
            c_cls = self.encoder_c(input_ids, attention_mask, type_ids)[0][:, 0, :]
        return {'embed': c_cls}


class RobertaCtxEncoder(nn.Module):

    def __init__(self,
                 config,
                 args
                 ):
        super().__init__()
        self.config = config
        self.args = args
        self.encoder = AutoModel.from_pretrained(args.model_name)
        self.project = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                     nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps))

    def forward(self, batch):
        input_ids, attention_mask = batch["input_ids"], batch["input_mask"]
        cls_rep = self.encoder(input_ids, attention_mask)[0][:, 0, :]
        vector = self.project(cls_rep)
        return {'embed': vector}
