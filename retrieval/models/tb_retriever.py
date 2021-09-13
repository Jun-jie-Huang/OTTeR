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

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from attention.SelfAttention import ScaledDotProductAttention

from transformers import (AutoConfig, AutoModel, AutoTokenizer,
                          BertConfig, BertModel, BertTokenizer,
                          XLNetConfig, XLNetModel, XLNetTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          ElectraConfig, ElectraModel, ElectraTokenizer,
                          AlbertConfig, AlbertModel, AlbertTokenizer,
                          LongformerConfig, LongformerModel, LongformerTokenizer)

from transformers import BertLayer

MODEL_CLASSES = {
    "bert": (BertConfig, BertModel, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetModel, XLNetTokenizer),
    "roberta": (RobertaConfig, RobertaModel, RobertaTokenizer),
    "electra": (ElectraConfig, ElectraModel, ElectraTokenizer),
    "albert": (AlbertConfig, AlbertModel, AlbertTokenizer),
    'longformer': (LongformerConfig, LongformerModel, LongformerTokenizer)
}


def pooling_masked_part(hidden, mask, method='mean'):
    if method == 'mean':
        pooled = torch.sum(hidden * mask.unsqueeze(2), dim=1) / torch.sum(mask, dim=1, keepdim=True)
    elif method == 'sum':
        pooled = torch.sum(hidden * mask.unsqueeze(2), dim=1)
    elif method == 'max':
        pooled = torch.max(hidden + ((1-mask)*-1e5).unsqueeze(2), dim=1)[0]
    elif method == 'cls':
        pooled = hidden[:, 0, :]
    elif method == 'first':
        dim = hidden.shape[-1]
        tmp = mask * torch.arange(mask.shape[1], 0, -1).to(mask.device)
        index = torch.argmax(tmp, 1, keepdim=True)
        pooled = torch.gather(hidden, dim=1, index=index.unsqueeze(2).repeat([1, 1, dim])).squeeze()
    else:
        pooled = hidden[:, 0, :]
    return pooled


class SingleRetrieverThreeCatPool(nn.Module):
    def __init__(self, config, args):
        super(SingleRetrieverThreeCatPool, self).__init__()
        self.shared_encoder = args.shared_encoder
        self.no_proj = args.no_proj
        self.encoder = AutoModel.from_pretrained(args.model_name)
        self.hidden_size = config.hidden_size
        self.part_pooling = args.part_pooling
        if not self.shared_encoder:
            self.encoder_q = AutoModel.from_pretrained(args.model_name)
        self.project = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                     nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps))

    def forward(self, batch):
        c_cls = self.encode_seq(batch['c_input_ids'], batch['c_mask'], batch['c_type_ids'], part2_mask=batch['c_part2_mask'], part3_mask=batch['c_part3_mask'])
        neg_c_cls = self.encode_seq(batch['neg_input_ids'], batch['neg_mask'], batch['neg_type_ids'], part2_mask=batch['neg_part2_mask'], part3_mask=batch['neg_part3_mask'])
        q_cls = self.encode_q(batch['q_input_ids'], batch['q_mask'], batch['q_type_ids'], part2_mask=batch['q_part2_mask'], part3_mask=batch['q_part3_mask'])
        return {'q': q_cls, 'c': c_cls, 'neg_c': neg_c_cls}

    def encode_seq(self, input_ids=None,
                   attention_mask=None,
                   token_type_ids=None,
                   position_ids=None,
                   head_mask=None,
                   output_hidden_states=True,
                   part2_mask=None,
                   part3_mask=None):
        hidden_states = self.encoder(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids,
                                     position_ids=position_ids,
                                     head_mask=head_mask,
                                     output_hidden_states=output_hidden_states)[0]
        cls_rep = hidden_states[:, 0, :]
        # pooled_output = self.dropout(cls_rep[1])
        if self.no_proj:
            part1 = cls_rep
        else:
            part1 = self.project(cls_rep)
        part2 = pooling_masked_part(hidden_states, part2_mask, method=self.part_pooling)
        part3 = pooling_masked_part(hidden_states, part3_mask, method=self.part_pooling)
        vector = torch.cat([part1, part2, part3], dim=1)
        return vector

    def encode_q(self, input_ids=None,
                 attention_mask=None,
                 token_type_ids=None,
                 position_ids=None,
                 head_mask=None,
                 output_hidden_states=True,
                 part2_mask=None,
                 part3_mask=None):
        if self.shared_encoder:
            hidden_states = self.encoder(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   position_ids=position_ids,
                                   head_mask=head_mask,
                                   output_hidden_states=output_hidden_states)[0]
        else:
            hidden_states = self.encoder_q(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids,
                                     position_ids=position_ids,
                                     head_mask=head_mask,
                                     output_hidden_states=output_hidden_states)[0]
        cls_rep = hidden_states[:, 0, :]
        if self.no_proj:
            part1 = cls_rep
        else:
            part1 = self.project(cls_rep)
        part2 = pooling_masked_part(hidden_states, part2_mask, method=self.part_pooling)
        part3 = pooling_masked_part(hidden_states, part3_mask, method=self.part_pooling)
        vector = torch.cat([part1, part2, part3], dim=1)
        return vector

    def evaluate_encode_tb(self, batch):
        c_cls = self.encode_seq(batch['input_ids'], batch['input_mask'], batch['input_type_ids'], part2_mask=batch['part2_mask'], part3_mask=batch['part3_mask'])
        # (Batch, dim)
        # logger.info("vector.shape:{}".format(cls_rep[0].shape))
        return {'embed': c_cls}

    def evaluate_encode_que(self, batch):
        q_cls = self.encode_q(batch['input_ids'], batch['input_mask'], batch['input_type_ids'], part2_mask=batch['part2_mask'], part3_mask=batch['part3_mask'])
        # (Batch, dim)
        # logger.info("vector.shape:{}".format(cls_rep[0].shape))
        return {'embed': q_cls}


class SingleEncoderThreeCatPool(nn.Module):
    def __init__(self, config, args):
        super(SingleEncoderThreeCatPool, self).__init__()
        self.config = config
        self.args = args
        self.shared_encoder = args.shared_encoder
        self.no_proj = args.no_proj
        self.part_pooling = args.part_pooling

        self.encoder = AutoModel.from_pretrained(args.model_name)
        if not self.shared_encoder:
            self.encoder_q = AutoModel.from_pretrained(args.model_name)
        self.project = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                     nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps))

    def forward(self, batch):
        if self.encode_table:
            cls = self.encode_seq(batch['input_ids'], batch['input_mask'], batch['input_type_ids'], part2_mask=batch['part2_mask'], part3_mask=batch['part3_mask'])
        else:
            cls = self.encode_q(batch['input_ids'], batch['input_mask'], batch['input_type_ids'], part2_mask=batch['part2_mask'], part3_mask=batch['part3_mask'])
        return {'embed': cls}

    def encode_seq(self, input_ids=None,
                   attention_mask=None,
                   token_type_ids=None,
                   position_ids=None,
                   head_mask=None,
                   output_hidden_states=True,
                   part2_mask=None,
                   part3_mask=None):
        hidden_states = self.encoder(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids,
                                     position_ids=position_ids,
                                     head_mask=head_mask,
                                     output_hidden_states=output_hidden_states)[0]
        cls_rep = hidden_states[:, 0, :]
        # pooled_output = self.dropout(cls_rep[1])
        if self.no_proj:
            part1 = cls_rep
        else:
            part1 = self.project(cls_rep)
        part2 = pooling_masked_part(hidden_states, part2_mask, method=self.part_pooling)
        part3 = pooling_masked_part(hidden_states, part3_mask, method=self.part_pooling)
        vector = torch.cat([part1, part2, part3], dim=1)
        return vector

    def encode_q(self, input_ids=None,
                 attention_mask=None,
                 token_type_ids=None,
                 position_ids=None,
                 head_mask=None,
                 output_hidden_states=True,
                 part2_mask=None,
                 part3_mask=None):
        if self.shared_encoder:
            hidden_states = self.encoder(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   position_ids=position_ids,
                                   head_mask=head_mask,
                                   output_hidden_states=output_hidden_states)[0]
        else:
            hidden_states = self.encoder_q(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids,
                                     position_ids=position_ids,
                                     head_mask=head_mask,
                                     output_hidden_states=output_hidden_states)[0]
        cls_rep = hidden_states[:, 0, :]
        if self.no_proj:
            part1 = cls_rep
        else:
            part1 = self.project(cls_rep)
        part2 = pooling_masked_part(hidden_states, part2_mask, method=self.part_pooling)
        part3 = pooling_masked_part(hidden_states, part3_mask, method=self.part_pooling)
        vector = torch.cat([part1, part2, part3], dim=1)
        return vector


class RobertaSingleRetrieverThreeCatPool(SingleRetrieverThreeCatPool):

    def __init__(self, config, args):
        super(RobertaSingleRetrieverThreeCatPool, self).__init__(config, args)

    def forward(self, batch):
        c_cls = self.encode_seq(batch['c_input_ids'], batch['c_mask'], part2_mask=batch['c_part2_mask'], part3_mask=batch['c_part3_mask'])
        neg_c_cls = self.encode_seq(batch['neg_input_ids'], batch['neg_mask'], part2_mask=batch['neg_part2_mask'], part3_mask=batch['neg_part3_mask'])
        q_cls = self.encode_q(batch['q_input_ids'], batch['q_mask'], part2_mask=batch['q_part2_mask'], part3_mask=batch['q_part3_mask'])
        return {'q': q_cls, 'c': c_cls, 'neg_c': neg_c_cls}

    def evaluate_encode_tb(self, batch):
        c_cls = self.encode_seq(batch['input_ids'], batch['input_mask'], part2_mask=batch['part2_mask'], part3_mask=batch['part3_mask'])
        return {'embed': c_cls}

    def evaluate_encode_que(self, batch):
        q_cls = self.encode_q(batch['input_ids'], batch['input_mask'], part2_mask=batch['part2_mask'], part3_mask=batch['part3_mask'])
        return {'embed': q_cls}


class RobertaSingleEncoderThreeCatPool(SingleEncoderThreeCatPool):

    def __init__(self, config, args):
        super(RobertaSingleEncoderThreeCatPool, self).__init__(config, args)
        self.encode_table = args.encode_table

    def forward(self, batch):
        if self.encode_table:
            cls = self.encode_seq(batch['input_ids'], batch['input_mask'], part2_mask=batch['part2_mask'], part3_mask=batch['part3_mask'])
        else:
            cls = self.encode_q(batch['input_ids'], batch['input_mask'], part2_mask=batch['part2_mask'], part3_mask=batch['part3_mask'])
        return {'embed': cls}


class SingleRetrieverThreeCatAtt(nn.Module):
    def __init__(self, config, args):
        super(SingleRetrieverThreeCatAtt, self).__init__()
        self.shared_encoder = args.shared_encoder
        self.no_proj = args.no_proj
        self.encoder = AutoModel.from_pretrained(args.model_name)
        self.hidden_size = config.hidden_size
        self.part_pooling = args.part_pooling
        if not self.shared_encoder:
            self.encoder_q = AutoModel.from_pretrained(args.model_name)
        self.project = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                     nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps))

        if self.part_pooling == 'att_self_pool':
            self.self_proj_tab = nn.Linear(config.hidden_size, 1)
            self.self_proj_psg = nn.Linear(config.hidden_size, 1)
            # self.att_pooling = self.att_self_pool
        elif self.part_pooling == 'att_self_multihead':
            self.self_proj_tab = ScaledDotProductAttention(d_model=config.hidden_size,
                                                           d_k=config.hidden_size//8, d_v=config.hidden_size//8, h=8)
            self.self_proj_psg = ScaledDotProductAttention(d_model=config.hidden_size,
                                                           d_k=config.hidden_size//8, d_v=config.hidden_size//8, h=8)
            # self.att_pooling = self.att_self_multihead
        # if self.part_pooling == 'att_cross_multihead':
        #     self.self_proj_tab = ScaledDotProductAttention(d_model=config.hidden_size,
        #                                                    d_k=config.hidden_size/8, d_v=config.hidden_size/8, h=8)
        #     self.self_proj_psg = ScaledDotProductAttention(d_model=config.hidden_size,
        #                                                    d_k=config.hidden_size/8, d_v=config.hidden_size/8, h=8)
        #     self.att_pooling = self.att_cross_multihead
        else:
            raise NotImplementedError("part pooling only support att_self_pool, att_self_multihead")

    def forward(self, batch):
        c_cls = self.encode_seq(batch['c_input_ids'], batch['c_mask'], batch['c_type_ids'], part2_mask=batch['c_part2_mask'], part3_mask=batch['c_part3_mask'])
        neg_c_cls = self.encode_seq(batch['neg_input_ids'], batch['neg_mask'], batch['neg_type_ids'], part2_mask=batch['neg_part2_mask'], part3_mask=batch['neg_part3_mask'])
        q_cls = self.encode_q(batch['q_input_ids'], batch['q_mask'], batch['q_type_ids'], part2_mask=batch['q_part2_mask'], part3_mask=batch['q_part3_mask'])
        return {'q': q_cls, 'c': c_cls, 'neg_c': neg_c_cls}

    def encode_seq(self, input_ids=None,
                   attention_mask=None,
                   token_type_ids=None,
                   position_ids=None,
                   head_mask=None,
                   output_hidden_states=True,
                   part2_mask=None,
                   part3_mask=None):
        hidden_states = self.encoder(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids,
                                     position_ids=position_ids,
                                     head_mask=head_mask,
                                     output_hidden_states=output_hidden_states)[0]
        cls_rep = hidden_states[:, 0, :]
        # pooled_output = self.dropout(cls_rep[1])
        if self.no_proj:
            part1 = cls_rep
        else:
            part1 = self.project(cls_rep)
        if self.part_pooling == 'att_self_pool':
            part2, part3 = self.att_self_pool(hidden_states, part2_mask, part3_mask)
        elif self.part_pooling == 'att_self_multihead':
            part2, part3 = self.att_self_multihead(hidden_states, part2_mask, part3_mask)
        else:
            raise NotImplementedError("part pooling only support att_self_pool, att_self_multihead")
        vector = torch.cat([part1, part2, part3], dim=1)
        return vector

    def encode_q(self, input_ids=None,
                 attention_mask=None,
                 token_type_ids=None,
                 position_ids=None,
                 head_mask=None,
                 output_hidden_states=True,
                 part2_mask=None,
                 part3_mask=None):
        if self.shared_encoder:
            hidden_states = self.encoder(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   position_ids=position_ids,
                                   head_mask=head_mask,
                                   output_hidden_states=output_hidden_states)[0]
        else:
            hidden_states = self.encoder_q(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids,
                                     position_ids=position_ids,
                                     head_mask=head_mask,
                                     output_hidden_states=output_hidden_states)[0]
        cls_rep = hidden_states[:, 0, :]
        if self.no_proj:
            part1 = cls_rep
        else:
            part1 = self.project(cls_rep)
        if self.part_pooling == 'att_self_pool':
            part2, part3 = self.att_self_pool(hidden_states, part2_mask, part3_mask)
        elif self.part_pooling == 'att_self_multihead':
            part2, part3 = self.att_self_multihead(hidden_states, part2_mask, part3_mask)
        else:
            raise NotImplementedError("part pooling only support att_self_pool, att_self_multihead")
        vector = torch.cat([part1, part2, part3], dim=1)
        return vector

    def att_self_pool(self, inputs, part2_mask, part3_mask):
        part2_mask, part3_mask = part2_mask.unsqueeze(2), part3_mask.unsqueeze(2)
        # import ipdb; ipdb.set_trace()
        logits_tab = self.self_proj_tab(inputs)
        # att_tab = torch.softmax(logits_tab.masked_fill(1-part2_mask, -1e6), dim=1)
        att_tab = torch.softmax(logits_tab + (1-part2_mask) * -1e6, dim=1)
        pooled2 = torch.sum(att_tab*inputs, dim=1).squeeze()

        logits_psg = self.self_proj_psg(inputs)
        # att_psg = torch.softmax(logits_psg.masked_fill(1-part3_mask, -1e6), dim=1)
        att_psg = torch.softmax(logits_psg + (1-part3_mask) * -1e6, dim=1)
        pooled3 = torch.sum(att_psg*inputs, dim=1).squeeze()

        return pooled2, pooled3

    def att_self_multihead(self, inputs, part2_mask, part3_mask):
        extended_part2_mask = part2_mask[:, None, None, :].to(part2_mask.device)
        extended_part3_mask = part3_mask[:, None, None, :].to(part3_mask.device)
        logits_tab = self.self_proj_tab(inputs, inputs, inputs, attention_mask=extended_part2_mask)
        logits_psg = self.self_proj_psg(inputs, inputs, inputs, attention_mask=extended_part3_mask)
        pooled2 = pooling_masked_part(logits_tab, part2_mask, method='sum')
        pooled3 = pooling_masked_part(logits_psg, part3_mask, method='sum')
        return pooled2, pooled3

    def evaluate_encode_tb(self, batch):
        c_cls = self.encode_seq(batch['input_ids'], batch['input_mask'], batch['input_type_ids'], part2_mask=batch['part2_mask'], part3_mask=batch['part3_mask'])
        # (Batch, dim)
        # logger.info("vector.shape:{}".format(cls_rep[0].shape))
        return {'embed': c_cls}

    def evaluate_encode_que(self, batch):
        q_cls = self.encode_q(batch['input_ids'], batch['input_mask'], batch['input_type_ids'], part2_mask=batch['part2_mask'], part3_mask=batch['part3_mask'])
        # (Batch, dim)
        # logger.info("vector.shape:{}".format(cls_rep[0].shape))
        return {'embed': q_cls}


class SingleEncoderThreeCatAtt(nn.Module):
    def __init__(self, config, args):
        super(SingleEncoderThreeCatAtt, self).__init__()
        self.config = config
        self.args = args
        self.shared_encoder = args.shared_encoder
        self.no_proj = args.no_proj
        self.part_pooling = args.part_pooling

        self.encoder = AutoModel.from_pretrained(args.model_name)
        if not self.shared_encoder:
            self.encoder_q = AutoModel.from_pretrained(args.model_name)
        self.project = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                     nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps))

        if self.part_pooling == 'att_self_pool':
            self.self_proj_tab = nn.Linear(config.hidden_size, 1)
            self.self_proj_psg = nn.Linear(config.hidden_size, 1)
            # self.att_pooling = self.att_self_pool
        elif self.part_pooling == 'att_self_multihead':
            self.self_proj_tab = ScaledDotProductAttention(d_model=config.hidden_size,
                                                           d_k=config.hidden_size//8, d_v=config.hidden_size//8, h=8)
            self.self_proj_psg = ScaledDotProductAttention(d_model=config.hidden_size,
                                                           d_k=config.hidden_size//8, d_v=config.hidden_size//8, h=8)
            # self.att_pooling = self.att_self_multihead
        # if self.part_pooling == 'att_cross_multihead':
        #     self.self_proj_tab = ScaledDotProductAttention(d_model=config.hidden_size,
        #                                                    d_k=config.hidden_size/8, d_v=config.hidden_size/8, h=8)
        #     self.self_proj_psg = ScaledDotProductAttention(d_model=config.hidden_size,
        #                                                    d_k=config.hidden_size/8, d_v=config.hidden_size/8, h=8)
        #     self.att_pooling = self.att_cross_multihead
        else:
            raise NotImplementedError("part pooling only support att_self_pool, att_cross_multihead")

    def forward(self, batch):
        if self.encode_table:
            cls = self.encode_seq(batch['input_ids'], batch['input_mask'], batch['input_type_ids'], part2_mask=batch['part2_mask'], part3_mask=batch['part3_mask'])
        else:
            cls = self.encode_q(batch['input_ids'], batch['input_mask'], batch['input_type_ids'], part2_mask=batch['part2_mask'], part3_mask=batch['part3_mask'])
        return {'embed': cls}

    def encode_seq(self, input_ids=None,
                   attention_mask=None,
                   token_type_ids=None,
                   position_ids=None,
                   head_mask=None,
                   output_hidden_states=True,
                   part2_mask=None,
                   part3_mask=None):
        hidden_states = self.encoder(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids,
                                     position_ids=position_ids,
                                     head_mask=head_mask,
                                     output_hidden_states=output_hidden_states)[0]
        cls_rep = hidden_states[:, 0, :]
        # pooled_output = self.dropout(cls_rep[1])
        if self.no_proj:
            part1 = cls_rep
        else:
            part1 = self.project(cls_rep)
        if self.part_pooling == 'att_self_pool':
            part2, part3 = self.att_self_pool(hidden_states, part2_mask, part3_mask)
        elif self.part_pooling == 'att_self_multihead':
            part2, part3 = self.att_self_multihead(hidden_states, part2_mask, part3_mask)
        else:
            raise NotImplementedError("part pooling only support att_self_pool, att_self_multihead")
        vector = torch.cat([part1, part2, part3], dim=1)
        return vector

    def encode_q(self, input_ids=None,
                 attention_mask=None,
                 token_type_ids=None,
                 position_ids=None,
                 head_mask=None,
                 output_hidden_states=True,
                 part2_mask=None,
                 part3_mask=None):
        if self.shared_encoder:
            hidden_states = self.encoder(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   position_ids=position_ids,
                                   head_mask=head_mask,
                                   output_hidden_states=output_hidden_states)[0]
        else:
            hidden_states = self.encoder_q(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids,
                                     position_ids=position_ids,
                                     head_mask=head_mask,
                                     output_hidden_states=output_hidden_states)[0]
        cls_rep = hidden_states[:, 0, :]
        if self.no_proj:
            part1 = cls_rep
        else:
            part1 = self.project(cls_rep)
        if self.part_pooling == 'att_self_pool':
            part2, part3 = self.att_self_pool(hidden_states, part2_mask, part3_mask)
        elif self.part_pooling == 'att_self_multihead':
            part2, part3 = self.att_self_multihead(hidden_states, part2_mask, part3_mask)
        else:
            raise NotImplementedError("part pooling only support att_self_pool, att_self_multihead")
        vector = torch.cat([part1, part2, part3], dim=1)
        return vector

    def att_self_pool(self, inputs, part2_mask, part3_mask):
        part2_mask, part3_mask = part2_mask.unsqueeze(2), part3_mask.unsqueeze(2)
        # import ipdb; ipdb.set_trace()
        logits_tab = self.self_proj_tab(inputs)
        # att_tab = torch.softmax(logits_tab.masked_fill(1-part2_mask, -1e6), dim=1)
        att_tab = torch.softmax(logits_tab + (1-part2_mask) * -1e6, dim=1)
        pooled2 = torch.sum(att_tab*inputs, dim=1).squeeze()

        logits_psg = self.self_proj_psg(inputs)
        # att_psg = torch.softmax(logits_psg.masked_fill(1-part3_mask, -1e6), dim=1)
        att_psg = torch.softmax(logits_psg + (1-part3_mask) * -1e6, dim=1)
        pooled3 = torch.sum(att_psg*inputs, dim=1).squeeze()

        return pooled2, pooled3

    def att_self_multihead(self, inputs, part2_mask, part3_mask):
        extended_part2_mask = part2_mask[:, None, None, :].to(part2_mask.device)
        extended_part3_mask = part3_mask[:, None, None, :].to(part3_mask.device)
        logits_tab = self.self_proj_tab(inputs, inputs, inputs, attention_mask=extended_part2_mask)
        logits_psg = self.self_proj_psg(inputs, inputs, inputs, attention_mask=extended_part3_mask)
        pooled2 = pooling_masked_part(logits_tab, part2_mask, method='sum')
        pooled3 = pooling_masked_part(logits_psg, part3_mask, method='sum')
        return pooled2, pooled3



class RobertaSingleRetrieverThreeCatAtt(SingleRetrieverThreeCatAtt):

    def __init__(self, config, args):
        super(RobertaSingleRetrieverThreeCatAtt, self).__init__(config, args)

    def forward(self, batch):
        c_cls = self.encode_seq(batch['c_input_ids'], batch['c_mask'], part2_mask=batch['c_part2_mask'], part3_mask=batch['c_part3_mask'])
        neg_c_cls = self.encode_seq(batch['neg_input_ids'], batch['neg_mask'], part2_mask=batch['neg_part2_mask'], part3_mask=batch['neg_part3_mask'])
        q_cls = self.encode_q(batch['q_input_ids'], batch['q_mask'], part2_mask=batch['q_part2_mask'], part3_mask=batch['q_part3_mask'])
        return {'q': q_cls, 'c': c_cls, 'neg_c': neg_c_cls}

    def evaluate_encode_tb(self, batch):
        c_cls = self.encode_seq(batch['input_ids'], batch['input_mask'], part2_mask=batch['part2_mask'], part3_mask=batch['part3_mask'])
        return {'embed': c_cls}

    def evaluate_encode_que(self, batch):
        q_cls = self.encode_q(batch['input_ids'], batch['input_mask'], part2_mask=batch['part2_mask'], part3_mask=batch['part3_mask'])
        return {'embed': q_cls}


class RobertaSingleEncoderThreeCatAtt(SingleEncoderThreeCatAtt):

    def __init__(self, config, args):
        super(RobertaSingleEncoderThreeCatAtt, self).__init__(config, args)
        self.encode_table = args.encode_table

    def forward(self, batch):
        if self.encode_table:
            cls = self.encode_seq(batch['input_ids'], batch['input_mask'], part2_mask=batch['part2_mask'], part3_mask=batch['part3_mask'])
        else:
            cls = self.encode_q(batch['input_ids'], batch['input_mask'], part2_mask=batch['part2_mask'], part3_mask=batch['part3_mask'])
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

