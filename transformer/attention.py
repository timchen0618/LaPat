import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e15)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    # mean on dim of head
    attn_dist = F.softmax(scores.mean(dim = 1), dim=-1)
    # print('attn_dist', attn_dist.size())
    return torch.matmul(p_attn, value), attn_dist


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        # self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            # mask = (nbatch, 1, seq_len, seq_len)
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        # query,key,value shape: (nbatches, h, seq_len, d_k)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, attn_dist = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        # print("attn")
        # print(self.attn.size())
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x), attn_dist

    # @torch.jit.export
    # def reorder_incremental_state(
    #     self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]], new_order: Tensor
    # ):
    #     """Reorder buffered internal state (for incremental generation)."""
    #     input_buffer = self._get_input_buffer(incremental_state)
    #     if input_buffer is not None:
    #         for k in input_buffer.keys():
    #             input_buffer_k = input_buffer[k]
    #             if input_buffer_k is not None:
    #                 if self.encoder_decoder_attention and input_buffer_k.size(0) == new_order.size(0):
    #                     break
    #                 input_buffer[k] = input_buffer_k.index_select(0, new_order)
    #         incremental_state = self._set_input_buffer(incremental_state, input_buffer)
    #     return incremental_state