from __future__ import division
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import torch.nn.functional as F
import math

class JointAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim, tag_dim, attn_type="general"):
        super(JointAttention, self).__init__()
        self.enc_dim = enc_dim
        self.dec_dim = dec_dim
        self.tag_dim = tag_dim
        self.attn_type = attn_type
        assert (self.attn_type in ["dot", "general", "mlp"]), (
                "Please select a valid attention type.")
        if self.attn_type == "general":
            self.dec_linear_in = nn.Linear(dec_dim, enc_dim, bias=False)
            self.tag_linear_in = nn.Linear(tag_dim, enc_dim, bias=False)
        # mlp wants it with bias
        out_bias = self.attn_type == "mlp"
        self.linear_out = nn.Linear(enc_dim+dec_dim+enc_dim, dec_dim, bias=out_bias)

        self.sm = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask

    def dec_score(self, h_t, h_s):
        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()

        if self.attn_type in ["general", "dot"]:
            if self.attn_type == "general":
                h_t_ = h_t.view(tgt_batch*tgt_len, tgt_dim)
                h_t_ = self.dec_linear_in(h_t_)
                h_t = h_t_.view(tgt_batch, tgt_len, self.enc_dim)
            h_s_ = h_s.transpose(1, 2)
            # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
            return torch.bmm(h_t, h_s_)

    def tag_score(self, h_t, h_s):
        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()

        if self.attn_type in ["general", "dot"]:
            if self.attn_type == "general":
                h_t_ = h_t.view(tgt_batch*tgt_len, tgt_dim)
                h_t_ = self.tag_linear_in(h_t_)
                h_t = h_t_.view(tgt_batch, tgt_len, self.enc_dim)
            h_s_ = h_s.transpose(1, 2)
            # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
            return torch.bmm(h_t, h_s_)

    def cal_tag_atten(self, tag_input, context):

        tag_input = tag_input.unsqueeze(0)
        batch, sourceL, dim = context.size()
        batch_, targetL, dim_ = tag_input.size()
        tag_align = self.tag_score(tag_input, context)
        tag_align = tag_align/math.sqrt(self.enc_dim)
        tag_align_vectors = self.sm(tag_align.view(batch, targetL, sourceL))
        tag_align_vectors = tag_align_vectors.view(batch, targetL, sourceL)
        tag_c = torch.bmm(tag_align_vectors, context)

        tag_c = tag_c.squeeze()
        return tag_c



    def forward(self, tgt_input, tag_input, context):

        # one step input
        if tgt_input.dim() == 2:
            one_step = True
            tgt_input = tgt_input.unsqueeze(1)
        else:
            one_step = False


        batch, sourceL, dim = context.size()
        batch_, targetL, dim_ = tgt_input.size()

        # compute attention scores, as in Luong et al.
        tgt_align = self.dec_score(tgt_input, context)
        tag_align = self.tag_score(tag_input, context)

        # smooth
        tgt_align = tgt_align/math.sqrt(self.enc_dim)
        tag_align = tag_align/math.sqrt(self.enc_dim)
        # Softmax to normalize attention weights
        tgt_align_vectors = self.sm(tgt_align.view(batch*targetL, sourceL))
        tgt_align_vectors = tgt_align_vectors.view(batch, targetL, sourceL)
        tag_align_vectors = self.sm(tag_align.view(batch*targetL, sourceL))
        tag_align_vectors = tag_align_vectors.view(batch, targetL, sourceL)
            
        # each context vector c_t is the weighted average
        # over all the source hidden states
        src_c = torch.bmm(tgt_align_vectors, context)
        tag_c = torch.bmm(tag_align_vectors, context)
        # concatenate
        concat_c = torch.cat([src_c, tag_c, tgt_input], 2).view(batch*targetL, self.enc_dim+self.enc_dim+self.dec_dim)
        attn_h = self.linear_out(concat_c).view(batch, targetL, self.dec_dim)
        if self.attn_type in ["general", "dot"]:
            attn_h = self.tanh(attn_h)


        if one_step:
            attn_h = attn_h.squeeze(1)
            tag_align_vectors = tag_align_vectors.squeeze(1)
            tgt_align_vectors = tgt_align_vectors.squeeze(1)
            tag_c = tag_c.squeeze(1)
        else:

            attn_h = attn_h.transpose(0, 1).contiguous()
            tag_align_vectors = tag_align_vectors.transpose(0, 1).contiguous()
            tgt_align_vectors = tgt_align_vectors.transpose(0, 1).contiguous()
            tag_c = tag_c.transpose(0, 1).contiguous()


        # align_vectors = (tgt_align_vectors,tag_align_vectors)
        attns = {}
        attns['ctx'] = tgt_align_vectors.data.clone()
        attns['tag'] = tag_align_vectors.data.clone()
        attns['tag_c'] = tag_c.data.clone()
        return attn_h, attns