from __future__ import division
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import torch.nn.functional as F


class GlobalAttention(nn.Module):
    def __init__(self, dim, attn_type="dot"):
        super(GlobalAttention, self).__init__()
        self.dim = dim
        self.attn_type = attn_type
        assert (self.attn_type in ["dot", "general", "mlp"]), (
                "Please select a valid attention type.")
        if self.attn_type == "general":
            self.linear_in = nn.Linear(dim, dim, bias=False)

        # mlp wants it with bias
        out_bias = self.attn_type == "mlp"
        self.linear_out = nn.Linear(dim*2, dim, bias=out_bias)

        self.sm = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask

    def score(self, h_t, h_s):
        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()

        if self.attn_type in ["general", "dot"]:
            if self.attn_type == "general":
                h_t_ = h_t.view(tgt_batch*tgt_len, tgt_dim)
                h_t_ = self.linear_in(h_t_)
                h_t = h_t_.view(tgt_batch, tgt_len, tgt_dim)
            h_s_ = h_s.transpose(1, 2)
            # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
            return torch.bmm(h_t, h_s_)

    def forward(self, input, context):

        # one step input
        if input.dim() == 2:
            one_step = True
            input = input.unsqueeze(1)
        else:
            one_step = False


        batch, sourceL, dim = context.size()
        batch_, targetL, dim_ = input.size()

        # compute attention scores, as in Luong et al.
        align = self.score(input, context)

        # Softmax to normalize attention weights
        align_vectors = self.sm(align.view(batch*targetL, sourceL))
        align_vectors = align_vectors.view(batch, targetL, sourceL)

        # each context vector c_t is the weighted average
        # over all the source hidden states
        c = torch.bmm(align_vectors, context)

        # concatenate
        concat_c = torch.cat([c, input], 2).view(batch*targetL, dim*2)
        attn_h = self.linear_out(concat_c).view(batch, targetL, dim)
        if self.attn_type in ["general", "dot"]:
            attn_h = self.tanh(attn_h)


        if one_step:
            attn_h = attn_h.squeeze(1)
            align_vectors = align_vectors.squeeze(1)
        else:

            attn_h = attn_h.transpose(0, 1).contiguous()
            align_vectors = align_vectors.transpose(0, 1).contiguous()

        # Check output sizes
        # targetL_, batch_, dim_ = attn_h.size()

        # targetL_, batch_, sourceL_ = align_vectors.size()
        return attn_h, align_vectors


class RNNEncoder(nn.Module):
    '''
        RNN Encoder
    '''
    def __init__(self, rnn_type, input_size, 
                hidden_size, num_layers=1, 
                dropout=0.1, bidirectional=False):
        super(RNNEncoder, self).__init__()
        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.no_pack_padded_seq = False
        self.rnn = getattr(nn, rnn_type)(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    bidirectional=bidirectional)

    def forward(self, input, lengths=None, hidden=None):
        packed_emb = input
        packed_emb = pack(input, lengths, enforce_sorted=False)

        outputs, hidden_t = self.rnn(packed_emb, hidden)
        outputs = unpack(outputs)[0]

        # consider both direction
        if self.bidirectional:
            if self.rnn_type == 'LSTM':
                h_n, c_n = hidden_t
                h_n = torch.cat([h_n[0:h_n.size(0):2], h_n[1:h_n.size(0):2]], 2)
                c_n = torch.cat([c_n[0:c_n.size(0):2], c_n[1:c_n.size(0):2]], 2)
                hidden_t = (h_n, c_n)
            else:
                hidden_t = torch.cat([hidden_t[0:hidden_t.size(0):2], hidden_t[1:hidden_t.size(0):2]], 2)
        return outputs, hidden_t



class RNNDecoder(nn.Module):
    """ The GlobalAttention-based RNN decoder. """
    def __init__(self, rnn_type, attn_type, input_size,
                hidden_size,  num_layers=1, dropout=0.1):
        super(RNNDecoder, self).__init__()
        # Basic attributes.
        self.rnn_type = rnn_type
        self.attn_type = attn_type
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.rnn = getattr(nn, rnn_type)(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout)

        if self.attn_type != 'none':
            self.attn = GlobalAttention(hidden_size, attn_type)

    def forward(self, input, context, state):
        emb = input
        rnn_outputs, hidden = self.rnn(emb, state)

        if self.attn_type != 'none':
            # Calculate the attention.
            attn_outputs, attn_scores = self.attn(
                rnn_outputs.transpose(0, 1).contiguous(),  # (output_len, batch, d)
                context.transpose(0, 1)                   # (contxt_len, batch, d)
            )
            outputs  = self.dropout(attn_outputs)    # (input_len, batch, d)

        else:
            outputs  = self.dropout(rnn_outputs)
        return outputs , hidden