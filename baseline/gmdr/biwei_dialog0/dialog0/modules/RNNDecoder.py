from __future__ import division
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import torch.nn.functional as F
from dialog0.modules.GlobalAttention import GlobalAttention

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