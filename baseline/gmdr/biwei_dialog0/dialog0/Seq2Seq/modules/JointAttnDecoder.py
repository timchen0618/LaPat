from __future__ import division
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import torch.nn.functional as F
from dialog0.Seq2Seq.modules.JointAttention import JointAttention

class JointAttnDecoder(nn.Module):
    """ The GlobalAttention-based RNN decoder. """
    def __init__(self, rnn_type, attention, input_size,
                hidden_size,  num_layers=1, dropout=0.1):
        super(JointAttnDecoder, self).__init__()
        # Basic attributes.
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.rnn = getattr(nn, rnn_type)(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout)


        self.attn = attention

    def cal_tag_atten(self, tag_hidden, context):
        tag_c = self.attn.cal_tag_atten(tag_hidden,
                                 context.transpose(0, 1).contiguous())
        return tag_c


    def forward(self, input, tag_hidden, context, state):
        emb = input
        rnn_outputs, hidden = self.rnn(emb, state)

        tag_hidden = tag_hidden.expand_as(rnn_outputs)
        # joint_hidden = torch.cat([tag_hidden.expand_as(rnn_outputs),rnn_outputs],dim=-1)
        # Calculate the attention.
        attn_outputs, attn_scores = self.attn(
            rnn_outputs.transpose(0, 1).contiguous(),
            tag_hidden.transpose(0, 1).contiguous(),  # (output_len, batch, d)
            context.transpose(0, 1)                   # (contxt_len, batch, d)
        )
        outputs  = self.dropout(attn_outputs)    # (input_len, batch, d)


        return outputs , hidden, attn_scores