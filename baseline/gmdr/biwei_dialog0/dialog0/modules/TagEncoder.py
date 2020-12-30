from __future__ import division
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import torch.nn.functional as F


class TagEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, padding_idx=1):
        super(TagEncoder, self).__init__()
        self.lin_out = nn.Linear(input_size,hidden_size)
        self.tanh = nn.Tanh()
    def forward(self, input_seqs):
        embedded = input_seqs
        embedded = embedded.sum(dim=0,keepdim=True)
        embedded = self.lin_out(embedded)
        output = self.tanh(embedded)
        return output