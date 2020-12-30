from __future__ import division
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import torch.nn.functional as F


class TagEmbeddings(nn.Module):
    def __init__(self, input_size, embedding_size, padding_idx=1):
        super(TagEmbeddings, self).__init__()
        self.padding_idx = padding_idx
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=padding_idx)
    def forward(self, input_seqs):
        embedded = self.embedding(input_seqs)
        return embedded