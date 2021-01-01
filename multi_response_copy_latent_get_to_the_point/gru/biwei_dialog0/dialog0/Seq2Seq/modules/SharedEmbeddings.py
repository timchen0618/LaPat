import torch
import torch.nn as nn
from dialog0.Seq2Seq import config

def init_wt_normal(wt):
    wt.data.normal_(std=config.trunc_norm_init_std)

class Embeddings(nn.Module):
    def __init__(self, input_size, embedding_dim):
        super(Embeddings, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(input_size, embedding_dim)
        init_wt_normal(self.embedding.weight)

    def forward(self, input_seqs):
        embedded = self.embedding(input_seqs)
        return embedded

