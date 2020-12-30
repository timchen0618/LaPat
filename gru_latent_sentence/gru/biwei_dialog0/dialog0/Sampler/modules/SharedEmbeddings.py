import pickle
import torch
import torch.nn as nn


class Embeddings(nn.Module):
    def __init__(self, input_size, embedding_dim):
        super(Embeddings, self).__init__()

        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.embedding.weight.requires_grad = True

    def forward(self, input_seqs):
        embedded = self.embedding(input_seqs)
        return embedded
