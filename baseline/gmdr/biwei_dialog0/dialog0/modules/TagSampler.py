from __future__ import division
import torch
import torch.nn as nn
from dialog0.modules.RNNEncoder import RNNEncoder
import torch.nn.functional as F

class TagSampler(nn.Module):
    def __init__(self,rnn_encoder):
        super(TagSampler, self).__init__()
        self.rnn_encoder = rnn_encoder
        self.linear1 = nn.Linear(rnn_encoder.hidden_size*2, rnn_encoder.hidden_size)
        self.linear2 = nn.Linear(rnn_encoder.hidden_size, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(-1)
    def forward(self, input, tag_embeddings):
        emb = input
        encoder_outputs, (h,c) = self.rnn_encoder(emb)
        context = c[-1].squeeze()
        # context: batch x dim
        # tag_emb: tag_size x dim
        # state: batch x tag_size x dim

        # tag_emb -> 1 x tag_size x dim -> batch x tag_size x dim
        # context -> batch x 1 x dim -> batch x tag_size x dim
        batch_size = context.size(0)
        tag_size = tag_embeddings.size(0)
        tag_embeddings = tag_embeddings.unsqueeze(0)
        context = context.unsqueeze(1)
        # tag_embeddings = tag_embeddings.expand(batch_size,tag_size,tag_embeddings.size(2))
        context = context.expand(batch_size,tag_size,context.size(2))
        state = torch.cat([tag_embeddings,context],dim=-1)
        # state: batch x tag_size x dim*2 -> batch x tag_size x dim
        state = self.linear1(state)
        state = self.relu(state)
        # state: batch x tag_size x dim -> batch x tag_size x 1 -> batch x 1 x tag_size
        state = self.linear2(state).transpose(2,1)
        output = self.softmax(state)
        return output