from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class TagSampler(nn.Module):
    def __init__(self,shared_embeddings, rnn_encoder, output_size):
        super(TagSampler, self).__init__()
        self.shared_embeddings = shared_embeddings
        self.rnn_encoder = rnn_encoder

        self.classifier = nn.Sequential(
                                nn.Linear(rnn_encoder.output_size, rnn_encoder.output_size),
                                nn.Tanh(),
                                nn.Linear(rnn_encoder.output_size, rnn_encoder.output_size),
                                nn.Tanh(),
                                nn.Linear(rnn_encoder.output_size, output_size),
                                nn.LogSoftmax(-1)
                          )
        # self.out = nn.Linear(1000, 50003)

    def forward(self, src_inputs, src_lengths):
        src_emb = self.shared_embeddings(src_inputs)
        # tag_emb = self.shared_embeddings(tag_inputs)
        encoder_outputs, enc_hidden = self.rnn_encoder(src_emb,src_lengths)
        log_prob = self.classifier(enc_hidden)
        #log_prob = self.out(enc_hidden)
        log_prob = log_prob.squeeze()
        # print('log_prob', log_prob.size())
        return log_prob

    def drop_checkpoint(self, epoch, opt, fname):
        torch.save({'tag_sampler_dict': self.state_dict(),
                    'epoch': epoch,
                    'opt': opt,
                    },
                   fname)


    def load_checkpoint(self, cpnt):
        cpnt = torch.load(cpnt,map_location=lambda storage, loc: storage)
        self.load_state_dict(cpnt['tag_sampler_dict'])
        epoch = cpnt['epoch']
        return epoch
