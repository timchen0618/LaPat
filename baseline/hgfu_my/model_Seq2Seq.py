import random

import torch
import torch.nn as nn

from modules import GlobalAttention, RNNEncoder

class Seq2Seq(nn.Module):
    """docstring for HGFU"""
    def __init__(self, config, use_cuda):
        super(Seq2Seq, self).__init__()
        self.config = config
        hidden_size = config['hidden_size']
        self.hidden_size = hidden_size
        self.encoder = RNNEncoder(rnn_type=config['rnn_type'],
                                  input_size=config['input_size'],
                                  hidden_size=config['hidden_size'],
                                  num_layers=config['encoder_num_layers'],
                                  dropout=config['dropout'],
                                  bidirectional=config['bidirectional'])
        self.decoder = getattr(nn, config['rnn_type'])(
                    input_size=config['input_size'],
                    hidden_size=config['hidden_size'],
                    num_layers=config['decoder_num_layers'],
                    dropout=config['dropout'])

        self.embedding = nn.Embedding(config['vocab_size'], config['input_size'], padding_idx=config['padding_idx'])
        # self.cue_decoder = RNNDecoder(rnn_type=config['rnn_type'],
        #                           attn_type=config['attn_type'],
        #                           input_size=config['input_size'],
        #                           hidden_size=config['hidden_size'],
        #                           num_layers=config['decoder_num_layers'],
        #                           dropout=config['dropout'])
        if config['attn_type'] != 'none':
            self.attn = GlobalAttention(hidden_size, config['attn_type'])

        self._cuda = use_cuda

        self.presm_layer = nn.Linear(hidden_size, config['vocab_size'])
        self.sm = nn.LogSoftmax(dim=-1)
        self.dropout = nn.Dropout(p=config['dropout'])
        self.ss = config['schedule_sampling']
        if self.ss:
            print('Do schedule_sampling...')

    def forward(self, batch):
        total_training_steps = 332000
        eps = 1.0 - float(batch['step'])/total_training_steps

        inp = batch['src'].long().transpose(0, 1)
        dec_inp = batch['tgt'].long().transpose(0, 1)
        lengths = batch['lengths'].long()
        if self._cuda:
            inp = inp.cuda()
            dec_inp = dec_inp.cuda()
            lengths = lengths.cuda()
        

        enc_out, hidden = self.encoder(self.embedding(inp), lengths)
        
        dec_inp = self.embedding(dec_inp)
        max_step, batch_size, dim = dec_inp.size()
        dec_hidden = hidden

        if self.ss:
            # do decoding
            for t in range(0, max_step):
                if t == 0:
                    step_emb = dec_inp[:1]
                else:
                    if random.random() > eps:
                        step_emb = self.embedding(step_pred.long())
                    else:
                        step_emb = dec_inp[t].unsqueeze(0)


                _, dec_hidden = self.decoder(step_emb, dec_hidden)

                # do attention
                attn_outputs, _ = self.attn(
                        dec_hidden.transpose(0, 1).contiguous(),  # (output_len, batch, d)
                        enc_out.transpose(0, 1).contiguous()       # (contxt_len, batch, d)
                    )
                step_out = self.sm(self.presm_layer(self.dropout(attn_outputs)))  # (1, batch, d)

                step_pred = step_out.argmax(dim=-1)

                if t == 0:
                    outputs = step_out
                else:
                    outputs = torch.cat((outputs, step_out), dim = 0)

                del step_out
        else:
            # do decoding
            dec_out, hy = self.decoder(dec_inp, dec_hidden)

            # do attention
            attn_outputs, attn_scores = self.attn(
                    dec_out.transpose(0, 1).contiguous(),  # (output_len, batch, d)
                    enc_out.transpose(0, 1).contiguous()       # (contxt_len, batch, d)
                )
            outputs = self.presm_layer(self.dropout(attn_outputs))

        return outputs.squeeze()


    def generate(self, batch, max_length, bos_token):
        return self._greedy_decode(batch, max_length, bos_token)

    def _greedy_decode(self, batch, max_step, bos_token):
        inp = batch['src'].long().transpose(0, 1)
        lengths = batch['lengths'].long()
        # print('inp', inp.size())
        # print('len', lengths.size())
        # print('word', word.size())
        if self._cuda:
            inp = inp.cuda()
            lengths = lengths.cuda()
        

        enc_out, hidden = self.encoder(self.embedding(inp), lengths)
        dec_hidden = hidden
        step_inp = torch.zeros((inp.size(1), )).fill_(bos_token).long().cuda()
        step_inp = self.embedding(step_inp).unsqueeze(0)

        # do decoding
        for t in range(0, max_step):
            gen_out, hy = self.decoder(step_inp, dec_hidden)
            
            dec_hidden = hy

            # do attention
            attn_outputs, attn_scores = self.attn(
                    dec_hidden.transpose(0, 1).contiguous(),  # (output_len, batch, d)
                    enc_out.transpose(0, 1).contiguous()       # (contxt_len, batch, d)
                )
            step_out = self.sm(self.presm_layer(self.dropout(attn_outputs)))  # (1, batch, d)

            step_pred = step_out.argmax(dim=-1)
            step_inp = self.embedding(step_pred.long())

            if t == 0:
                outputs = step_pred
            else:
                outputs = torch.cat((outputs, step_pred), dim = 0)
        

        return outputs.squeeze()