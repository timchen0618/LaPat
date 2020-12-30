import torch
import torch.nn as nn

from modules import GlobalAttention, RNNEncoder

class HGFU(nn.Module):
    """docstring for HGFU"""
    def __init__(self, config, use_cuda):
        super(HGFU, self).__init__()
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
        self.cue_decoder = getattr(nn, config['rnn_type'])(
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
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.Wk = nn.Linear(hidden_size*2, hidden_size)
        self.presm_layer = nn.Linear(hidden_size, config['vocab_size'])
        self.sm = nn.LogSoftmax(dim=-1)
        self.dropout = nn.Dropout(p=config['dropout'])

    def forward(self, batch):
        inp = batch['src'].long().transpose(0, 1)
        dec_inp = batch['tgt'].long().transpose(0, 1)
        lengths = batch['lengths'].long()
        word = self.embedding(batch['word'].long())
        if self._cuda:
            inp = inp.cuda()
            dec_inp = dec_inp.cuda()
            lengths = lengths.cuda()
        

        enc_out, hidden = self.encoder(self.embedding(inp), lengths)
        
        dec_inp = self.embedding(dec_inp)
        max_step, batch_size, dim = dec_inp.size()
        dec_hidden = hidden

        # do decoding
        for t in range(0, max_step):
            gen_out, hy = self.decoder(dec_inp[t].unsqueeze(0), dec_hidden)
            cue_out, hw = self.cue_decoder(word.unsqueeze(0), dec_hidden)
            hy_ = torch.tanh(self.W1(hy))
            hw_ = torch.tanh(self.W2(hw))
            k = torch.sigmoid(self.Wk(torch.cat((hy_, hw_), dim=-1)))
            dec_hidden = k * hy + (1-k) * hw
            if t == 0:
                dec_out = dec_hidden
            else:
                dec_out = torch.cat((dec_out, dec_hidden), dim=0)

        # do attention
        attn_outputs, attn_scores = self.attn(
                dec_out.transpose(0, 1).contiguous(),  # (output_len, batch, d)
                enc_out.transpose(0, 1).contiguous()       # (contxt_len, batch, d)
            )
        outputs = self.presm_layer(self.dropout(attn_outputs))
        return self.sm(outputs)


    def generate(self, batch, max_length, bos_token):
        return self._greedy_decode(batch, max_length, bos_token)

    def _greedy_decode(self, batch, max_step, bos_token):
        inp = batch['src'].long().transpose(0, 1)
        lengths = batch['lengths'].long()
        word = self.embedding(batch['word'].long())
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
            cue_out, hw = self.cue_decoder(word.unsqueeze(0), dec_hidden)
            hy_ = torch.tanh(self.W1(hy))
            hw_ = torch.tanh(self.W2(hw))
            k = torch.sigmoid(self.Wk(torch.cat((hy_, hw_), dim=-1)))
            dec_hidden = k * hy + (1-k) * hw
            
            dec_out = dec_hidden

            # do attention
            attn_outputs, attn_scores = self.attn(
                    dec_out.transpose(0, 1).contiguous(),  # (output_len, batch, d)
                    enc_out.transpose(0, 1).contiguous()       # (contxt_len, batch, d)
                )
            step_out = self.sm(self.presm_layer(self.dropout(attn_outputs)))  # (1, batch, d)

            step_pred = step_out.argmax(dim=-1)

            step_inp = self.embedding(step_pred.long())

            if t == 0:
                outputs = step_pred
            else:
                outputs = torch.cat((outputs, step_pred), dim = 0)
                # print('outputs', outputs)

        return outputs.squeeze()