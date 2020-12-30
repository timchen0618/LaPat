import torch
import torch.nn as nn
from torch.autograd import Variable


class Seq2SeqWithTag(nn.Module):
    def __init__(self, 
                 shared_embeddings, 
                 tag_encoder,
                 encoder,
                 decoder, 
                 generator, 
                 feat_merge, 
                 use_tag=False):
        super(Seq2SeqWithTag, self).__init__()
        self.shared_embeddings = shared_embeddings
        self.tag_encoder = tag_encoder
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.use_tag = use_tag
        self.feat_merge = feat_merge
    def forward(self, input):
        src_inputs = input[0]
        tgt_inputs = input[1]
        tag_inputs = input[2]
        src_lengths = input[3]
        # Run wrods through encoder
        encoder_outputs, encoder_hidden = self.encode(src_inputs, src_lengths, None)
        tag_hidden = self.tag_encode(tag_inputs)

        decoder_init_hidden = self.init_decoder_state((encoder_hidden,tag_hidden))
        decoder_outputs, decoder_hiddens, attn_scores = \
                    self.decode(
                tgt_inputs, tag_hidden, encoder_outputs, decoder_init_hidden
                                )

        return decoder_outputs

    def init_decoder_state(self, input):
        enc_hidden = input[0]
        tag_hidden = input[1]

        
        if not isinstance(enc_hidden, tuple):  # GRU
            h= enc_hidden
            # if self.feat_merge == 'sum':
            #     h = enc_hidden+tag_hidden.expand_as(enc_hidden)
            # elif self.feat_merge == 'concat':
            #     h = torch.cat([enc_hidden,tag_hidden.expand_as(enc_hidden)],dim=-1)
            return h

        else:  # LSTM
            h,c = enc_hidden
            # tag_hidden = tag_hidden.expand_as(c)
            # if self.feat_merge == 'sum':
            #     tag_hidden = tag_hidden.expand_as(c)
            #     h += tag_hidden
            #     c += tag_hidden
            # elif self.feat_merge == 'concat':
            #     h = torch.cat([h,tag_hidden.expand_as(h)],dim=-1)
            #     c = torch.cat([c,tag_hidden.expand_as(c)],dim=-1)
            return (h,c)

    def tag_encode(self, input):
        tag_embeddings = self.shared_embeddings(input)
        tag_hidden = self.tag_encoder(tag_embeddings)
        return tag_hidden


    def encode(self, input, lengths=None, hidden=None):
        encoder_input = self.shared_embeddings(input)
        encoder_outputs, encoder_hidden = self.encoder(encoder_input, lengths, None)

        return encoder_outputs, encoder_hidden

    def decode(self, input, tag_hidden, context, state):
        decoder_input = self.shared_embeddings(input)
        decoder_outputs, decoder_hiddens, attn_scores= self.decoder(
                decoder_input, tag_hidden, context, state
            )
        return decoder_outputs, decoder_hiddens, attn_scores

    def drop_checkpoint(self, epoch, opt, fname):
        torch.save({'shared_embeddings_dict': self.shared_embeddings.state_dict(),
                    'tag_encoder_dict':self.tag_encoder.state_dict(),
                    'encoder_dict': self.encoder.state_dict(),
                    'decoder_dict': self.decoder.state_dict(),
                    'generator_dict': self.generator.state_dict(),
                    'epoch': epoch,
                    'opt': opt,
                    },
                   fname)


    def load_checkpoint(self, cpnt):
        cpnt = torch.load(cpnt,map_location=lambda storage, loc: storage)
        self.shared_embeddings.load_state_dict(cpnt['shared_embeddings_dict'])
        self.tag_encoder.load_state_dict(cpnt['tag_encoder_dict'])
        self.encoder.load_state_dict(cpnt['encoder_dict'])
        self.decoder.load_state_dict(cpnt['decoder_dict'])
        self.generator.load_state_dict(cpnt['generator_dict'])
        epoch = cpnt['epoch']
        return epoch