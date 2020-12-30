import torch
import torch.nn as nn


class Seq2SeqWithRL(nn.Module):
    def __init__(self, sampler_model, seq2seq_model):
        super(Seq2SeqWithRL, self).__init__()

        self.sampler_model = sampler_model
        self.seq2seq_model = seq2seq_model

    def sample_latent_sentence(self, enc_batch, enc_lens):
        log_probs = self.sampler_model.forward(enc_batch=enc_batch, enc_lens=enc_lens)

        return log_probs

    def drop_checkpoint(self, epoch, fname):
        torch.save({'sampler_model_state_dict': self.sampler_model.state_dict(),
                    'seq2seq_model_state_dict': self.seq2seq_model.state_dict(),
                    'epoch': epoch},
                    fname)

    def load_checkpoint(self, cpnt):
        cpnt = torch.load(cpnt,map_location=lambda storage, loc: storage)
        self.sampler_model.load_state_dict(cpnt['sampler_model_state_dict'])
        self.seq2seq_model.load_state_dict(cpnt['seq2seq_model_state_dict'])
        epoch = cpnt['epoch']
        return epoch
