import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from numpy import random
from dialog0.Sampler import config

random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)


def init_gru_wt(gru):
    for names in gru._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(gru, name)
                wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(gru, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)


def init_linear_wt(linear):
    linear.weight.data.normal_(std=config.trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=config.trunc_norm_init_std)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.gru = nn.GRU(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        init_gru_wt(self.gru)

    #seq_lens should be in descending order
    def forward(self, input_embed, seq_lens):
        packed = pack_padded_sequence(input_embed, seq_lens, batch_first=True, enforce_sorted = False)
        output, hidden = self.gru(packed)

        encoder_outputs = pad_packed_sequence(output, batch_first=True)[0]

        hidden_t = torch.cat([hidden[0:hidden.size(0):2], hidden[1:hidden.size(0):2]], 2)

        return encoder_outputs, hidden_t


class Classifier(nn.Module):
    def __init__(self, rnn_output_size, output_size):
        super(Classifier, self).__init__()

        self.classify_layer = nn.Sequential(nn.Linear(rnn_output_size, rnn_output_size),
                                            nn.Tanh(),
                                            # nn.ReLU(),
                                            nn.Linear(rnn_output_size, rnn_output_size),
                                            nn.Tanh(),
                                            # nn.ReLU(),
                                            nn.Linear(rnn_output_size, output_size),
                                            nn.LogSoftmax(-1))

    def forward(self, enc_hidden):
        log_prob = self.classify_layer(enc_hidden)
        log_prob = log_prob.squeeze()

        return log_prob


class Model(nn.Module):
    def __init__(self, model_file_path=None, is_eval=False, hidden_dim=500, output_size=50000):
        super(Model, self).__init__()
        encoder = Encoder()
        classifier = Classifier(rnn_output_size=hidden_dim*2, output_size=output_size)

        if is_eval:
            encoder = encoder.eval()
            classifier = classifier.eval()

        self.encoder = encoder
        self.classifier = classifier

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location= lambda storage, location: storage)
            self.encoder.load_state_dict(state['sampler_encoder_state_dict'])
            self.classifier.load_state_dict(state['sampler_classifier_state_dict'])
