import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack 

class RNNEncoder(nn.Module):
    '''
        RNN Encoder
    '''
    def __init__(self, rnn_type, input_size, 
                hidden_size, num_layers=1, 
                dropout=0.1, bidirectional=False):
        super(RNNEncoder, self).__init__()
        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.no_pack_padded_seq = False
        self.rnn = getattr(nn, rnn_type)(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    bidirectional=bidirectional)

    def forward(self, input, lengths=None, hidden=None):
        packed_emb = input
        packed_emb = pack(input, lengths, enforce_sorted=False)

        outputs, hidden_t = self.rnn(packed_emb, hidden)
        outputs = unpack(outputs)[0]

        # consider both direction
        if self.bidirectional:
            if rnn_type == 'LSTM':
                h_n, c_n = hidden_t
                h_n = torch.cat([h_n[0:h_n.size(0):2], h_n[1:h_n.size(0):2]], 2)
                c_n = torch.cat([c_n[0:c_n.size(0):2], c_n[1:c_n.size(0):2]], 2)
                hidden_t = (h_n, c_n)
            else:
                hidden_t = torch.cat([hidden_t[0:hidden_t.size(0):2], hidden_t[1:hidden_t.size(0):2]], 2)


