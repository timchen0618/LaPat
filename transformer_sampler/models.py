import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import *


class Sampler(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, src_embed, generator):
        super(Sampler, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.generator = generator
        
    def forward(self, src, src_mask):
        "Take in and process masked src and target sequences."
        emb = self.src_embed(src)
        # print(emb.grad)
        enc_out = self.encoder(emb, src_mask)[:, 0, :]
        # enc_out = self.encoder(emb, src_mask).mean(dim = 1)
        # print('enc', enc_out.grad)
        return self.generator(enc_out)
    
    def load_embedding(self, path):
        state_dict = torch.load(path)['state_dict']
        own_state = self.state_dict()
        # self.src_embed[0].lut.weight.data = state_dict['src_embed.0.lut.weight']
        
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            # print('name: ', name)
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)

    def loss_compute(self, out, y, multi):
        #out -> [b, ]
        if multi:
            losses = []
            for index, target in enumerate(y):
                true_dist = out[index].data.clone().fill_(0.).unsqueeze(0)
                true_dist.scatter_(1, torch.LongTensor(target).cuda().unsqueeze(0), 1.)
                loss = -(true_dist*out[index]).sum()
                losses.append(loss.unsqueeze(0))
            return torch.cat(losses, dim = 0).mean()
        else:
            # print('out', out.size())
            # print('y', y.size())  
            true_dist = out.data.clone()
            true_dist.fill_(0.)
            true_dist.scatter_(1, y.long().unsqueeze(1), 1.)
            return -(true_dist*out).sum(dim=1).mean()

    def compute_acc(self, out, y, multi):
        if multi:
            pred = out.argmax(dim=-1)
            acc = 0
            total = 0
            for i, target in enumerate(y):
                if pred[i] in target:
                    acc += 1
                total += 1          
            return float(acc)/len(y)

        else:
            pred = out.argmax(dim = -1)
        return (pred == y).sum().item()/(pred.fill_(1.0).sum().float())

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab, dropout):
        super(Generator, self).__init__()
        # layer_list = [(nn.Linear(d_model, d_model), nn.ReLU()) for i in range(num_layers)] + \
        #              [nn.Linear(d_model, vocab), nn.Dropout(dropout)]
        # print('layer_list', layer_list)
        # self.proj = nn.Sequential(*layer_list)
        hidden = 512
        self.proj = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, vocab),
            nn.Dropout(dropout)
            )

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


    
class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)[0])
        return self.sublayer[1](x, self.feed_forward)

  
