import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import *
from typing import NamedTuple, Optional

EncoderOut = NamedTuple(
    "EncoderOut",
    [
        ("encoder_out", torch.Tensor),  # T x B x C
        ("encoder_padding_mask", Optional[torch.Tensor]),  # B x T
        # ("encoder_embedding", Optional[Tensor]),  # B x T x C
        # ("encoder_states", Optional[List[Tensor]]),  # List[T x B x C]
        # ("src_tokens", Optional[Tensor]),  # B x T
        # ("src_lengths", Optional[Tensor]),  # B x 1
    ],
)


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask, src)
    
    def forward_with_mask(self, src, tgt, src_mask, tgt_mask, posmask):
        return self.decode_with_mask(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask, src, posmask)
        
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask, src):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask, src)

    def decode_with_mask(self, memory, src_mask, tgt, tgt_mask, src, posmask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask, src, posmask)

    def loss_compute(self, out, y, padding_idx):
        true_dist = out.data.clone()
        true_dist.fill_(0.)
        true_dist.scatter_(2, y.unsqueeze(2), 1.)
        true_dist[:,:,padding_idx] *= 0
        return -(true_dist*out).sum(dim=2).mean()
        #return -torch.gather(out, 2, y.unsqueeze(2)).squeeze(2).mean()


    # def greedy_decode(self, src, src_mask, max_len, start_symbol):
    #     memory = self.encode(src, src_mask)
    #     ys = torch.zeros(1, 1).type_as(src.data) + start_symbol
    #     for i in range(max_len-1):
    #         log_prob = self.decode(memory, src_mask, 
    #                            Variable(ys), 
    #                            Variable(subsequent_mask(ys.size(1))
    #                                     .type_as(src.data)), src)
    #         _, next_word = torch.max(log_prob, dim = -1)
    #         next_word = next_word.data[0,-1]
    #         ys = torch.cat([ys, 
    #                         torch.zeros(1, 1).type_as(src.data)+next_word], dim=1)
    #     return ys


    def greedy_decode(self, src, src_mask, max_len, start_symbol, posmask=None):
        memory = self.encode(src, src_mask)
        ys = torch.zeros(src.size()[0], 1).type_as(src.data) + start_symbol

        for i in range(max_len):
            if posmask != None:
                # print('ppp', posmask[:, i].unsqueeze(1).size())
                log_prob, _ = self.decode_with_mask(memory, src_mask, Variable(ys), 
                               Variable(subsequent_mask(ys.size(1)).type_as(src.data).expand((ys.size(0), 
                               ys.size(1), ys.size(1)))), src, posmask[:, i].unsqueeze(1))
            else:
                log_prob, _ = self.decode(memory, src_mask, Variable(ys), 
                               Variable(subsequent_mask(ys.size(1)).type_as(src.data).expand((ys.size(0), 
                               ys.size(1), ys.size(1)))), src)

            _, next_word = torch.max(log_prob, dim = -1)
            next_word = next_word.data[:,-1]

            ys = torch.cat([ys, 
                            (torch.zeros(src.size()[0], 1).type_as(src.data)) + (next_word.view(-1, 1))], dim=1)

        return ys


# class Sampler(nn.Module):
#     """
#     A standard Encoder-Decoder architecture. Base for this and many 
#     other models.
#     """
#     def __init__(self, encoder, src_embed, generator):
#         super(Sampler, self).__init__()
#         self.encoder = encoder
#         self.src_embed = src_embed
#         self.generator = generator
        
#     def forward(self, src, src_mask):
#         "Take in and process masked src and target sequences."
#         return self.generator(self.encoder(self.src_embed(src), src_mask)[:, -1, :])
    
#     def loss_compute(self, out, y, multi):
#         #out -> [b, ]
#         if multi:
#             losses = []
#             for index, target in enumerate(y):
#                 true_dist = out[index].data.clone().fill_(0.).unsqueeze(0)
#                 true_dist.scatter_(1, torch.LongTensor(target).cuda().unsqueeze(0), 1.)
#                 loss = -(true_dist*out[index]).sum()
#                 losses.append(loss.unsqueeze(0))
#             return torch.cat(losses, dim = 0).mean()
#         else:  
#             true_dist = out.data.clone()
#             true_dist.fill_(0.)
#             true_dist.scatter_(1, y.long().unsqueeze(1), 1.)
#             return -(true_dist*out).sum(dim=1).mean()

#     def compute_acc(self, out, y):
#         pred = out.argmax(dim = -1)
#         return (pred == y).sum().item()

class Sampler(nn.Module):
    def __init__(self, encoder, src_embed, generator):
        super(Sampler, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.generator = generator
        
    def forward(self, src, src_mask):
        "Take in and process masked src and target sequences."
        emb = self.src_embed(src)
        enc_out = self.encoder(emb, src_mask)[:, 0, :]
        # enc_out = self.encoder(emb, src_mask).mean(dim = 1)
        return self.generator(enc_out)
    
    def load_embedding(self, path):
        state_dict = torch.load(path)['state_dict']
        own_state = self.state_dict()
        self.src_embed[0].lut.weight.data = state_dict['src_embed.0.lut.weight']
        
        # for name, param in state_dict.items():
        #     if name not in own_state:
        #          continue
        #     # print('name: ', name)
        #     if isinstance(param, nn.Parameter):
        #         # backwards compatibility for serialized parameters
        #         param = param.data
        #     own_state[name].copy_(param)

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
    def __init__(self, d_model, vocab, dropout, hidden):
        super(Generator, self).__init__()
        # layer_list = [(nn.Linear(d_model, d_model), nn.ReLU()) for i in range(num_layers)] + \
        #              [nn.Linear(d_model, vocab), nn.Dropout(dropout)]
        # print('layer_list', layer_list)
        # self.proj = nn.Sequential(*layer_list)
        # print('hidden', hidden)
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

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: EncoderOut, new_order):
        """
        Reorder encoder output according to *new_order*.
        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order
        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        """
        Since encoder_padding_mask and encoder_embedding are both of type
        Optional[Tensor] in EncoderOut, they need to be copied as local
        variables for Torchscript Optional refinement
        """
        # encoder_padding_mask: Optional[Tensor] = encoder_out.encoder_padding_mask
        # encoder_embedding: Optional[Tensor] = encoder_out.encoder_embedding
        
        # print('encoder_embedding', encoder_embedding.size())
        # print('src_tokens', encoder_out.src_tokens.size())
        # print('src_lengths', encoder_out.src_lengths.size())
        # print('encoder_states', encoder_out.encoder_states.size())

        new_encoder_out = (
            encoder_out.encoder_out
            if encoder_out.encoder_out is None
            else encoder_out.encoder_out.index_select(0, new_order)
        )
        new_encoder_padding_mask = (
            encoder_out.encoder_padding_mask
            if encoder_out.encoder_padding_mask is None
            else encoder_out.encoder_padding_mask.index_select(0, new_order)
        )
        # new_encoder_embedding = (
        #     encoder_embedding
        #     if encoder_embedding is None
        #     else encoder_embedding.index_select(0, new_order)
        # )
        # src_tokens = encoder_out.src_tokens
        # if src_tokens is not None:
        #     src_tokens = src_tokens.index_select(0, new_order)

        # src_lengths = encoder_out.src_lengths
        # if src_lengths is not None:
        #     src_lengths = src_lengths.index_select(0, new_order)

        # encoder_states = encoder_out.encoder_states
        # if encoder_states is not None:
        #     for idx, state in enumerate(encoder_states):
        #         encoder_states[idx] = state.index_select(1, new_order)

        return EncoderOut(
            encoder_out=new_encoder_out,  # T x B x C
            encoder_padding_mask=new_encoder_padding_mask,  # B x T
            # encoder_embedding=new_encoder_embedding,  # B x T x C
            # encoder_states=encoder_states,  # List[T x B x C]
            # src_tokens=src_tokens,  # B x T
            # src_lengths=src_lengths,  # B x 1
        )



class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N, d_model, vocab, pointer_gen):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.proj = nn.Linear(d_model, vocab)
        if pointer_gen:
            print('pointer_gen')
            self.bptr = nn.Parameter(torch.FloatTensor(1, 1))
            self.Wh = nn.Linear(d_model, 1)
            self.Ws = nn.Linear(d_model, 1)
            self.Wx = nn.Linear(d_model, 1)
            self.pointer_gen = True
        else:
            self.pointer_gen = False
        self.sm = nn.LogSoftmax(dim = -1) 

    def forward(self, x, memory, src_mask, tgt_mask, src, posmask=None):
        # index = src;
        # p_gen = 0.8;
        # #print(index.size())
        # for layer in self.layers[:-1]:
        #     x, _ = layer(x, memory, src_mask, tgt_mask)
        # x, attn_weights = self.layers[-1](x, memory, src_mask, tgt_mask)
        # # print(x[:, :, 0])
        # dec_dist = self.proj(self.norm(x))
        # # print(dec_dist.size())
        # # print(dec_dist[:, :, :5])
        # index = index.unsqueeze(1).expand_as(attn_weights)
        # enc_attn_dist = Variable(torch.zeros(dec_dist.size())).cuda().scatter_(-1, index, attn_weights)
        # torch.cuda.synchronize()
        # #print(torch.nonzero(enc_attn_dist))

        # # attn_weights, index = attn_weights.squeeze(1), index.transpose(0,1)
        # # output, attn_weights = (output * p_gen), attn_weights * (1-p_gen)
        # # output = output.scatter_add_(dim = 1, index = index, src = attn_weights)

        # #return (1 - p_gen) * F.log_softmax(dec_dist, dim=-1) + p_gen * 
        # return torch.log(p_gen * F.softmax(dec_dist, dim=-1) + (1 - p_gen)  * enc_attn_dist)
        # #return F.log_softmax(dec_dist + enc_attn_dist, dim=-1)

        ########################################################
        ######################### new ##########################
        ########################################################
        # σ(w>h ht* + w>s st + w>x xt + bptr)
        #memory                     [batch, src_len, d_model]
        #x_t                        [batch, dec_len, d_model]
        #s_t                        [batch, dec_len, d_model]
        #attn_weights               [batch, dec_len, src_len]  16, 100, 400
        #h_t = memory * attn_dist   [batch, dec_len, d_model]

        # every decoder step needs a p_gen -> p_gen   [batch, dec_len]
        if self.pointer_gen:
            index = src
            x_t = x 
        for layer in self.layers[:-1]:
            x, _ = layer(x, memory, src_mask, tgt_mask)
        x, attn_weights = self.layers[-1](x, memory, src_mask, tgt_mask)
        dec_dist = self.proj(self.norm(x))
        
        if self.pointer_gen:
            s_t = x
            h_t = torch.bmm(attn_weights, memory)  #context vector
            p_gen = self.Wh(h_t) + self.Ws(s_t) + self.Wx(x_t) + self.bptr.expand_as(self.Wh(h_t))
            p_gen = torch.sigmoid(p_gen)
            index = index.unsqueeze(1).expand_as(attn_weights)
            enc_attn_dist = Variable(torch.zeros(dec_dist.size())).cuda().scatter_add_(dim = -1, index= index, src=attn_weights)
            # print('==========================')
            # print('attn_weights', attn_weights.sum(dim = -1)[0][0])
            # print('attn_weights', attn_weights.sum(dim = -1)[0][1])
            # print('attn_weights', attn_weights.size())
            # print('index', index.size())
            # print('enc_attn_dist', enc_attn_dist.size())
            # print('enc_attn_dist_sum', enc_attn_dist.sum(dim = -1)[0][0])
            # print('enc_attn_dist_sum', enc_attn_dist.sum(dim = -1)[0][1])
            # print('enc_attn_dist_sum1', enc_attn_dist.sum(dim=-1).size())
            # print('dec_dist', self.sm(dec_dist).size())
            # print('dec_sum', self.sm(dec_dist).sum(dim = -1)[0][0])
            
            # print('sum', (p_gen * self.sm(dec_dist) + (1-p_gen) * enc_attn_dist).sum(dim=-1)[0][0])

        torch.cuda.synchronize()

        if posmask != None:
            dec_dist *= posmask

        if self.pointer_gen:
            return torch.log(p_gen * self.sm(dec_dist) + (1-p_gen) * enc_attn_dist), p_gen 
        else:
            return self.sm(dec_dist), torch.zeros((1,1))

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


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask)[0])
        _, attn_dist = self.src_attn(x, m, m, src_mask)
        #print(attn_dist.size())
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask)[0])
        #x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward), attn_dist     