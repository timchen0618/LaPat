import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from numpy import random
from dialog0.Seq2Seq import config


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



class Encoder_P(nn.Module):
    def __init__(self):
        super(Encoder_P, self).__init__()
        self.encoder_p = nn.GRU(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        init_gru_wt(self.encoder_p)

        self.w_h_o_p = nn.Linear(config.hidden_dim * 2, config.hidden_dim, bias=False)
        self.w_h_h_p = nn.Linear(config.hidden_dim * 2, config.hidden_dim, bias=False)
        self.W_h_f_p = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

    #seq_lens should be in descending order
    def forward(self, input_embed, seq_lens):
        packed = pack_padded_sequence(input_embed, seq_lens, batch_first=True, enforce_sorted = False)
        output, hidden = self.encoder_p(packed)

        encoder_outputs, _ = pad_packed_sequence(output, batch_first=True)  # h dim = B x t_k x n
        encoder_outputs = encoder_outputs.contiguous()
        encoder_outputs = self.w_h_o_p(encoder_outputs)

        hidden = hidden.transpose(0,1).contiguous()
        hidden = hidden.view(hidden.size(0), 2*config.hidden_dim)
        hidden = self.w_h_h_p(hidden)
        hidden = hidden.unsqueeze(0)

        encoder_feature = encoder_outputs.view(-1, config.hidden_dim)  # B * t_k x hidden_dim
        encoder_feature = self.W_h_f_p(encoder_feature)    # B * t_k x hidden_dim

        return encoder_outputs, encoder_feature, hidden



class Encoder_L(nn.Module):
    def __init__(self):
        super(Encoder_L, self).__init__()
        self.encoder_l = nn.GRU(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        init_gru_wt(self.encoder_l)

        self.w_h_o_l = nn.Linear(config.hidden_dim * 2, config.hidden_dim, bias=False)
        self.w_h_h_l = nn.Linear(config.hidden_dim * 2, config.hidden_dim, bias=False)
        self.W_h_f_l = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

    #seq_lens should be in descending order
    def forward(self, input_embed, seq_lens):
        packed = pack_padded_sequence(input_embed, seq_lens, batch_first=True, enforce_sorted = False)
        output, hidden = self.encoder_l(packed)

        encoder_outputs, _ = pad_packed_sequence(output, batch_first=True)  # h dim = B x t_k x n
        encoder_outputs = encoder_outputs.contiguous()
        encoder_outputs = self.w_h_o_l(encoder_outputs)

        hidden = hidden.transpose(0,1).contiguous()
        hidden = hidden.view(hidden.size(0), 2*config.hidden_dim)
        hidden = self.w_h_h_l(hidden)
        hidden = hidden.unsqueeze(0)

        encoder_feature = encoder_outputs.view(-1, config.hidden_dim)  # B * t_k x hidden_dim
        encoder_feature = self.W_h_f_l(encoder_feature)    # B * t_k x hidden_dim

        return encoder_outputs, encoder_feature, hidden



class Attention_P(nn.Module):
    def __init__(self):
        super(Attention_P, self).__init__()
        # attention
        self.decode_proj_p = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.v_p = nn.Linear(config.hidden_dim, 1, bias=False)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, s_t_hat, encoder_outputs, encoder_feature, enc_padding_mask):
        b, t_k, n = list(encoder_outputs.size())

        dec_fea = self.decode_proj_p(s_t_hat)    # B x hidden_dim
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, t_k, n).contiguous()    # B x t_k x hidden_dim
        dec_fea_expanded = dec_fea_expanded.view(-1, n)    # B*t_k x hidden_dim

        attn_features = encoder_feature + dec_fea_expanded    # B*t_k x hidden_dim

        e = self.tanh(attn_features)    # B * t_k x hidden_dim
        scores = self.v_p(e)    # B * t_k x 1
        scores = scores.view(-1, t_k)    # B x t_k

        attn_dist_ = self.softmax(scores)*enc_padding_mask.float()    # B x t_k
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor

        attn_dist = attn_dist.unsqueeze(1)    # B x 1 x t_k
        c_t = torch.bmm(attn_dist, encoder_outputs)    # B x 1 x n
        c_t = c_t.view(-1, config.hidden_dim)    # B x hidden_dim

        attn_dist = attn_dist.view(-1, t_k)    # B x t_k

        return c_t, attn_dist



class Attention_L(nn.Module):
    def __init__(self):
        super(Attention_L, self).__init__()
        # attention
        self.decode_proj_l = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.v_l = nn.Linear(config.hidden_dim, 1, bias=False)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, s_t_hat, encoder_outputs, encoder_feature, enc_padding_mask):
        b, t_k, n = list(encoder_outputs.size())

        dec_fea = self.decode_proj_l(s_t_hat)    # B x hidden_dim
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, t_k, n).contiguous()    # B x t_k x hidden_dim
        dec_fea_expanded = dec_fea_expanded.view(-1, n)    # B*t_k x hidden_dim

        attn_features = encoder_feature + dec_fea_expanded    # B*t_k x hidden_dim

        e = self.tanh(attn_features)    # B * t_k x hidden_dim
        scores = self.v_l(e)    # B * t_k x 1
        scores = scores.view(-1, t_k)    # B x t_k

        attn_dist_ = self.softmax(scores)*enc_padding_mask.float()    # B x t_k
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor

        attn_dist = attn_dist.unsqueeze(1)    # B x 1 x t_k
        c_t = torch.bmm(attn_dist, encoder_outputs)    # B x 1 x n
        c_t = c_t.view(-1, config.hidden_dim)    # B x hidden_dim

        attn_dist = attn_dist.view(-1, t_k)    # B x t_k

        return c_t, attn_dist



class Decoder(nn.Module):
    def __init__(self, vocab_size):
        super(Decoder, self).__init__()
        self.attention_p = Attention_P()
        self.attention_l = Attention_L()

        # decoder
        self.decoder = nn.GRU(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        init_gru_wt(self.decoder)

        # p_vocab
        self.out = nn.Linear(config.hidden_dim * 2, vocab_size, bias=True)
        init_linear_wt(self.out)

        # pointer generator
        if config.pointer_gen:
            self.p_gen_linear = nn.Linear((config.hidden_dim * 3), 1, bias=True)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, y_t_1_embed, s_t_1,
                encoder_outputs_p, encoder_feature_p, enc_padding_mask_p,
                encoder_outputs_l, encoder_feature_l, enc_padding_mask_l,
                extra_zeros, enc_batch_extend_vocab_l):


        # run decoder | o _ o |
        gru_out, s_t = self.decoder(y_t_1_embed.unsqueeze(1), s_t_1)


        # attention mechanism | ~ _ ~ |
        s_t_hat = s_t.view(-1, config.hidden_dim)

        c_t_p, attn_dist_p = self.attention_p(s_t_hat, encoder_outputs_p, encoder_feature_p, enc_padding_mask_p)
        c_t_l, attn_dist_l = self.attention_l(s_t_hat, encoder_outputs_l, encoder_feature_l, enc_padding_mask_l)


        # generation | ^ _ ^ |
        out_input = torch.cat((s_t_hat, c_t_p), 1)
        output = self.out(out_input)
        vocab_dist = self.softmax(output)    # B x vocab_size


        # copy mechanism | Q o Q |
        p_gen = None
        l_copy = None
        if config.pointer_gen:
            probs_input = torch.cat((s_t_hat, c_t_p, c_t_l), 1)    # B x (3*hidden_dim)
            probs = self.p_gen_linear(probs_input)    # B x 1
            probs = self.sigmoid(probs)    # B x 1
            # distribute action
            p_gen = probs    # B x 1
            l_copy = 1 - p_gen    # B x 1


        if config.pointer_gen:
            vocab_dist_ = p_gen * vocab_dist
            attn_dist_l_ = l_copy * attn_dist_l

            if extra_zeros is not None:
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)

            final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab_l, attn_dist_l_)
        else:
            final_dist = vocab_dist


        return final_dist, s_t, p_gen, l_copy



class Model(nn.Module):
    def __init__(self, model_file_path=None, is_eval=False, vocab_size = 50069):
        super(Model, self).__init__()
        encoder_p = Encoder_P()
        encoder_l = Encoder_L()
        decoder = Decoder(vocab_size)

        if is_eval:
            encoder_p = encoder_p.eval()
            encoder_l = encoder_l.eval()
            decoder = decoder.eval()

        self.encoder_p = encoder_p
        self.encoder_l = encoder_l
        self.decoder = decoder

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location= lambda storage, location: storage)
            self.encoder_p.load_state_dict(state['encoder_p_state_dict'])
            self.encoder_l.load_state_dict(state['encoder_l_state_dict'])
            self.decoder.load_state_dict(state['decoder_state_dict'], strict=False)
