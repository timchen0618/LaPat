from math import log
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from dialog0.Seq2Seq import config


class Seq2SeqWithTag(nn.Module):
    def __init__(self, shared_embeddings, model, vocab, vocab_size, device):
        super(Seq2SeqWithTag, self).__init__()

        self.shared_embeddings = shared_embeddings
        self.model = model
        self.vocab = vocab
        self.vocab_size = vocab_size

        self.hidden_size = config.hidden_dim

        self.device = device


    def forward(self, input, train=True, valid=False, test=False):
        # distribute input data
        post_enc_batch = input[0][0].transpose(0, 1).contiguous()
        post_enc_padding_mask = input[0][1].transpose(0, 1).contiguous()
        post_enc_lens = torch.LongTensor(input[0][2])
        post_enc_batch_extend_vocab = input[0][3].transpose(0, 1).contiguous()

        tag_enc_batch = input[1][0].transpose(0, 1).contiguous()
        tag_enc_padding_mask = input[1][1].transpose(0, 1).contiguous()
        tag_enc_lens = torch.LongTensor(input[1][2])
        tag_enc_batch_extend_vocab = input[1][3].transpose(0, 1).contiguous()

        target_padVar = torch.LongTensor(input[2]).transpose(0, 1).contiguous()
        # target_batch = input[2][1].transpose(0, 1).contiguous()
        # dec_padding_mask = input[2][2].transpose(0, 1).contiguous()
        # target_padVar = input[2][3].transpose(0, 1).contiguous()
        # target_mask = input[2][4].transpose(0, 1).contiguous()
        # dec_lens_var = torch.LongTensor(input[2][5])
        # max_dec_len = input[2][6]

        oov_list = input[3][0]
        max_oov_len = input[3][1]

        batch_size = post_enc_batch.size(0)
        extra_zeros = torch.zeros((batch_size, max_oov_len))


        if config.use_cuda:
            post_enc_batch = post_enc_batch.to(self.device)
            post_enc_padding_mask = post_enc_padding_mask.to(self.device)
            post_enc_lens = post_enc_lens.to(self.device)
            post_enc_batch_extend_vocab = post_enc_batch_extend_vocab.to(self.device)

            tag_enc_batch = tag_enc_batch.to(self.device)
            tag_enc_padding_mask = tag_enc_padding_mask.to(self.device)
            tag_enc_lens = tag_enc_lens.to(self.device)
            tag_enc_batch_extend_vocab = tag_enc_batch_extend_vocab.to(self.device)

            # dec_batch = dec_batch.to(self.device)
            # target_batch = target_batch.to(self.device)
            # dec_padding_mask = dec_padding_mask.to(self.device)
            # dec_lens_var = dec_lens_var.to(self.device)

            extra_zeros = extra_zeros.to(self.device)



        # Run post words through encoder
        input_post_embed = self.shared_embeddings(post_enc_batch)
        post_encoder_outputs, post_encoder_feature, post_encoder_hidden = self.model.encoder_p(input_post_embed, post_enc_lens)
        s_t_1 = post_encoder_hidden

        input_latent_embed = self.shared_embeddings(tag_enc_batch)
        tag_encoder_outputs, tag_encoder_feature, tag_encoder_hidden = self.model.encoder_l(input_latent_embed, tag_enc_lens)


        #start training with for loop
        step_losses = []
        final_dist_corpus = []
        p_gen_lst = []
        l_copy_lst = []
        y_t_1 = None
        for di in range(target_padVar.size(1)):
            if di == 0:
                sos = self.vocab.word2index["<SOS>"]
                y_t_1 = torch.tensor([sos])
                y_t_1 = y_t_1.expand(target_padVar.size(0)).to(self.device)


            y_t_1_embed = self.shared_embeddings(y_t_1)

            final_dist, s_t_1, p_gen, l_copy = self.model.decoder(y_t_1_embed, s_t_1,
                                                                  post_encoder_outputs, post_encoder_feature, post_enc_padding_mask,
                                                                  tag_encoder_outputs, tag_encoder_feature, tag_enc_padding_mask,
                                                                  extra_zeros, post_enc_batch_extend_vocab, tag_enc_batch_extend_vocab)

            p_gen_lst.append(p_gen)
            l_copy_lst.append(l_copy)


            # show training phase sentence decoding situation
            step_pred = final_dist.argmax(dim=-1)

            final_dist_corpus.append(final_dist.unsqueeze(1))


            # 清理 gpu memory
            del y_t_1
            torch.cuda.empty_cache()

            y_t_1 = torch.LongTensor(step_pred.size(0))

            for i in range(step_pred.size(0)):
                y_t_1[i] = step_pred[i]

            y_t_1 = y_t_1.to(self.device)


            # target = target_batch[:, di]
            # gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            # step_loss = -torch.log(gold_probs + config.eps)
            #
            # step_mask = dec_padding_mask[:, di]
            # step_loss = step_loss * step_mask.float()
            # step_losses.append(step_loss)


        # sum_losses = torch.sum(torch.stack(step_losses, 1), 1)

        # batch_avg_loss = sum_losses/dec_lens_var.float().to(self.device)
        # loss = torch.mean(batch_avg_loss)

        # 將 beam search 結果 放進 pred
        pred = []
        final_dist_corpus = torch.cat(final_dist_corpus, 1)    # B x t_k x vocab_size
        final_dist_corpus = final_dist_corpus.tolist()
        for counter, final_dist_ in enumerate(final_dist_corpus):
            bs_result = self.beam_search(final_dist_, k=4)
            print('finish beam decode no.{}'.format(counter+1))
            pred.append(bs_result[0][0])


        # 清理 gpu memory
        del post_enc_batch
        del post_enc_padding_mask
        del post_enc_lens
        del post_enc_batch_extend_vocab

        del tag_enc_batch
        del tag_enc_padding_mask
        del tag_enc_lens
        del tag_enc_batch_extend_vocab

        # del dec_batch
        # del target_batch
        # del dec_padding_mask
        # del dec_lens_var

        del extra_zeros

        torch.cuda.empty_cache()


        p_gen_lst = torch.cat(p_gen_lst, dim=-1)
        p_gen_lst = torch.mean(p_gen_lst[0])
        l_copy_lst = torch.cat(l_copy_lst, dim=-1)
        l_copy_lst = torch.mean(l_copy_lst[0])

        return pred, p_gen_lst.item(), l_copy_lst.item()



    def beam_search(self, data, k=4):
        # data has size: tokens x Vocab_size
        sequences = [[list(), 1.0]]
        for row in data:
            all_candidates = list()
            # expand each current candidate
            for i in range(len(sequences)):
                seq, score = sequences[i]
                for j in range(len(row)):
                    candidate = [seq + [j], score * -log((row[j]+1e-12))]
                    all_candidates.append(candidate)
            # order all candidates by score
            ordered = sorted(all_candidates, key=lambda tup:tup[1])
            # select k best
            sequences = ordered[:k]
        return sequences



    def drop_checkpoint(self, epoch, optim, fname):
        torch.save({'shared_embeddings_state_dict': self.shared_embeddings.state_dict(),
                    'encoder_p_state_dict': self.model.encoder_p.state_dict(),
                    'encoder_l_state_dict': self.model.encoder_l.state_dict(),
                    'decoder_state_dict': self.model.decoder.state_dict(),
                    'epoch': epoch},
                    fname)



    def load_checkpoint(self, cpnt):
        cpnt = torch.load(cpnt,map_location=lambda storage, loc: storage)
        self.shared_embeddings.load_state_dict(cpnt['shared_embeddings_state_dict'])
        self.model.encoder_p.load_state_dict(cpnt['encoder_p_state_dict'])
        self.model.encoder_l.load_state_dict(cpnt['encoder_l_state_dict'])
        self.model.decoder.load_state_dict(cpnt['decoder_state_dict'])
        epoch = cpnt['epoch']
        return epoch
