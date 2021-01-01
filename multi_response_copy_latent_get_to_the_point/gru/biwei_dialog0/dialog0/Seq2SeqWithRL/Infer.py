import time
import os
import sys
import math
import torch
from torch.nn.utils import clip_grad_norm_


class Infer(object):
    def __init__(self, rl_model,
                 test_data, vocab, index_2_latent_sentence,
                 use_cuda=True, device=0):

        self.rl_model = rl_model

        self.test_data = test_data
        self.vocab = vocab
        self.index_2_latent_sentence = index_2_latent_sentence

        # use GPU or not
        self.use_cuda = use_cuda
        self.device = device


    def infer(self, make_data=None, make_2stage_data=None, report_func=None):
        # set model to 'eval mode'
        self.rl_model.eval()

        # make testing data iteration
        self.test_batches = make_data(self.test_data, train_or_valid=False)
        self.test_iter = iter(self.test_batches)
        self.max_test_iter = len(self.test_batches)

        for step_batch, batch in enumerate(self.test_iter):

            print("\nbatch: {}".format(step_batch))

            # assign batch input data
            enc_ids = batch[0][0]
            enc_batch = batch[0][1]
            enc_padding_mask = batch[0][2]
            enc_lens = torch.LongTensor(batch[0][3])

            target_responses = batch[1][0]

            # put data into GPU if use_cuda
            if self.use_cuda:
                enc_batch = enc_batch.to(self.device)
                enc_lens = enc_lens.to(self.device)
                enc_padding_mask = enc_padding_mask.to(self.device)


            # Run Sampler model
            log_probs, _ = self.rl_model.sample_latent_sentence(enc_batch=enc_batch, enc_lens=enc_lens)
            selected_tag_logprob, selected_tag = log_probs.max(dim=-1)
            selected_tag = selected_tag.tolist()
            if type(selected_tag) == list:
                selected_latent_sentences = [self.index_2_latent_sentence[int(id)] for id in selected_tag]
            else:
                selected_latent_sentences = [self.index_2_latent_sentence[int(selected_tag)]]

            ########## combine sampled pos sequence with post ##########
            latent_sentence_batch, target_response_batch, oovs_batch = make_2stage_data(selected_latent_sentences, target_responses, self.vocab)


            latent_sentence_ids = latent_sentence_batch[0]
            latent_sentence_padVar = latent_sentence_batch[1]
            latent_sentence_mask = latent_sentence_batch[2]
            latent_sentence_lens = torch.LongTensor(latent_sentence_batch[3])

            dec_batch = target_response_batch[0].transpose(0,1).contiguous()
            target_batch = target_response_batch[1].transpose(0,1).contiguous()
            dec_padding_mask = target_response_batch[2].transpose(0,1).contiguous()
            target_padVar = target_response_batch[3]
            target_mask = target_response_batch[4].transpose(0,1).contiguous()
            dec_lens_var = torch.LongTensor(target_response_batch[5])

            oov_list = oovs_batch[0]
            max_oov_len = oovs_batch[1]


            # put data into GPU if use_cuda
            if self.use_cuda:
                latent_sentence_padVar = latent_sentence_padVar.to(self.device)
                latent_sentence_mask = latent_sentence_mask.to(self.device)
                latent_sentence_lens = latent_sentence_lens.to(self.device)

                dec_batch = dec_batch.to(self.device)
                dec_padding_mask = dec_padding_mask.to(self.device)
                target_batch = target_batch.to(self.device)
                dec_lens_var = dec_lens_var.to(self.device)


            # Run Seq2Seq Encode phase
            post_encoder_outputs, post_encoder_feature, post_encoder_hidden, tag_encoder_outputs, tag_encoder_feature, tag_encoder_hidden = self.rl_model.seq2seq_model.encode(enc_batch, enc_lens, latent_sentence_padVar, latent_sentence_lens)

            # Run Seq2Seq Decode phase
            beam_search_result = self.rl_model.seq2seq_model.decode(post_encoder_outputs, post_encoder_feature, post_encoder_hidden, enc_padding_mask,
                                                                    tag_encoder_outputs, tag_encoder_feature, tag_encoder_hidden, latent_sentence_mask,
                                                                    dec_batch, dec_padding_mask, dec_lens_var, target_batch,
                                                                    oov_list, max_oov_len, latent_sentence_padVar,
                                                                    train=False, valid=False, test=True)


            # transfer to word-form & show
            src = enc_ids
            pred_latent_sentence = selected_latent_sentences
            tgt = target_batch.tolist()
            pred_response = beam_search_result

            results = []
            for id in range(len(enc_ids)):
                src_words = self.indices2word(src[id], self.vocab, oov_list[id])
                tgt_words = self.indices2word(tgt[id], self.vocab, oov_list[id])
                pred_words = self.indices2word(pred_response[id], self.vocab, oov_list[id])

                results.append([src_words, pred_latent_sentence[id], tgt_words, pred_words])

            report_func(results)



    def indices2word(self, indices, vocab, oov_list):
        vocab_size = vocab.n_words
        out = []
        for index in indices:
            if index >= self.vocab.n_words:
                out.append(oov_list[index-self.vocab.n_words])
            else:
                out.append(vocab.index2word[index])

        return ' '.join(out)


    def save_per_epoch(self, epoch, out_dir):
        f = open(os.path.join(out_dir,'checkpoint'),'w')
        f.write('latest_checkpoint:checkpoint_epoch%d.pkl'%(epoch))
        f.close()
        self.rl_model.drop_checkpoint(epoch, os.path.join(out_dir,"checkpoint_epoch%d.pkl"%(epoch)))


    def load_checkpoint(self, filenmae):
        self.rl_model.load_checkpoint(filenmae)
