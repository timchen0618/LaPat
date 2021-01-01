import time
import os
import sys
import math
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_


class Statistics(object):
    """
    Train/validate loss statistics.
    """
    def __init__(self, sampler_loss=0, seq2seq_loss=0, rl_reward=0,
                 reward1=0, reward2=0, reward3=0, avg_f1=0):

        self.sampler_loss = sampler_loss
        self.seq2seq_loss = seq2seq_loss
        self.rl_reward = rl_reward
        self.reward1 = reward1
        self.reward2 = reward2
        self.reward3 = reward3
        self.avg_f1 = avg_f1

        self.start_time = time.time()

        self.batches = 0

    def update(self, stats, batches):
        self.sampler_loss += stats.sampler_loss
        self.seq2seq_loss += stats.seq2seq_loss
        self.rl_reward += stats.rl_reward

        self.reward1 += stats.reward1
        self.reward2 += stats.reward2
        self.reward3 += stats.reward3
        self.avg_f1 += stats.avg_f1

        self.batches += batches

    def ppl(self):
        return self.seq2seq_loss/self.batches

    def print_out(self, epoch, step_batch=None, step_batches=None):
        end_time = time.time()

        out_info = "Epoch" + str(epoch) + ", " + str(step_batch) + "/" + str(step_batches) + "| ppl:" \
                + str(self.ppl().item())[:6]  + "| sampler: %6.6f| " %(self.sampler_loss.item()/(self.batches))  \
                + "reward :" + str(self.rl_reward.item()/(self.batches))[:6] + "| " \
                + "reward1: %6.6f| " %(self.reward1.item()/self.batches) \
                + "reward2: " + str(self.reward2.item()/(self.batches))[:6]  + "| " \
                + "reward3: " + str(self.reward3.item()/(self.batches))[:6]  + "| " \
                + "avg_f1: " + str(self.avg_f1.item()/(self.batches))[:6]  + "| " \
                + str(end_time - self.start_time)[:5]  + "s elapsed\n"  \

        print(out_info)
        print()
        sys.stdout.flush()



class Trainer(object):
    def __init__(self, rl_model,
                 train_data, valid_data, vocab, index_2_latent_sentence,
                 rl_loss, sampler_optim, seq2seq_optim, lr_scheduler,
                 use_cuda=True, device=0, save_model_path=None):

        self.rl_model = rl_model

        self.train_data = train_data
        self.valid_data = valid_data
        self.vocab = vocab
        self.index_2_latent_sentence = index_2_latent_sentence

        self.rl_loss = rl_loss
        self.sampler_optim = sampler_optim
        self.seq2seq_optim = seq2seq_optim
        self.lr_scheduler = lr_scheduler

        # use GPU or not
        self.use_cuda = use_cuda
        self.device = device
        self.save_model_path = save_model_path

        # steps
        self.global_step = 0

    def train(self, epoch, make_data=None, make_2stage_data=None, report_func=None):
        # set model to 'train mode'
        self.rl_model.train()

        # report training phase info
        train_stats_report = Statistics()

        # make training data iteration
        self.train_batches = make_data(self.train_data, train_or_valid=True)
        self.train_iter = iter(self.train_batches)
        self.max_train_iter = len(self.train_batches)

        for step_batch, batch in enumerate(self.train_iter):
            # empty gradient descent info
            self.rl_model.zero_grad()

            # step record
            self.global_step += 1

            # assign batch input data
            enc_ids = batch[0][0]
            enc_batch = batch[0][1]
            enc_padding_mask = batch[0][2]
            enc_lens = torch.LongTensor(batch[0][3])

            target_ls_ids = batch[1][0]

            target_responses = batch[2][0]

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

            # decide whether copy or not
            copy_p = np.random.normal(0,1,1)[0]
            copy_decide = True
            if copy_p > 0.4:
                copy_decide = True
            else:
                copy_decide = False

            # Run Seq2Seq Decode phase
            loss, pred, prob, p_gen_value, l_copy_value = self.rl_model.seq2seq_model.decode(post_encoder_outputs, post_encoder_feature, post_encoder_hidden, enc_padding_mask,
                                                                                             tag_encoder_outputs, tag_encoder_feature, tag_encoder_hidden, latent_sentence_mask,
                                                                                             dec_batch, dec_padding_mask, dec_lens_var, target_batch,
                                                                                             oov_list, max_oov_len, latent_sentence_padVar,
                                                                                             train=True, valid=False, test=False, copy_decide=copy_decide)

            # Run Seq2Seq Greedy Decode phase
            pred, decoded_words = self.rl_model.seq2seq_model.greedy_decode(post_encoder_outputs, post_encoder_feature, post_encoder_hidden, enc_padding_mask,
                                                                            tag_encoder_outputs, tag_encoder_feature, tag_encoder_hidden, latent_sentence_mask,
                                                                            dec_batch,
                                                                            oov_list, max_oov_len, latent_sentence_padVar,
                                                                            train=True, valid=False, test=False, copy_decide=copy_decide)

            # check size and compute Loss & Reward
            if (prob.size(0) == target_batch.size(0)) and (prob.size(1) == target_batch.size(1)):
                stats, sampler_loss, seq2seq_loss = self.rl_loss.compute_loss(selected_tag_logprob, prob, pred, target_batch, dec_padding_mask, dec_lens_var)
            else:
                print('Error arises during computing loss and reward...')

            # upgrade RL model
            self.sampler_optim.optimizer.zero_grad()
            sampler_loss.backward(retain_graph=True)
            self.sampler_optim.step()

            self.seq2seq_optim.optimizer.zero_grad()
            seq2seq_loss.backward(retain_graph=True)
            self.seq2seq_optim.step()


            # transfer to word-form & show
            src = enc_ids[0]
            pred_latent_sentence = selected_latent_sentences[0]
            tgt = target_batch.tolist()[0]
            pred_response = ' '.join(decoded_words)

            src_words = self.indices2word(src, self.vocab)
            tgt_words = self.indices2word(tgt, self.vocab)


            # update Statistics info.
            train_stats_report.update(stats, len(enc_ids))
            if (step_batch+1) % 100 == 0:
                report_func(epoch, step_batch+1, self.max_train_iter, train_stats_report, zoom_in=(src_words, pred_latent_sentence, tgt_words, pred_response))

            if (step_batch+1) % 1000 == 0:
                self.save_per_1000steps(str(epoch) + str(step_batch), self.save_model_path)


    def valid(self, epoch, make_data=None, make_2stage_data=None, report_func=None):
        # set model to 'eval mode'
        self.rl_model.eval()

        # report training phase info
        valid_stats_report = Statistics()

        # make training data iteration
        self.valid_batches = make_data(self.valid_data, train_or_valid=True)
        self.valid_iter = iter(self.valid_batches)
        self.max_valid_iter = len(self.valid_batches)

        for step_batch, batch in enumerate(self.valid_iter):
            # empty gradient descent info
            self.rl_model.zero_grad()

            # step record
            self.global_step += 1

            # assign batch input data
            enc_ids = batch[0][0]
            enc_batch = batch[0][1]
            enc_padding_mask = batch[0][2]
            enc_lens = torch.LongTensor(batch[0][3])

            target_ls_ids = batch[1][0]

            target_responses = batch[2][0]

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
            loss, pred, prob, p_gen_value, l_copy_value = self.rl_model.seq2seq_model.decode(post_encoder_outputs, post_encoder_feature, post_encoder_hidden, enc_padding_mask,
                                                                                             tag_encoder_outputs, tag_encoder_feature, tag_encoder_hidden, latent_sentence_mask,
                                                                                             dec_batch, dec_padding_mask, dec_lens_var, target_batch,
                                                                                             oov_list, max_oov_len, latent_sentence_padVar,
                                                                                             train=True, valid=False, test=False)

            # Run Seq2Seq Greedy Decode phase

            pred, decoded_words = self.rl_model.seq2seq_model.greedy_decode(post_encoder_outputs, post_encoder_feature, post_encoder_hidden, enc_padding_mask,
                                                                            tag_encoder_outputs, tag_encoder_feature, tag_encoder_hidden, latent_sentence_mask,
                                                                            dec_batch,
                                                                            oov_list, max_oov_len, latent_sentence_padVar,
                                                                            train=True, valid=False, test=False)

            # check size and compute Loss & Reward
            if (prob.size(0) == target_batch.size(0)) and (prob.size(1) == target_batch.size(1)):
                stats, sampler_loss, seq2seq_loss = self.rl_loss.compute_loss(selected_tag_logprob, prob, pred, target_batch, dec_padding_mask, dec_lens_var)
            else:
                print('Error arises during computing loss and reward...')


            # transfer to word-form & show
            src = enc_ids[0]
            pred_latent_sentence = selected_latent_sentences[0]
            tgt = target_batch.tolist()[0]
            pred_response = ' '.join(decoded_words)

            src_words = self.indices2word(src, self.vocab)
            tgt_words = self.indices2word(tgt, self.vocab)


            # update Statistics info.
            valid_stats_report.update(stats, len(enc_ids))
            if (step_batch+1) == self.max_valid_iter:
                report_func(epoch, valid_stats_report, zoom_in=(src_words, pred_latent_sentence, tgt_words, pred_response))


    def indices2word(self, indices, vocab):
        vocab_size = vocab.n_words
        out = []
        for index in indices:
            out.append(vocab.index2word[index])

        return ' '.join(out)


    def save_per_epoch(self, epoch, out_dir):
        f = open(os.path.join(out_dir,'checkpoint'),'w')
        f.write('latest_checkpoint:checkpoint_epoch%d.pkl'%(epoch))
        f.close()
        self.rl_model.drop_checkpoint(epoch, os.path.join(out_dir,"checkpoint_epoch%d.pkl"%(epoch)))


    def save_per_1000steps(self, epoch_steps, out_dir):
        f = open(os.path.join(out_dir,'checkpoint'),'w')
        f.write('latest_checkpoint:checkpoint_epoch%s.pkl'%(epoch_steps))
        f.close()
        self.rl_model.drop_checkpoint(epoch_steps, os.path.join(out_dir,"checkpoint_epoch%s.pkl"%(epoch_steps)))


    def load_checkpoint(self, filenmae):
        self.rl_model.load_checkpoint(filenmae)
