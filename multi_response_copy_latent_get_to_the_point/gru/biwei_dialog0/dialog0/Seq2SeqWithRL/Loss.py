import torch
import torch.nn as nn
from torch.autograd import Variable
from dialog0.Seq2SeqWithRL.Trainer import Statistics


class RLLossCompute(nn.Module):
    """
        RL Loss Computation.
    """
    def __init__(self, vocab, avr_seq2seq_loss=False):
        super(RLLossCompute, self).__init__()

        self.padding_idx = vocab.word2index["<PAD>"]
        self.avr_seq2seq_loss = avr_seq2seq_loss
        self.avg_reward = 0
        self.reward_hist = []
        self.step = 0


    def compute_sampler_reward(self, decoder_output_prob, infer_ids, target_batch, dec_padding_mask, dec_lens_var):

        decoder_output_logprobs = torch.log(decoder_output_prob)

        target_batch_flat = target_batch.view(-1,1)
        decoder_output_logprobs_flat = decoder_output_logprobs.view(-1, decoder_output_logprobs.size(-1))
        # print('decoder_output_logprobs', decoder_output_logprobs.size())                  #[4, 17, 50003]
        # print('decoder_output_logprobs_flat', decoder_output_logprobs_flat.size())        #[68, 50003])
        # print('target_batch', target_batch.size())                        #[4, 17]
        # print('decoder_output_prob', decoder_output_prob.size())                        #[4, 17, 50003]
        token_log_probs = torch.gather(decoder_output_logprobs_flat, dim=1, index=target_batch_flat)
        token_log_probs = token_log_probs.view(*target_batch.size())

        # mask
        # pad_mask = self._sequence_mask(sequence_length=dec_lens_var, max_len=target_batch.size(1))
        masked_token_log_probs = token_log_probs * dec_padding_mask.float()

        # print('masked_token_log_probs ', masked_token_log_probs.size())
        sents_log_probs = masked_token_log_probs.sum(-1)
        sents_probs = sents_log_probs.exp()
        # print(sents_log_probs, '   ', dec_lens_var)
        sents_loss = -sents_log_probs / dec_lens_var.float()

        sents_acc = self.get_accuracy(decoder_output_logprobs, target_batch, dec_padding_mask)

        sents_rewards = []
        sents_f1_score = []
        if torch.numel(infer_ids) == 1:
            infer_ids = [infer_ids.tolist()]
        else:
            infer_ids = infer_ids.tolist()
        # print(infer_ids)
        ground_words = target_batch.data.tolist()
        # print(infer_ids)
        # print(ground_words)

        for b_ground_words, max_len, acc, prob in zip(ground_words, dec_lens_var.data.tolist(), sents_acc.data.tolist(), sents_probs.data.tolist()):
            #問題所在
            f1_score = self.get_f1_score(infer_ids, b_ground_words[:max_len])

            # print('=============================')
            # print('prob', prob)
            # print('accc', acc)
            # print('f1', f1_score)
            sent_reward = prob + acc + f1_score

            sent_reward = sent_reward - self.avg_reward
            # if sent_reward < 0.3:
                # sent_reward = -1.0
            sents_rewards.append(sent_reward)
            sents_f1_score.append(f1_score)

        # self.step += 1
        # if self.step == 100:
        #     self.step = 0
        #     print('==================')
        #     for b_ground_words, max_len in zip(ground_words, dec_lens_var.data.tolist()):
        #         print('infer', infer_ids)
        #         print('gt', b_ground_words[:max_len])
        #         print()

            # print('avg_f1: ', sents_f1_score.mean().item(), ' max_f1: ', reward3.item())


        sents_rewards = Variable(torch.FloatTensor(sents_rewards)).cuda()
        # print('sent_reward ', sents_rewards)
        sents_f1_score = Variable(torch.FloatTensor(sents_f1_score)).cuda()
        max_reward, max_id = sents_rewards.max(0)
        final_reward = max_reward
        reward1 = sents_probs[max_id]
        reward2 = sents_acc[max_id]
        reward3 = sents_f1_score[max_id]
        avg_f1 = sents_f1_score.mean()

        if len(self.reward_hist) < 10:
            self.reward_hist.append(sent_reward)
        else:
            del self.reward_hist[0]
            self.reward_hist.append(sent_reward)
        self.avg_reward = sum(self.reward_hist)/ len(self.reward_hist)
        # print('rrrrrrrrrrr')
        # print(self.avr_seq2seq_loss, sents_loss)
        # print(self.reward_hist)
        # print(self.avg_reward)
        if self.avr_seq2seq_loss:
            selected_sent_loss = sents_loss.mean()
        else:
            selected_sent_loss = sents_loss[max_id]

        # print(selected_sent_loss)
        return final_reward, selected_sent_loss, reward1, reward2, reward3, avg_f1


    def compute_loss(self, selected_tag_logprob, decoder_output_prob, infer_ids, target_batch, dec_padding_mask, dec_lens_var):
        final_reward, selected_sent_loss, reward1, reward2, reward3, avg_f1= self.compute_sampler_reward(decoder_output_prob, infer_ids, target_batch, dec_padding_mask, dec_lens_var)
        seq2seq_loss = selected_sent_loss
        sampler_loss = -selected_tag_logprob * final_reward
        sampler_loss = sampler_loss.mean()
        # print(selected_tag_logprob)
        # print(final_reward)
        # print('sampler_loss', sampler_loss)

        seq2seq_loss_data = seq2seq_loss.data.clone()
        sampler_loss_data = sampler_loss.data.clone()
        final_reward_data = final_reward.data.clone()
        reward1_data = reward1.data.clone()
        reward2_data = reward2.data.clone()
        reward3_data = reward3.data.clone()
        avg_f1_data = avg_f1.data.clone()
        stats = Statistics(sampler_loss_data, seq2seq_loss_data, final_reward_data,
                           reward1_data,reward2_data,reward3_data, avg_f1_data)

        return stats, sampler_loss, seq2seq_loss


    def _sequence_mask(self, sequence_length, max_len=None):
        if max_len is None:
            max_len = sequence_length.data.max()

        batch_size = sequence_length.size(0)
        seq_range = torch.arange(0, max_len).long()
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
        seq_range_expand = Variable(seq_range_expand)
        if sequence_length.is_cuda:
            seq_range_expand = seq_range_expand.cuda()

        seq_length_expand = (sequence_length.unsqueeze(1).expand_as(seq_range_expand))
        return seq_range_expand < seq_length_expand


    def get_accuracy(self, decoder_output_logprob, target_batch, mask):
        batch_size = decoder_output_logprob.size(0)
        time_step = decoder_output_logprob.size(1)
        decoder_output_logprob = self.bottle(decoder_output_logprob)
        target_batch = target_batch.view(-1)
        pred = decoder_output_logprob.max(1)[1]
        non_padding = target_batch.ne(self.padding_idx).view(batch_size, time_step)
        num_correct = pred.eq(target_batch)
        num_correct = num_correct.view(batch_size,time_step) * mask

        return num_correct.sum(1).float() / non_padding.sum(1).float()


    def get_f1_score(self, infer_ids, ground_words):
        infer_set = set(infer_ids)
        ground_set = set(ground_words)

        intersect = ground_set.intersection(infer_set)
        precision = len(intersect)/len(infer_set)
        recall = len(intersect)/len(ground_set)
        if precision == 0.0 or recall == 0.0:
            f1_score = 0.0
        else:
            f1_score = 2.0*(precision*recall) / (precision+recall)

        return f1_score


    def bottle(self, v):
        return v.view(-1, v.size(2))
