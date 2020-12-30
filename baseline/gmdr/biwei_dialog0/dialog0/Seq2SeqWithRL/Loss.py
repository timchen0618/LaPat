from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from dialog0.Seq2SeqWithRL.Trainer import Statistics
import dialog0.Seq2SeqWithRL.IO as IO
class RLLossCompute(nn.Module):
    """
    Tag Sapmler Loss Computation.
    """
    def __init__(self, generator, tgt_vocab, avr_seq2seq_loss=False):
        super(RLLossCompute, self).__init__()
        self.generator = generator
        self.padding_idx = tgt_vocab.stoi[IO.PAD_WORD]
        self.avr_seq2seq_loss = avr_seq2seq_loss
    def compute_sampler_reward(self, infer, output, target, length):
        length = Variable(length)
        if output.is_cuda:
            length = length.cuda()
        log_probs = self.generator(output)
        target_flat = target.view(-1,1)
        log_probs_flat = log_probs.view(-1,log_probs.size(-1))
        token_log_probs = torch.gather(log_probs_flat,dim=1,index=target_flat)
        token_log_probs = token_log_probs.view(*target.size())
        # mask
        pad_mask = self._sequence_mask(sequence_length=length, max_len=target.size(1))
        masked_token_log_probs = token_log_probs * pad_mask.float()

        sents_log_probs = masked_token_log_probs.sum(-1)
        sents_probs = sents_log_probs.exp()
        sents_loss = -sents_log_probs/length.float()

        sents_acc = self.get_accuracy(log_probs,target,pad_mask)
        
        sents_rewards = []
        sents_f1_score = []
        infer_words = infer.tolist()
        ground_words = target.data.tolist()

        for b_ground_words,max_len,acc,prob in zip(ground_words,
                                                    length.data.tolist(),
                                                    sents_acc.data.tolist(),
                                                    sents_probs.data.tolist()):
            f1_score = self.get_f1_score(infer_words,b_ground_words[:max_len])
            sent_reward = prob + acc + f1_score
            if sent_reward < 0.3:
                sent_reward = -1.0
            sents_rewards.append(sent_reward)
            sents_f1_score.append(f1_score)

        sents_rewards = Variable(torch.FloatTensor(sents_rewards)).cuda()
        sents_f1_score = Variable(torch.FloatTensor(sents_f1_score)).cuda()
        max_reward,max_id = sents_rewards.max(0)
        final_reward = max_reward
        reward1 = sents_probs[max_id]
        reward2 = sents_acc[max_id]
        reward3 = sents_f1_score[max_id]

        if self.avr_seq2seq_loss:
            selected_sent_loss = sents_loss.mean()
        else:
            selected_sent_loss = sents_loss[max_id]
        return final_reward, selected_sent_loss, reward1, reward2, reward3
    def compute_loss(self, output, target, length, selected_tag_logprob, infer_words):
        final_reward, selected_sent_loss, reward1, reward2, reward3= \
            self.compute_sampler_reward(infer_words,output, target, length)
        seq2seq_loss = selected_sent_loss
        sampler_loss = -selected_tag_logprob * final_reward

        seq2seq_loss_data = seq2seq_loss.data.clone()
        sampler_loss_data = sampler_loss.data.clone()
        final_reward_data = final_reward.data.clone()
        reward1_data = reward1.data.clone()
        reward2_data = reward2.data.clone()
        reward3_data = reward3.data.clone()
        stats = Statistics(seq2seq_loss_data, sampler_loss_data, final_reward_data,
                            reward1_data,reward2_data,reward3_data)
        return stats, seq2seq_loss, sampler_loss

    def _sequence_mask(self, sequence_length, max_len=None):
        if max_len is None:
            max_len = sequence_length.data.max()
        batch_size = sequence_length.size(0)
        seq_range = torch.arange(0, max_len).long()
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
        seq_range_expand = Variable(seq_range_expand)
        if sequence_length.is_cuda:
            seq_range_expand = seq_range_expand.cuda()
        seq_length_expand = (sequence_length.unsqueeze(1)
                            .expand_as(seq_range_expand))
        return seq_range_expand < seq_length_expand

    def get_accuracy(self,output,target,mask):
        batch_size = output.size(0)
        time_step = output.size(1)
        output = self.bottle(output)
        target = target.view(-1)
        pred = output.max(1)[1]
        non_padding = target.ne(self.padding_idx).view(batch_size,time_step)
        num_correct = pred.eq(target)
        num_correct = num_correct.view(batch_size,time_step) * mask

        
        return num_correct.sum(1).float()/non_padding.sum(1).float()
        
    def get_f1_score(self, infer_words, ground_words):
        infer_set = set(infer_words)
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