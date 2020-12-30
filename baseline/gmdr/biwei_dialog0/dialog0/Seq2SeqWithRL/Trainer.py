from __future__ import division
import time
import os
import sys
import math
from torch.autograd import Variable
import torch
import pdb

def safe_exp(value):
    """Exponentiation with catching of overflow error."""
    try:
        ans = math.exp(value)
    except OverflowError:
        ans = float("inf")
    return ans


class Statistics(object):
    """
    Train/validate loss statistics.
    """
    def __init__(self, seq2seq_loss=0, sampler_loss=0, sampler_reward=0, 
                    reward1=0, reward2=0, reward3=0, n_words=0,
                    infer_words=""):
        self.seq2seq_loss = seq2seq_loss
        self.sampler_loss = sampler_loss
        self.sampler_reward = sampler_reward
        self.reward1 = reward1
        self.reward2 = reward2
        self.reward3 = reward3
        self.n_words = n_words

        self.infer_words = infer_words
        self.start_time = time.time()

    def update(self, stats):
        self.seq2seq_loss += stats.seq2seq_loss
        self.sampler_loss += stats.sampler_loss
        self.sampler_reward += stats.sampler_reward

        self.reward1 += stats.reward1
        self.reward2 += stats.reward2
        self.reward3 += stats.reward3
        self.n_words += stats.n_words

    def ppl(self):
        return self.seq2seq_loss/100

    def get_infer_words(self):
        return " ".join(self.infer_words)

    def elapsed_time(self):
        return time.time() - self.start_time

    def print_out(self, epoch, batch, n_batches, start):
        t = self.elapsed_time()

        out_info = ("Epoch %2d, %5d/%5d| ppl: %6.2f| sampler: %6.6f| " + \
                    "reward: %6.6f|reward1: %1.2f|reward2: %1.2f|reward3: %1.2f|" + \
                    "%4.0f s elapsed") % \
              (epoch, batch, n_batches,
               self.ppl(),
               self.sampler_loss/(100),
               self.sampler_reward/(100),
               self.reward1/(100),
               self.reward2/(100),
               self.reward3/(100),
               time.time() - self.start_time)

        print(out_info)
        sys.stdout.flush()


class Trainer(object):
    def __init__(self, model, train_iter,
                 seq2seq_loss, 
                 sampler_loss, 
                 seq2seq_optim, 
                 sampler_optim, 
                 lr_scheduler):

        self.model = model
        self.train_iter = train_iter
        self.seq2seq_loss = seq2seq_loss
        self.sampler_loss = sampler_loss
        self.seq2seq_optim = seq2seq_optim
        self.sampler_optim = sampler_optim
        self.lr_scheduler = lr_scheduler

        # Set model in training mode.
        self.model.train()

        self.global_step = 0
        self.step_epoch = 0

    def update(self, batch, fields):
        self.seq2seq_optim.optimizer.zero_grad()
        self.sampler_optim.optimizer.zero_grad()
        self.model.zero_grad()

        src_inputs = batch.src[0]
        src_lengths = batch.src[1].tolist()
        tgt_inputs = batch.tgt[0][:-1]
        tgt_lengths = batch.tgt[1]
        tag_inputs = batch.tag
        src_input = src_inputs[:,0].unsqueeze(-1)
        src_length = [src_lengths[0]]
        sampler_output = self.model.sample_tag(src_input,src_length)
        selected_tag, selected_tag_logprob = \
        self.model.sample_tag_with_kmeans((src_inputs,src_lengths,tgt_inputs,tag_inputs),sampler_output)
        
        total_stats = Statistics()

        # torch.autograd.set_detect_anomaly(True)
        for idx,tag in enumerate(selected_tag):
            if 1:
                #self.sampler_optim.optimizer.zero_grad()
                #self.seq2seq_optim.optimizer.zero_grad()

                tag_log_prob = selected_tag_logprob[idx]
                tag = Variable(torch.LongTensor([[tag]])).cuda()
                seq2seq_outputs, decoded_indices, decoded_words = \
                    self.model((src_inputs,src_lengths,tgt_inputs,tag),fields)

                # self.model.greedy_infer()

                stats, seq2seq_loss, sampler_loss = \
                                self.sampler_loss.compute_loss(
                                            seq2seq_outputs.transpose(0,1).contiguous(), 
                                            batch.tgt[0][1:].transpose(0,1).contiguous(), 
                                            tgt_lengths,
                                            tag_log_prob,
                                            decoded_indices)

                total_stats.update(stats)
                if idx != 2:
                    sampler_loss.backward(retain_graph=True)
                    seq2seq_loss.backward(retain_graph=True)
                else:
                    sampler_loss.backward()
                    seq2seq_loss.backward()

        self.sampler_optim.step()
        self.seq2seq_optim.step()
            #except BaseException as e:
            #    print(e)
            #    # pdb.set_trace()
            #    continue
        # print('Yeah finish eeeeeeeeeeeeeeeeeeeeeeeeee')
        total_stats.n_words += sum(tgt_lengths.tolist())
        total_stats.infer_words = decoded_words
        return total_stats

    def train(self, epoch, fields, report_func=None):
        """ Called for each epoch to train. """
        total_stats = Statistics()
        report_stats = Statistics()

        for step_batch, batch in enumerate(self.train_iter):
            
            for k in range(1):
                with torch.autograd.set_detect_anomaly(True):
                    try:
                        self.global_step += 1
                        stats = self.update(batch, fields)
                        report_stats.update(stats)
                        total_stats.update(stats)

                        if report_func is not None:
                            report_stats = report_func(self.global_step,
                                epoch, step_batch, len(self.train_iter),
                                total_stats.start_time, self.sampler_optim.lr, report_stats)
                    except BaseException as e:
                        print(e)
                        continue


        return total_stats


    def save_per_epoch(self, epoch, out_dir):
        f = open(os.path.join(out_dir,'checkpoint'),'w')
        f.write('latest_checkpoint:checkpoint_epoch%d.pkl'%(epoch))
        f.close()
        self.model.drop_checkpoint(epoch,
                    os.path.join(out_dir,"checkpoint_epoch%d.pkl"%(epoch)))

    def load_checkpoint(self, filenmae):
        self.model.load_checkpoint(filenmae)
