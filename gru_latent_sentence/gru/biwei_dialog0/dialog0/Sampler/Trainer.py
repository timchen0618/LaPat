import time
import os
import sys
import math
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_


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
    def __init__(self, loss=0):
        self.loss = loss

        self.batches = 0
        self.avg_loss = 0
        self.start_time = time.time()

    def update(self, loss):
        self.loss += loss
        self.batches += 1

    def elapsed_time(self):
        return time.time() - self.start_time

    def print_out(self, state, step_batch, step_batches, epoch):
        t = self.elapsed_time()
        self.avg_loss = self.loss / self.batches

        out_info = (" Epoch %2d | step: %10d | steps: %10d | loss: %6.3f | %4.0f s elapsed\n") % (epoch, step_batch, step_batches, self.avg_loss, t)
        print(state + out_info)

        self.start_time = time.time()
        sys.stdout.flush()

        return self.avg_loss



class Trainer(object):
    def __init__(self, model, train_data, valid_data, vocab, index_2_latent_sentence,
                 optim, lr_scheduler, use_cuda=True, device=0):

        self.model = model

        self.train_data = train_data
        self.max_train_iter = 0
        self.valid_data = valid_data
        self.max_valid_iter = 0

        self.train_batches = None
        self.valid_batches = None

        self.vocab = vocab
        self.index_2_latent_sentence = index_2_latent_sentence

        self.optim = optim
        self.lr_scheduler = lr_scheduler

        self.use_cuda = use_cuda
        self.device = device

        # model config parameters
        self.max_grad_norm = 5.0


    def train(self, epoch, make_data, report_func=None):
        """ Called for each epoch to train. """

        # Set model in training mode.
        self.model.model.train()

        # training data iteration
        self.train_batches = make_data(self.train_data, train_or_valid=True)
        self.train_iter = iter(self.train_batches)
        self.max_train_iter = len(self.train_batches)

        total_loss = 0
        total_correct = 0
        total_instance = 0

        global_step = 0
        global_time = time.time()
        start_time = time.time()
        for step_batch, batch in enumerate(self.train_iter):
            # empty gradient descent info
            self.model.zero_grad()
            self.model.model.zero_grad()
            torch.cuda.empty_cache()


            # distribute batch data
            src_ids = batch[0][0]
            src_padVar = batch[0][1]
            src_lens = batch[0][2]

            tgt_ids = batch[1]


            # put data into GPU if use_cuda
            if self.use_cuda:
                src_padVar = src_padVar.to(self.device)
                src_lens = src_lens.to(self.device)


            # feed batches of data into model
            log_probs, selected_latent_sentence_id = self.model.forward(src_padVar, src_lens)
            avg_loss = self.model.compute_loss(log_probs, tgt_ids)

            avg_loss.backward()

            #clip_grad_norm - dropout
            self.norm_enc = clip_grad_norm_(self.model.model.encoder.parameters(), self.max_grad_norm)

            # iterate learning rate
            self.optim.step()

            # statistics
            # calculate loss
            total_loss += avg_loss.item()
            global_step += 1

            # calculate accuracy
            pred_latent_sentence = selected_latent_sentence_id.tolist()
            for pred_latent_sentence_idx, tgt_latent_sentence_idx in zip(pred_latent_sentence, tgt_ids):
                correct_bool = False
                for tgt_idx in tgt_latent_sentence_idx:
                    if str(pred_latent_sentence_idx) == str(tgt_idx):
                        correct_bool = True
                        break

                if correct_bool:
                    total_correct += 1
                total_instance += 1

        # show training info.
        elapsed_time = time.time() - start_time

        # show input output instances
        src_instance = src_ids[0]
        pred_instance = selected_latent_sentence_id.tolist()[0]

        src_words = self.indices2word(src_instance, self.vocab)
        pred_latent_sentence_instance = self.index_2_latent_sentence[pred_instance]

        report_func(epoch=epoch, avg_loss=avg_loss.item(), pred_acc=total_correct/total_instance, zoom_in=(src_words, pred_latent_sentence_instance))
        print("Train Epoch: %d  Loss: %f  Acc: %f  Elapsed Time: %f\n" %(epoch, avg_loss.item(), total_correct/total_instance, elapsed_time))

        start_time = time.time()

        return total_loss/global_step, total_correct/total_instance, time.time()-global_time


    def valid(self, epoch, make_data, report_func=None):
        """ Called for each epoch to valid. """

        # Set model in training mode.
        self.model.model.eval()

        # training data iteration
        self.valid_batches = make_data(self.valid_data, train_or_valid=True)
        self.valid_iter = iter(self.valid_batches)
        self.max_valid_iter = len(self.valid_batches)

        total_loss = 0
        total_correct = 0
        total_instance = 0

        global_step = 0
        global_time = time.time()
        for step_batch, batch in enumerate(self.valid_iter):
            #empty gpu cache
            torch.cuda.empty_cache()


            # distribute batch data
            src_ids = batch[0][0]
            src_padVar = batch[0][1]
            src_lens = batch[0][2]

            tgt_ids = batch[1]


            # put data into GPU if use_cuda
            if self.use_cuda:
                src_padVar = src_padVar.to(self.device)
                src_lens = src_lens.to(self.device)


            # feed batches of data into model
            log_probs, selected_latent_sentence_id = self.model.forward(src_padVar, src_lens)
            avg_loss = self.model.compute_loss(log_probs, tgt_ids)


            # statistics

            # calculate loss
            total_loss += avg_loss.item()
            global_step += 1

            # calculate accuracy
            pred_latent_sentence = selected_latent_sentence_id.tolist()
            for pred_latent_sentence_idx, tgt_latent_sentence_idx in zip(pred_latent_sentence, tgt_ids):
                correct_bool = False
                for tgt_idx in tgt_latent_sentence_idx:
                    if str(pred_latent_sentence_idx) == str(tgt_idx):
                        correct_bool = True
                        break

                if correct_bool:
                    total_correct += 1
                total_instance += 1

        # show dev. info.
        elapsed_time = time.time() - global_time

        # show input output instances
        src_instance = src_ids[0]
        pred_instance = selected_latent_sentence_id.tolist()[0]

        src_words = self.indices2word(src_instance, self.vocab)
        pred_latent_sentence_instance = self.index_2_latent_sentence[pred_instance]

        avg_loss = total_loss/global_step
        pred_acc = total_correct/total_instance

        report_func(epoch=epoch, avg_loss=avg_loss, pred_acc=pred_acc, zoom_in=(src_words, pred_latent_sentence_instance))
        print("Valid Epoch: %d  Loss: %f  Acc: %f  Elapsed Time: %f\n" %(epoch, avg_loss, pred_acc, elapsed_time))

        return avg_loss, pred_acc, elapsed_time


    def save_per_epoch(self, epoch, out_dir):
        f = open(os.path.join(out_dir,'checkpoint'),'w')
        f.write('latest_checkpoint:checkpoint_epoch%d.pkl'%(epoch))
        f.close()
        self.model.drop_checkpoint(epoch, os.path.join(out_dir,"checkpoint_epoch%d.pkl"%(epoch)))


    def load_checkpoint(self, cpnt):
        self.model.load_checkpoint(cpnt)


    def indices2word(self, indices, vocab):
        vocab_size = vocab.n_words
        out = []
        for index in indices:
            out.append(vocab.index2word[index])

        return ' '.join(out)
