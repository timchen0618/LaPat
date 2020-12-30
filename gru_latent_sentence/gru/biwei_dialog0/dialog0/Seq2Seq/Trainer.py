import time
import os
import sys
import math
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
    def __init__(self, loss=0, batches=0):
        self.loss = loss
        self.batches = batches
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
    def __init__(self, model, train_data, valid_data, vocab,
                 optim, lr_scheduler):

        self.model = model

        self.train_data = train_data
        self.max_train_iter = 0
        self.valid_data = valid_data
        self.max_valid_iter = 0

        self.train_batches = None
        self.valid_batches = None

        self.vocab = vocab

        self.optim = optim
        self.lr_scheduler = lr_scheduler


        # Set model in training mode.
        self.model.train()

        self.global_step = 0
        self.step_epoch = 0

        # model_copy config parameters
        self.max_grad_norm = 5.0


    def train(self, epoch, make_data=None, report_func=None):
        """ Called for each epoch to train. """
        report_train_stats = Statistics(loss=0, batches=0)

        # training data iteration
        self.train_batches = make_data(self.train_data, vocab_size=self.vocab.n_words, train_or_valid=True)
        self.train_iter = iter(self.train_batches)
        self.max_train_iter = len(self.train_batches)

        for step_batch, batch in enumerate(self.train_iter):
            # empty gradient descent info
            self.model.zero_grad()

            # feed batches of data into model
            loss, pred, p_gen, l_copy = self.model(batch, train=True, valid=False, test=False)

            loss.backward()

            #clip_grad_norm - dropout
            self.norm_enc_post = clip_grad_norm_(self.model.model.encoder_p.parameters(), self.max_grad_norm)
            self.norm_enc_latent = clip_grad_norm_(self.model.model.encoder_l.parameters(), self.max_grad_norm)
            self.norm_dec = clip_grad_norm_(self.model.model.decoder.parameters(), self.max_grad_norm)

            self.optim.step()

            report_train_stats.update(loss=loss.item())


            # print out decoder's prediction
            if (report_func is not None) and ((step_batch+1) % 100 == 0):
                oovs = batch[3][0]
                src = batch[0][3].transpose(0, 1).contiguous()[0]
                src = src.tolist()
                tag = batch[1][3].transpose(0, 1).contiguous()[0]
                tag = tag.tolist()
                target = batch[2][1].transpose(0, 1).contiguous()[0]
                target = target.tolist()

                report_func(epoch, step_batch+1, self.max_train_iter, report_train_stats, (self.indices2word(src, self.vocab.index2word, oovs[0]),
                                                                                           self.indices2word(tag, self.vocab.index2word, oovs[0]),
                                                                                           self.indices2word(target, self.vocab.index2word, oovs[0]),
                                                                                           self.indices2word(pred[0], self.vocab.index2word, oovs[0])),
                                                                                          (p_gen, l_copy))



    def valid(self, epoch, make_data=None, report_func=None):
        report_valid_stats = Statistics(loss=0, batches=0)

        # validation data iteration
        self.valid_batches = make_data(self.valid_data, vocab_size=self.vocab.n_words, train_or_valid=True)
        self.valid_iter = iter(self.valid_batches)
        self.max_valid_iter = len(self.valid_batches)

        avg_loss = None
        for step_batch, batch in enumerate(self.valid_iter):

            # feed batches of data into model
            loss, pred, p_gen, l_copy = self.model(batch, train=False, valid=True, test=False)

            report_valid_stats.update(loss=loss.item())


            # print out decoder's prediction
            if (report_func is not None) and ((step_batch+1) % 100 == 0):
                oovs = batch[3][0]
                src = batch[0][3].transpose(0, 1).contiguous()[0]
                src = src.tolist()
                tag = batch[1][3].transpose(0, 1).contiguous()[0]
                tag = tag.tolist()
                target = batch[2][1].transpose(0, 1).contiguous()[0]
                target = target.tolist()

                avg_loss = report_func(epoch, report_valid_stats, (self.indices2word(src, self.vocab.index2word, oovs[0]),
                                                                   self.indices2word(tag, self.vocab.index2word, oovs[0]),
                                                                   self.indices2word(target, self.vocab.index2word, oovs[0]),
                                                                   self.indices2word(pred[0], self.vocab.index2word, oovs[0])),
                                                                  (p_gen, l_copy))

        return avg_loss


    def save_per_epoch(self, epoch, out_dir):
        f = open(os.path.join(out_dir,'checkpoint'),'w')
        f.write('latest_checkpoint:checkpoint_epoch%d.pkl'%(epoch))
        f.close()
        self.model.drop_checkpoint(epoch, self.optim, os.path.join(out_dir,"checkpoint_epoch%d.pkl"%(epoch)))


    def load_checkpoint(self, cpnt):
        self.model.load_checkpoint(cpnt)


    def indices2word(self, indices, index2word, oovs):
        vocab_size = len(index2word)
        out = []
        for index in indices:
            if index >= vocab_size:
                out.append(oovs[index - vocab_size])
            else:
                out.append(index2word[index])

        return ' '.join(out)
