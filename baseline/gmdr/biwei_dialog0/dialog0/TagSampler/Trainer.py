from __future__ import division
import time
import os
import sys
import math

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
    def __init__(self, loss=0, n_labels=0, n_correct=0):
        self.loss = loss
        self.start_time = time.time()

    def update(self, stat):
        self.loss += stat.loss



    def elapsed_time(self):
        return time.time() - self.start_time

    def print_out(self, epoch, batch, n_batches):
        t = self.elapsed_time()

        out_info = ("Epoch %2d, %5d/%5d| loss: %6.2f| " + \
               "%4.0f s elapsed") % \
              (epoch, batch, n_batches,
               self.loss/100,
               time.time() - self.start_time)

        print(out_info)
        sys.stdout.flush()

class Trainer(object):
    def __init__(self, opt, model, train_iter,
                 train_loss, optim, lr_scheduler, device):

        self.model = model
        self.train_iter = train_iter
        self.train_loss = train_loss
        self.optim = optim
        self.lr_scheduler = lr_scheduler
        #Jay edit
        self.device = device

        # Set model in training mode.
        self.model.train()

        self.global_step = 0
        self.step_epoch = 0
        self.opt = opt

    def update(self, batch):
        self.model.zero_grad()
        #Jay edit
        src_inputs = batch.src[0].to(self.device)
        src_lengths = batch.src[1].tolist()
        outputs = self.model(src_inputs,src_lengths)

        stats = self.train_loss.monolithic_compute_loss(batch, outputs)
        self.optim.step()
        return stats

    def train(self, epoch, report_func=None):
        """ Called for each epoch to train. """
        total_stats = Statistics()
        report_stats = Statistics()

        for step_batch, batch in enumerate(self.train_iter):
            self.global_step += 1

            stats = self.update(batch)


            report_stats.update(stats)
            total_stats.update(stats)

            if report_func is not None:
                report_stats = report_func(self.global_step,
                        epoch, step_batch, len(self.train_iter),
                        self.optim.lr, report_stats)


        return total_stats

    def save_per_epoch(self, epoch, out_dir):
        f = open(os.path.join(out_dir,'checkpoint'),'w')
        f.write('latest_checkpoint:checkpoint_epoch%d.pkl'%(epoch))
        f.close()
        self.model.drop_checkpoint(epoch,self.opt,
                    os.path.join(out_dir,"checkpoint_epoch%d.pkl"%(epoch)))

    def load_checkpoint(self, cpnt):
        self.model.load_checkpoint(cpnt)
