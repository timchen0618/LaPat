from __future__ import division
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



class Tester(object):
    def __init__(self, model, test_data, vocab):

        self.model = model

        self.test_data = test_data
        self.vocab = vocab

        # Set model in testing mode.
        self.model.eval()


    def test(self, make_data, report_func=None):
        """ Called for each epoch to train. """

        # testing data iteration
        self.test_batches = make_data(self.test_data, self.vocab.n_words, train_or_valid=False)
        self.test_iter = iter(self.test_batches)

        p_gen_lst = []
        l_copy_lst = []

        for step_batch, batch in enumerate(self.test_iter):
            print('\nbatch: {}'.format(step_batch+1))
            pred, p_gen, l_copy = self.model(batch, train=False, valid=False, test=True)

            p_gen_lst.append(p_gen)
            l_copy_lst.append(l_copy)

            # write out decoder's prediction
            src = batch[0][0].transpose(0, 1).contiguous()
            src = src.tolist()
            # tag = batch[1][0].transpose(0, 1).contiguous()
            # tag = tag.tolist()
            # target = batch[2][1].transpose(0, 1).contiguous()
            # target = target.tolist()
            oovs = batch[3][0]

            # write & save prediction result
            results = []
            for id in range(len(src)):
                # src_words = self.indices2word(src[id], self.vocab.index2word, oovs[id])
                # tag_words = self.indices2word(tag[id], self.vocab.index2word, oovs[id])
                # tgt_words = self.indices2word(target[id], self.vocab.index2word, oovs[id])
                pred_words = self.indices2word(pred[id], self.vocab.index2word, oovs[id])

                results.append(pred_words)

            report_func(results)

        return sum(p_gen_lst)/len(p_gen_lst), sum(l_copy_lst)/len(l_copy_lst)



    def save_per_epoch(self, epoch, out_dir):
        f = open(os.path.join(out_dir,'checkpoint'),'w')
        f.write('latest_checkpoint:checkpoint_epoch%d.pkl'%(epoch))
        f.close()
        self.model.drop_checkpoint(epoch, self.optim, os.path.join(out_dir,"checkpoint_epoch%d.pkl"%(epoch)))


    def load_checkpoint(self, cpnt):
        self.model.load_checkpoint(cpnt)


    def indices2word(self, indices, dict, oovs):
        vocab_size = len(dict)
        out = []
        for index in indices:
            if index >= vocab_size:
                out.append(oovs[index - vocab_size])
            else:
                out.append(dict[index])

        # print(out)
        return ' '.join(out)
