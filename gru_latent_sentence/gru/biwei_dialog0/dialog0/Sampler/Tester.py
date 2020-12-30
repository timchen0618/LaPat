import time
import os
import sys
import math
import numpy as np
import torch
from torch.autograd import Variable


class Tester(object):
    def __init__(self, model, test_data, vocab, index_2_latent_sentence,
                 use_cuda=True, device=0):

        self.model = model

        self.test_data = test_data

        self.test_batches = None

        self.vocab = vocab
        self.index_2_latent_sentence = index_2_latent_sentence

        self.use_cuda = use_cuda
        self.device = device

    def test(self, make_data, report_func=None):
        """ Called for 1 epoch to test. """

        # Set model in training mode.
        self.model.eval()
        self.model.model.eval()

        # training data iteration
        self.test_batches = make_data(self.test_data, train_or_valid=False)
        self.test_iter = iter(self.test_batches)

        for step_batch, batch in enumerate(self.test_iter):
            #empty gpu cache
            torch.cuda.empty_cache()

            # distribute batch data
            src_ids = batch[0]
            src_padVar = batch[1]
            src_lens = batch[2]


            # put data into GPU if use_cuda
            if self.use_cuda:
                src_padVar = src_padVar.to(self.device)
                src_lens = src_lens.to(self.device)


            # feed batches of data into model
            log_probs, selected_latent_sentence_id = self.model.forward(src_padVar, src_lens)

            # show input output instances
            src_instances = src_ids
            pred_instances = selected_latent_sentence_id.tolist()

            results = []
            for id in range(len(src_instances)):
                src_words = self.indices2word(src_instances[id], self.vocab)
                pred_latent_sentence_instance = self.index_2_latent_sentence[pred_instances[id]]

                results.append((src_words, pred_latent_sentence_instance))

            report_func(results)

    def load_checkpoint(self, cpnt):
        self.model.load_checkpoint(cpnt)


    def indices2word(self, indices, vocab):
        vocab_size = vocab.n_words
        out = []
        for index in indices:
            out.append(vocab.index2word[index])

        return ' '.join(out)
