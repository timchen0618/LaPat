from __future__ import print_function
import numpy as np
import random
import json
import os
import re
import sys
import torch
from tqdm import tqdm
import operator
import torch.autograd as autograd
from nltk.corpus import stopwords
import time
import codecs
import pickle

def read_json(filename):
    with open(filename, 'r') as fp:
        data = json.load(fp)
    return data


def write_json(filename,data):
    with open(filename, 'w') as fp:
        json.dump(data, fp)


def make_save_dir(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir



def cc(arr):
    return torch.from_numpy(np.array(arr)).cuda()


class bert_utils():
    def __init__(self, config, train, task):
        self.batch_size = config['batch_size']
        self.train = train 
        self.task = task
        self.data_path = config['data_path']
        self.dict_path = config['dict']
        self.test_path = os.path.join(self.data_path, config['test_corpus'])
        self.train_path = os.path.join(self.data_path, config['corpus'])
        self.valid_path = os.path.join(self.data_path, config['valid_corpus'])
        pos_dict_path = os.path.join(self.data_path, config['pos_dict_path'])
        self.src_max_len = config['src_max_len']
        self.max_len = config['max_len']
        self.config = config
        self.pos_dict = pickle.load(open(pos_dict_path, 'rb'))
        # self.pos_tokens = ["<n>", "<nt>", "<nd>", "<nl>", "<nh>", "<nhf>", "<nhs>", "<ns>", "<nn>", "<ni>", "<nz>", "<v>", "<vd>", "<vl>", "<vu>", "<a>", "<f>", "<m>", "<mq>", "<q>", "<d>", "<r>", "<p>", "<c>", "<u>", "<e>", "<o>", "<i>", "<j>", "<h>", "<k>", "<g>", "<x>", "<w>", "<ws>", "<wu>"]

        print('loading pos_dict_path: ',config['pos_dict_path'])
        print('dict size: %d'%(len(self.pos_dict['idx2structure'])))
        print()
        assert os.path.exists(self.dict_path)
        print('Loading Dictionary %s' %self.dict_path)
        self.dictionary = read_json(self.dict_path)
        # else:
            # self.word2id = make_dict(50000, self.dict_path, self.train_path, self.valid_path, self.test_path)
        self.all_tokens = [k for k, v in self.dictionary.items()]
        self.vocab_size = len(self.all_tokens)
        

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.word2id = self.tokenizer.get_vocab()

        self.index2word = [[]]*len(self.word2id)
        for word in self.word2id:
            self.index2word[self.word2id[word]] = word

        self.vocab_size = len(self.word2id)
        print('vocab_size:',self.vocab_size)
        assert len(self.tokenizer) == self.vocab_size
        # self.eos = self.word2id['</s>']
        # self.bos = self.word2id['<s>']
        self.pad = self.word2id['<pad>']
        print('pad: ', self.pad)
        # self.unk = self.word2id['<blank>']

    def text2id(self, text, seq_length=40):
        # vec = np.zeros([seq_length] ,dtype=np.int32)
        vec = np.full([seq_length], self.pad, dtype = np.int32)
        unknown = 0.
        word_list = text.strip().split()
        word_list.reverse()
        length = len(word_list)

        for i,word in enumerate(word_list):
            if i >= seq_length:
                break
            if word in self.word2id:
                vec[-(i+1)] = self.word2id[word]
            else:
                vec[-(i+1)] = self.word2id['<blank>']
                unknown += 1

        # if unknown / length > 0.1 or length > seq_length*1.5:
        #     vec = None
        if self.train:
            if unknown / length > 0.05 or length > seq_length*2:
                vec = None
        return vec

    def tokenize(self, x):
        return self.tokenizer.batch_encode_plus(x, max_length=self.src_max_len,
                                                  truncation_stratgegy='only_first',
                                                  return_tensors='pt',
                                                  pad_to_max_length=True,
                                                  return_special_token_masks=True,
                                                  return_lengths=True,
                                                  return_attention_masks=True)

    def data_yielder(self, valid=False):
        
        if self.train:
            batch = {'src':[],'tgt_mask':[],'y':[]}
            src_file = self.valid_path if valid else self.train_path

            with codecs.open(src_file, encoding="utf8",errors='ignore') as train_f:
                start_time = time.time()
                for line in train_f:
                    data = line.strip().split('\t')                    
                    vec2 = self.text2id(data[2].strip(), self.max_len)
                    text1 = (data[0].strip(), ' '.join(['<' + word + '>' for word in data[3].strip().split()]))

                    if vec2 is not None:
                        batch['src'].append(text1)
                        batch['tgt_mask'].append(self.subsequent_mask(vec2))
                        batch['y'].append(vec2)

                        if len(batch['src']) == self.batch_size:
                            batch = {k: (cc(v) if k != 'src' else v) for k, v in batch.items()}
                            batch['src'] = self.tokenize(batch['src'])
                            torch.cuda.synchronize()
                            yield batch
                            batch = {'src':[],'tgt_mask':[],'y':[]}
                    else:
                        print('ffffff')
                if len(batch['src']) != 0:
                    batch = {k: (cc(v) if k != 'src' else v) for k, v in batch.items()}
                    batch['src'] = self.tokenize(batch['src'])
                    torch.cuda.synchronize()
                    yield batch
                    batch = {'src':[],'tgt_mask':[],'y':[]}
                end_time = time.time()
        else:
            batch = {'src':[]}
            miss = 0
            for epo in range(1):
                start_time = time.time()
                filepath = self.test_path

                print("start epo %d" % (epo))
                print('Reading from testing data %s ... '%filepath)

                for line1 in open(filepath):
                    data = line1.strip().split('\t')
                    text1 = (data[0].strip(), ' '.join(['<' + word + '>' for word in data[3].strip().split()]))

                    if True:
                        batch['src'].append(text1)

                        if len(batch['src']) == self.batch_size:
                            batch['src'] = self.tokenize(batch['src'])
                            yield batch
                            batch = {'src':[]}
                    else:
                        miss += 1
                        print('Wrong Data....', miss)

                if len(batch['src']) != 0:
                    batch['src'] = self.tokenize(batch['src'])
                    yield batch
                    batch = {'src':[], 'src_mask':[]}
                end_time = time.time()


    def id2sent(self, indices, test=False):
        sent = []
        word_dict={}
        for index in indices:
            if test and (index == self.unk or index == self.pad or index in word_dict):
                continue
            sent.append(self.index2word[index])
            word_dict[index] = 1

        return ' '.join(sent)


    def subsequent_mask(self, vec):
        attn_shape = (vec.shape[-1], vec.shape[-1])
        return (np.triu(np.ones((attn_shape)), k=1).astype('uint8') == 0).astype(np.float)

