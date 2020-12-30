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


def one_hot(indices, depth):
    shape = list(indices.size())+[depth]
    indices_dim = len(indices.size())
    a = torch.zeros(shape,dtype=torch.float).cuda()
    torch.cuda.synchronize()
    return a.scatter_(indices_dim,indices.unsqueeze(indices_dim),1)


def tens2np(tensor):
    return tensor.detach().cpu().numpy()


def make_dict(max_num, dict_path, train_path, valid_path, test_path):
    #create dict with voc length of max_num
    word_count = dict()
    word2id = dict()
    line_count = 0
    
    # for line in tqdm(open(train_path)):
    #     line_count += 1.0
    #     for word in line.split():
    #         word_count[word] = word_count.get(word,0) + 1
    data = open(train_path)
    data2 = open(valid_path)
    testdata = open(test_path)
     
    for line in tqdm(data):
        line_count += 1.0
        data = line.strip('\n').split('\t')
        for word in data[0].strip().split():
            word_count[word] = word_count.get(word,0) + 1
        for word in data[2].strip().split():
            word_count[word] = word_count.get(word,0) + 1
        # for word in line:
        #     word_count[word] = word_count.get(word,0) + 1
    for line in tqdm(data2):
        line_count += 1.0
        # for word in line:
        #     word_count[word] = word_count.get(word,0) + 1
        data = line.strip('\n').split('\t')
        for word in data[0].strip().split():
            word_count[word] = word_count.get(word,0) + 1
        for word in data[2].strip().split():
            word_count[word] = word_count.get(word,0) + 1
    for line in tqdm(testdata):
        line_count += 1.0
        # for word in line:
        #     word_count[word] = word_count.get(word,0) + 1
        data = line.strip('\n').split('\t')
        for word in data[0].strip().split():
            word_count[word] = word_count.get(word,0) + 1
        for word in data[2].strip().split():
            word_count[word] = word_count.get(word,0) + 1
    word2id['</s>'] = len(word2id)
    word2id['<s>'] = len(word2id)
    word2id['<blank>'] = len(word2id)

    ###### for POS ######
    postags_complete_list = ['<n>', '<nt>', '<nd>', '<nl>', '<nh>', '<nhf>', '<nhs>', '<ns>', '<nn>', '<ni>', '<nz>', '<v>', '<vd>', '<vl>', '<vu>', '<a>', '<f>', '<m>', '<mq>', '<q>', '<d>', '<r>', '<p>', '<c>', '<u>', '<e>', '<o>', '<i>', '<j>', '<h>', '<k>', '<g>', '<x>', '<w>', '<ws>', '<wu>']
    for x in postags_complete_list:
        word2id[x] = len(word2id)

    word_count_list = sorted(word_count.items(), key=operator.itemgetter(1))
    print(len(word_count_list))
    # print(word_count_list[-max_num])
    
    for item in word_count_list[-(max_num*2):][::-1]:
        if item[1] < word_count_list[-max_num][1]:
            continue
        word = item[0]
        word2id[word] = len(word2id)

    with open(dict_path,'w') as fp:
        json.dump(word2id, fp)

    return word2id


class data_utils():
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.train = True if args.train else False
        self.dict_path = './5w_pos.json'
        self.test_path = args.test_corpus
        self.train_path = args.corpus
        self.valid_path = args.valid_corpus
        self.src_max_len = args.src_max_len
        self.max_len = args.max_len
        self.pos_dict = pickle.load(open(args.pos_dict_path, 'rb'))

        if not self.train:
            assert (os.path.exists(self.dict_path))
        
        if os.path.exists(self.dict_path):
            self.word2id = read_json(self.dict_path)
            i = 0 
        else:
            self.word2id = make_dict(50000, self.dict_path, self.train_path, self.valid_path, self.test_path)

        self.index2word = [[]]*len(self.word2id)
        for word in self.word2id:
            self.index2word[self.word2id[word]] = word

        self.vocab_size = len(self.word2id)
        # print('vocab_size:',self.vocab_size)
        self.eos = self.word2id['</s>']
        self.bos = self.word2id['<s>']
        self.pad = self.word2id['<pad>']
        # self.pad = self.eos
        self.unk = self.word2id['<blank>']

        if args.pos_masking:
            self.pos2mask = read_json(args.posmask)
            for k, v in self.pos2mask.items():
                self.pos2mask[k] = cc(v)


    def text2id(self, text, seq_length=40, train=False):
        # vec = np.zeros([seq_length] ,dtype=np.int32)
        vec = np.full([seq_length], self.pad, dtype = np.int32)
        unknown = 0.
        word_list = text.strip().split()
        length = len(word_list)

        for i,word in enumerate(word_list):
            if i >= seq_length:
                break
            if word in self.word2id:
                vec[i] = self.word2id[word]
            else:
                #vec[i] = self.word2id['__UNK__']
                vec[i] = self.word2id['<blank>']
                unknown += 1

        # if unknown / length > 0.1 or length > seq_length*1.5:
        #     vec = None
        if train:
            if unknown / length > 0.1 or length > seq_length*1.5:
                vec = None

        return vec

    def addpos2text(self, pos_list, vec1, max_length):
        seq_len = np.sum(vec1 != self.pad)
        # print(seq_len)
        # print(len(pos_list))
        # print(max_length)
        if seq_len + len(pos_list) >= max_length:
            idx = -1
            pos_list.reverse()
            for pos in pos_list:
                # print('pos', pos)
                # print(self.word2id[pos])
                vec1[idx] = self.word2id[pos]
                # print(self.word2id[pos])
                # vec1[idx] = pos
                idx -= 1
                if (-idx) == max_length:
                    break
        else:
            idx = seq_len

            for pos in pos_list:
                # print('pos', pos)
                # print(self.word2id[pos])
                # print(idx)
                # print(vec1[idx])
                vec1[idx] = self.word2id[pos]

                # vec1[idx] = pos
                idx += 1

        # for i,word in enumerate(pos_list):
        #     if i >= seq_length:
        #         break
        #     vec[i] = self.word2id[word]

        return vec1
            # if word in self.word2id:
            # else:
            #     #vec[i] = self.word2id['__UNK__']
            #     vec[i] = self.word2id['<blank>']
            #     unknown += 1

    def addpos2src(self, pos_list, src, max_length):
        seq_len = torch.sum(src != self.pad)
        if seq_len + len(pos_list) >= max_length:
            idx = -1
            pos_list.reverse()
            for pos in pos_list:
                src[idx] = self.word2id[pos]
                idx -= 1
                if (-idx) == max_length:
                    break
        else:
            idx = seq_len
            for pos in pos_list:
                src[idx] = self.word2id[pos]
                idx += 1

        return src

    def data_yielder(self, valid=False):
        if self.train:
            batch = {'src':[],'tgt':[],'src_mask':[],'tgt_mask':[],'y':[], 'sampler_label':[]}
            # with codecs.open(self.train_path, encoding="utf8",errors='ignore') as train_f:
            src_file = self.valid_path if valid else self.train_path

            with codecs.open(src_file, encoding="utf8",errors='ignore') as train_f:
                # for epo in range(num_epoch):
                start_time = time.time()
                # print("start epo %d" % (epo))
                prev_sent = ""
                for line in train_f:
                    data = line.strip().split('\t')

                    if data[0] != prev_sent or len(batch['src']) >= self.batch_size:
                        if len(batch['src']) != 0:
                            batch = {k: cc(v) for k, v in batch.items()}
                            torch.cuda.synchronize()
                            # print('prev_sent', prev_sent)
                            prev_sent = data[0]
                            yield batch
                            batch = {'src':[],'tgt':[],'src_mask':[],'tgt_mask':[],'y':[], 'sampler_label':[]}
                        else:
                            prev_sent = data[0]
                        
                    vec1 = self.text2id(data[0].strip(), self.src_max_len, not valid)
                    vec2 = self.text2id(data[2].strip()+' </s>', self.max_len, not valid)

                    if vec1 is not None and vec2 is not None:
                        batch['src'].append(vec1)
                        batch['src_mask'].append(np.expand_dims(vec1 != self.pad, -2).astype(np.float))
                        batch['tgt'].append(np.concatenate([[self.bos],vec2], axis=0)[:-1])
                        batch['tgt_mask'].append(self.subsequent_mask(vec2))
                        batch['y'].append(vec2)
                        batch['sampler_label'].append(int(data[5]))

                        
                if len(batch['src']) != 0:
                    batch = {k: cc(v) for k, v in batch.items()}
                    torch.cuda.synchronize()
                    yield batch
                    batch = {'src':[],'tgt':[],'src_mask':[],'tgt_mask':[],'y':[], 'sampler_label':[]}
                end_time = time.time()
                    # print('finish epo %d, time %f' % (epo,end_time-start_time))

        else:
            # batch = {'src':[], 'src_mask':[], 'sampler_label':[], 'tgt':[], 'tgt_mask':[], 'y':[]}
            batch = {'src':[], 'src_mask':[], 'sampler_label':[]}
            miss = 0
            for epo in range(1):
                start_time = time.time()
                # print("start epo %d" % (epo))
                filepath = self.test_path
                # print('filename: ', filepath)
                for line1 in open(filepath):
                    data = line1.strip().split('\t')
                    # pos = self.pos_dict['idx2structure'][int(data[5])]
                    # pos = ['<' + l + '>' for l in pos.strip().split()]
                    vec1 = self.text2id(data[0].strip(), self.src_max_len, False)
                    # vec2 = self.text2id(data[1].strip(), self.max_len, False)

                    if vec1 is not None:
                        # vec1 = self.addpos2text(pos, vec1, self.src_max_len)
                        batch['src'].append(vec1)
                        batch['src_mask'].append(np.expand_dims(vec1 != self.pad, -2).astype(np.float))
                        batch['sampler_label'].append(int(data[4]))
                        # batch['tgt'].append(np.concatenate([[self.bos],vec2], axis=0)[:-1])
                        # batch['y'].append(vec2)
                        # batch['tgt_mask'].append(self.subsequent_mask(vec2))

                        if len(batch['src']) == self.batch_size:
                            batch = {k: cc(v) for k, v in batch.items()}
                            torch.cuda.synchronize()
                            yield batch
                            # batch = {'src':[], 'src_mask':[], 'sampler_label':[], 'tgt':[], 'tgt_mask':[], 'y':[]}
                            batch = {'src':[], 'src_mask':[], 'sampler_label':[]}
                    else:
                        miss += 1
                        print('Wrong Data....', miss)
                if len(batch['src']) != 0:
                    batch = {k: cc(v) for k, v in batch.items()}
                    torch.cuda.synchronize()
                    yield batch
                end_time = time.time()
                print('finish epo %d, time %f' % (epo,end_time-start_time))





    def id2sent(self, indices, test=False):
        sent = []
        word_dict={}
        for index in indices:
            if test and (index == self.unk or index == self.pad or index == self.eos or index in word_dict):
                continue
            sent.append(self.index2word[index])
            word_dict[index] = 1

        return ' '.join(sent)


    def subsequent_mask(self, vec):
        attn_shape = (vec.shape[-1], vec.shape[-1])
        return (np.triu(np.ones((attn_shape)), k=1).astype('uint8') == 0).astype(np.float)



# class data_utils():
#     def __init__(self, args):
#         self.batch_size = args.batch_size

#         dict_path = '../origin/data/dictionary.json'
#         self.train_path = '../origin/data/train.article.txt'
#         if os.path.exists(dict_path):
#             self.word2id = read_json(dict_path)
#         else:
#             self.word2id = make_dict(25000, dict_path, self.train_path)

#         self.index2word = [[]]*len(self.word2id)
#         for word in self.word2id:
#             self.index2word[self.word2id[word]] = word

#         self.vocab_size = len(self.word2id)
#         print('vocab_size:',self.vocab_size)
#         self.eos = self.word2id['__EOS__']
#         self.bos = self.word2id['__BOS__']


#     def text2id(self, text, seq_length=40):
#         vec = np.zeros([seq_length] ,dtype=np.int32)
#         unknown = 0.
#         word_list = text.strip().split()
#         length = len(word_list)

#         for i,word in enumerate(word_list):
#             if i >= seq_length:
#                 break
#             if word in self.word2id:
#                 vec[i] = self.word2id[word]
#             else:
#                 vec[i] = self.word2id['__UNK__']
#                 unknown += 1

#         if unknown / length > 0.1 or length > seq_length*1.5:
#             vec = None

#         return vec


#     def data_yielder(self):
#         batch = {'src':[],'tgt':[],'src_mask':[],'tgt_mask':[],'y':[]}
#         for epo in range(20):
#             start_time = time.time()
#             print("start epo %d" % (epo))
#             for line1,line2 in zip(open('../origin/data/train.article.txt'),open('../origin/data/train.title.txt')):
#                 vec1 = self.text2id(line1.strip(), 45)
#                 vec2 = self.text2id(line2.strip(), 14)

#                 if vec1 is not None and vec2 is not None:
#                     batch['src'].append(vec1)
#                     batch['src_mask'].append(np.expand_dims(vec1 != self.eos, -2).astype(np.float))
#                     batch['tgt'].append(np.concatenate([[self.bos],vec2], axis=0)[:-1])
#                     batch['tgt_mask'].append(self.subsequent_mask(vec2))
#                     batch['y'].append(vec2)

#                     if len(batch['src']) == self.batch_size:
#                         batch = {k: cc(v) for k, v in batch.items()}
#                         torch.cuda.synchronize()
#                         yield batch
#                         batch = {'src':[],'tgt':[],'src_mask':[],'tgt_mask':[],'y':[]}
#             end_time = time.time()
#             print('finish epo %d, time %f' % (epo,end_time-start_time))



#     def id2sent(self,indices, test=False):
#         sent = []
#         word_dict={}
#         for index in indices:
#             if test and (index == self.word2id['__EOS__'] or index in word_dict):
#                 continue
#             sent.append(self.index2word[index])
#             word_dict[index] = 1

#         return ' '.join(sent)


#     def subsequent_mask(self, vec):
#         attn_shape = (vec.shape[-1], vec.shape[-1])
#         return (np.triu(np.ones((attn_shape)), k=1).astype('uint8') == 0).astype(np.float)
