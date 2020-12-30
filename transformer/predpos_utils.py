from __future__ import print_function
import json
import os
import operator
import time
import codecs
import pickle

import numpy as np
import torch
from tqdm import tqdm


def read_json(filename):
    '''Read in a json file.'''
    with open(filename, 'r') as json_file:
        data = json.load(json_file)
    return data


def write_json(filename, data):
    '''Write a json file to filename.'''
    with open(filename, 'w') as json_file:
        json.dump(data, json_file)


def make_save_dir(save_dir):
    '''Make a directory if it does not exist.'''
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir


def cc(arr):
    '''Convert a function to cuda tensor'''
    return torch.from_numpy(np.array(arr)).cuda()

def np2tens(p):
    return torch.from_numpy(p).cuda()

def tens2np(tensor):
    '''Convert tensor to numpy array'''
    return tensor.detach().cpu().numpy()


def subsequent_mask(vec):
    ''' Create mask for decoding(in order to be causal). '''
    attn_shape = (vec.shape[-1], vec.shape[-1])
    return (np.triu(np.ones((attn_shape)), k=1).astype('uint8') == 0).astype(np.float)


def make_dict(max_num, dict_path, train_path, valid_path, test_path):
    '''create dict with voc length of max_num'''
    word_count = dict()
    word2id = dict()
    line_count = 0

    for line in tqdm(open(train_path)):
        line_count += 1.0
        data = line.strip('\n').split('\t')
        for word in data[0].strip().split():
            word_count[word] = word_count.get(word, 0) + 1
        for word in data[2].strip().split():
            word_count[word] = word_count.get(word, 0) + 1
        # for word in line:
        #     word_count[word] = word_count.get(word,0) + 1
    for line in tqdm(open(valid_path)):
        line_count += 1.0
        # for word in line:
        #     word_count[word] = word_count.get(word,0) + 1
        data = line.strip('\n').split('\t')
        for word in data[0].strip().split():
            word_count[word] = word_count.get(word, 0) + 1
        for word in data[2].strip().split():
            word_count[word] = word_count.get(word, 0) + 1

    for line in tqdm(open(test_path)):
        line_count += 1.0
        # for word in line:
        #     word_count[word] = word_count.get(word,0) + 1
        data = line.strip('\n').split('\t')
        for word in data[0].strip().split():
            word_count[word] = word_count.get(word, 0) + 1
        for word in data[2].strip().split():
            word_count[word] = word_count.get(word, 0) + 1
    word2id['</s>'] = len(word2id)
    word2id['<s>'] = len(word2id)
    word2id['<blank>'] = len(word2id)
    word2id['<pad>'] = len(word2id)

    ###### for POS ######
    postags_complete_list = ['<n>', '<nt>', '<nd>', '<nl>', '<nh>', '<nhf>', \
                             '<nhs>', '<ns>', '<nn>', '<ni>', '<nz>', '<v>', \
                             '<vd>', '<vl>', '<vu>', '<a>', '<f>', '<m>', \
                             '<mq>', '<q>', '<d>', '<r>', '<p>', '<c>', '<u>', \
                             '<e>', '<o>', '<i>', '<j>', '<h>', '<k>', '<g>', \
                             '<x>', '<w>', '<ws>', '<wu>']
    for pos in postags_complete_list:
        word2id[pos] = len(word2id)

    word_count_list = sorted(word_count.items(), key=operator.itemgetter(1))
    print(len(word_count_list))

    for item in word_count_list[-(max_num*2):][::-1]:
        if item[1] < word_count_list[-max_num][1]:
            continue
        word = item[0]
        word2id[word] = len(word2id)

    write_json(dict_path, word2id)

    return word2id

###########################################
######### for Sequence Generator ##########
###########################################
def apply_to_sample(func, sample):
    '''apply a function to batch(sample)'''
    if hasattr(sample, '__len__') and len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return func(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        elif isinstance(x, tuple):
            return tuple(_apply(x) for x in x)
        elif isinstance(x, set):
            return {_apply(x) for x in x}
        else:
            return x

    return _apply(sample)


def move_to_cuda(sample):
    '''Move a batch to cuda.'''
    def _move_to_cuda(tensor):
        return tensor.cuda()

    return apply_to_sample(_move_to_cuda, sample)


def strip_pad(tensor, pad):
    '''Strip padding.'''
    return tensor[tensor.ne(pad)]

######### for Sequence Generator ##########

class DataUtils(object):
    '''

    Handle data loading and other utilities.
    Anything about data.

    Args:
        config (yaml): config file from solver
        train (bool): whether doing training
        task (string): which task the solver is currently doing

    Attributes:
        _data_path (str): directory to all the data
        _dict_path (str): dictionary path
        _src_max_len (int): maximum length of input data(including POS)
        _max_len (int): maximum length of target data

        _test_path (str): path to test corpus
        _train_path (str): path to training corpus
        _valid_path (str): path to validation corpus

        pos_dict (dict): 'idx2structure' -> index to POS sequence
                         'structure2idx' -> POS sequence to index
        posmask (dict): POS masking for each POS tag
        word2id (dict): lookup dictionary for word to index
        index2word (dict): lookup dictionary for index to word
        eos, bos, pad, unk (int): index for special tokens

    '''
    def __init__(self, config, train, task):
        self.config = config
        self.train = train
        self.task = task

        self._batch_size = config['batch_size']
        self._data_path = config['data_path']
        self._dict_path = config['dict']
        self._src_max_len = config['src_max_len']
        self._max_len = config['max_len']

        self._test_path = os.path.join(self._data_path, config['test_corpus'])
        self._train_path = os.path.join(self._data_path, config['corpus'])
        self._valid_path = os.path.join(self._data_path, config['valid_corpus'])
        pos_dict_path = os.path.join(self._data_path, config['pos_dict_path'])

        self.pos_dict = pickle.load(open(pos_dict_path, 'rb'))
        print('loading pos_dict_path: ', config['pos_dict_path'])
        print('dict size: %d'%(len(self.pos_dict['idx2structure'])))
        print()
        if config['pos_masking']:
            self.pos2mask = read_json(config['posmask'])
            for pos, mask in self.pos2mask.items():
                self.pos2mask[pos] = cc(mask)

        if not self.train:
            assert os.path.exists(self._dict_path)

        if os.path.exists(self._dict_path):
            print('Loading Dictionary %s' %self._dict_path)
            self.word2id = read_json(self._dict_path)
        else:
            self.word2id = make_dict(50000,
                                     self._dict_path,
                                     self._train_path,
                                     self._valid_path,
                                     self._test_path
                                    )

        self.index2word = [[]]*len(self.word2id)
        for word in self.word2id:
            self.index2word[self.word2id[word]] = word

        self.vocab_size = len(self.word2id)
        print('vocab_size:', self.vocab_size)
        if self.config['pretrain_utils']:
            self.eos = self.word2id['<EOS>']
            self.bos = self.word2id['<SOS>']
            self.pad = self.word2id['<PAD>']
            self.unk = self.word2id['<UNK>']
        else:
            self.eos = self.word2id['</s>']
            self.bos = self.word2id['<s>']
            self.pad = self.word2id['<pad>']
            self.unk = self.word2id['<blank>']

        if self.task == 'joint_gen':
            self.seg = self.word2id['<seg>']

    def text2id(self, text, seq_length=40, pos=False, train=False):
        ''' Convert text index via word2id dictionary. '''
        vec = np.full([seq_length], self.pad, dtype=np.int32)
        unknown = 0.
        word_list = text.strip().split()
        length = len(word_list)

        for i, word in enumerate(word_list):
            if i >= seq_length:
                break
            if pos:
                word = '<' + word + '>'
            if word in self.word2id:
                vec[i] = self.word2id[word]
            else:
                #vec[i] = self.word2id['__UNK__']
                vec[i] = self.unk
                unknown += 1

        if train:
            if unknown / length > 0.1 or length > seq_length*1.5:
                vec = None
        return vec

    def addpos2text(self, pos_list, vec1, max_length):
        ''' Add POS to source text. '''
        seq_len = np.sum(vec1 != self.pad)
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

    def data_yielder(self, valid=False):
        assert self.task == 'pred_pos'
        ''' The function that helps create batch data. '''
        if self.train:
            batch = {'src':[], 'tgt':[], 'src_mask':[], 'tgt_mask':[], 'y':[], 'pos_tgt':[], 'pos_tgt_mask':[], 'pos_y':[]}

            src_file = self._valid_path if valid else self._train_path
            num_epoch = 1 if valid else self.config['num_epoch']
            print('reading from %s, train for %d epoch(s)...'%(src_file, num_epoch))

            with codecs.open(src_file, encoding="utf8", errors='ignore') as train_f:
                start_time = time.time()

                for line in train_f:
                    # handle vector data
                    data = line.strip().split('\t')
                    vec1 = self.text2id(text=data[0].strip(),
                                        seq_length=self._src_max_len,
                                        pos=False,
                                        train=not valid
                                       )
                    vec2 = self.text2id(data[3].strip()+' </s>', self._max_len, pos=True, train=not valid)
                    vec3 = self.text2id(data[2].strip()+' </s>', self._max_len, pos=False, train=not valid)

                    if vec1 is not None and vec2 is not None and vec3 is not None:
                        
                        batch['src'].append(vec1)
                        batch['src_mask'].append(np.expand_dims(vec1 != self.pad, -2).astype(np.float))
                        batch['pos_tgt'].append(np.concatenate([[self.bos], vec2], axis=0)[:-1])
                        batch['pos_tgt_mask'].append(subsequent_mask(vec2))
                        batch['tgt'].append(np.concatenate([[self.bos], vec3], axis=0)[:-1])
                        batch['tgt_mask'].append(subsequent_mask(vec2))
                        batch['pos_y'].append(vec3)
                        batch['y'].append(vec2)


                        if len(batch['src']) == self._batch_size:
                            batch = {k: cc(v) for k, v in batch.items()}
                            yield batch
                            batch = {'src':[], 'tgt':[], 'src_mask':[], 'tgt_mask':[], 'y':[], 'pos_tgt':[], 'pos_tgt_mask':[], 'pos_y':[]}

                if batch['src']:
                    batch = {k: cc(v) for k, v in batch.items()}
                    yield batch

                end_time = time.time()

        else:
            batch = {'src':[], 'src_mask':[]}
            miss = 0
            for epo in range(1):
                start_time = time.time()
                filepath = self._test_path

                print("start epo %d" % (epo))
                print('Reading from testing data %s ... '%filepath)

                for line1 in open(filepath):
                    data = line1.strip().split('\t')
                    vec1 = self.text2id(data[0].strip(), self._src_max_len)

                    if vec1 is not None:
                        batch['src'].append(vec1)
                        batch['src_mask'].append(np.expand_dims(vec1 != self.pad, -2).astype(np.float))

                        if len(batch['src']) == self._batch_size:
                            batch = {k: cc(v) for k, v in batch.items()}
                            yield batch
                            batch = {'src':[], 'src_mask':[]}
                    else:
                        miss += 1
                        print('Wrong Data....', miss)

                # last batch
                if batch['src']:
                    batch = {k: cc(v) for k, v in batch.items()}
                    yield batch
                end_time = time.time()
                print('finish epo %d, time %f' % (epo, end_time-start_time))

    def id2sent(self, indices, test=False):
        ''' Convert output index into sentences(string).'''
        sent = []
        word_dict = {}
        for index in indices:
            if test and index == self.eos:
                break
            if test and (index == self.unk or index == self.pad or index == self.eos or index in word_dict):
                continue

            sent.append(self.index2word[index])
            word_dict[index] = 1

        return ' '.join(sent)

