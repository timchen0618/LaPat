# -*- coding: utf-8 -*-
from __future__ import print_function
import yaml
from torch.autograd import Variable
import torch
import codecs
import os
import sys

class HParams(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)


def load_hparams(config_file):
    with codecs.open(config_file, 'r', encoding='utf8') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
        hparams = HParams(**configs)
        return hparams

def load_config(config_file):
    with codecs.open(config_file, 'r', encoding='utf8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        return config


def latest_checkpoint(model_dir):
    cnpt_file = os.path.join(model_dir,'checkpoint')
    try:
        cnpt = open(cnpt_file,'r').readline().strip().split(':')[-1]
    except:
        return None
    cnpt = os.path.join(model_dir,cnpt)
    return cnpt


def print_out(s, f=None, new_line=True):
  """Similar to print but with support to flush and output to a file."""
  if isinstance(s, bytes):
    s = s.decode("utf-8")

  if f:
    f.write(s.encode("utf-8"))
    if new_line:
      f.write(b"\n")

  # stdout
  out_s = s.encode("utf-8")
  if not isinstance(out_s, str):
    out_s = out_s.decode("utf-8")
  print(out_s, end="", file=sys.stdout)

  if new_line:
    sys.stdout.write("\n")
  sys.stdout.flush()

# Pad a with the PAD symbol
def pad_seq(seq, max_length, padding_idx):
    seq += [padding_idx for i in range(max_length - len(seq))]
    return seq

def get_src_input_seq(seq):
    seq = seq.split(' ')
    seq = [IO.BOS_WORD]+seq
    return seq

def seq2indices(seq, word2index, max_len=None):
    seq_idx = []

    if max_len is not None:
        seq = seq[:max_len]
    for w in seq:
        seq_idx.append(word2index[w])

    return seq_idx

def batch_seq2var(batch_src_seqs, word2index, use_cuda=True):
    src_seqs = [get_src_input_seq(seq) for seq in batch_src_seqs]
    src_seqs = sorted(src_seqs, key=lambda p: len(p), reverse=True)
    src_inputs = [seq2indices(s,word2index) for s in src_seqs]

    src_input_lengths = [len(s) for s in src_inputs]
    paded_src_inputs = [pad_seq(s, max(src_input_lengths), word2index[IO.PAD_WORD]) for s in src_inputs]
    src_input_var = Variable(torch.LongTensor(paded_src_inputs), volatile=True).transpose(0, 1)
    if use_cuda:
        src_input_var = src_input_var.cuda()
    return src_input_var, src_input_lengths

def indices2words(idxs, index2word):
    words_list = [index2word[idx] for idx in idxs]
    return words_list


class ShowProcess():
    """
    显示处理进度的类
    调用该类相关函数即可实现处理进度的显示
    """
    i = 1 # 当前的处理进度
    max_steps = 0 # 总共需要处理的次数
    max_arrow = 50 #进度条的长度

    # 初始化函数，需要知道总共的处理次数
    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.i = 1

    # 显示函数，根据当前的处理进度i显示进度
    # 效果为[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>]100.00%
    def show_process(self, i=None):
        if i is not None:
            self.i = i
        num_arrow = int(self.i * self.max_arrow / self.max_steps) #计算显示多少个'>'
        num_line = self.max_arrow - num_arrow #计算显示多少个'-'
        percent = self.i * 100.0 / self.max_steps #计算完成进度，格式为xx.xx%
        process_bar = '[' + '>' * num_arrow + '-' * num_line + ']'\
                      + '%.2f' % percent + '%' + '\r' #带输出的字符串，'\r'表示不换行回到最左边
        sys.stdout.write(process_bar) #这两句打印字符到终端
        sys.stdout.flush()
        self.i += 1

    def close(self, words='done'):
        print('')
        print(words)
        self.i = 1
