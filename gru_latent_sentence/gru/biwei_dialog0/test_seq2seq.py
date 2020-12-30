import argparse
import os
import pickle
import csv
import torch
from torch import cuda
import dialog0.Utils as utils
from dialog0.Seq2Seq.Tester import Tester
from dialog0.Seq2Seq.ModelHelper_Beam import create_seq2seq_model
from dialog0.Seq2Seq.Vocab_conference import *
from dialog0.Seq2Seq.Dataset_Loader import setup_vocab, Plain_Text_to_Train_Data, load_format, make_data
from dialog0.Seq2Seq import config


parser = argparse.ArgumentParser()
parser.add_argument("-test_corpus", type=str)
parser.add_argument("-test_data", type=str)
parser.add_argument("-vocab_data", type=str)
parser.add_argument("-report", type=str)
parser.add_argument("-gpuid", default=[], nargs='+', type=int)


args = parser.parse_args()
if args.gpuid:
    cuda.set_device(args.gpuid[0])
    device = torch.device('cuda:{}'.format(args.gpuid[0]))



test_seq2seq_report = open(os.path.join(args.report, 'test_seq2seq_report.tsv'), 'w')
tsv_writer = csv.writer(test_seq2seq_report, delimiter='\t')

def report_test_func(results):
    for result in results:
        tsv_writer.writerow([result])


def setup_data(vocab):
    if not os.path.exists(args.test_data):
        print('testing data not found ...')
        print('set up testing data ...\n')
        Plain_Text_to_Train_Data(args.test_corpus, args.test_data, vocab)
    else:
        print('testing data found !')


def load_vocab(vocab_file_path):
    word2index_file_path = vocab_file_path + '/word2index.pkl'
    index2word_file_path = vocab_file_path + '/index2word.pkl'
    wordcount_file_path = vocab_file_path + '/wordcount.pkl'

    # load vocabulary data
    with open(word2index_file_path, 'rb') as pkl_in:
        word2index = pickle.load(pkl_in)

    with open(index2word_file_path, 'rb') as pkl_in:
        index2word = pickle.load(pkl_in)

    with open(wordcount_file_path, 'rb') as pkl_in:
        wordcount = pickle.load(pkl_in)

    vocab = Vocab()
    vocab.word2index = word2index
    vocab.index2word = index2word
    vocab.n_words = wordcount

    return vocab


def build_or_load_model(vocab, device):

    model = create_seq2seq_model(vocab=vocab, device=device)
    latest_ckpt = utils.latest_checkpoint(config.out_dir)

    ckpt = latest_ckpt
    # latest_ckpt = nmt.misc_utils.latest_checkpoint(model_dir)
    if ckpt:
        print('Loding model from %s...'%(ckpt))
        start_epoch_at = model.load_checkpoint(ckpt)
    else:
        print('Building model...')

    print('\n')
    print(model)
    print('\n')

    return model


def test_model(model, test_data, vocab):
    tester = Tester(model, test_data, vocab)

    print('start testing...')
    p_gen, l_copy = tester.test(make_data, report_test_func)
    tsv_writer.writerow([p_gen, l_copy])


def main():
    # set up vocabulary for training corpus
    print('setting up data & vocabulary ...')
    vocab = load_vocab(args.vocab_data)
    print('dictionary containing {} words'.format(vocab.n_words))


    # set up testing data
    setup_data(vocab)

    # load testing data
    test_data = load_format(args.test_data)

    # construct model architecture
    print('seq2seq attention & copy mechanism model building ...')
    model = build_or_load_model(vocab, device)

    # distribute model into GPU
    if config.use_cuda:
        model = model.cuda()

    # Do training
    test_model(model, test_data, vocab)



if __name__ == '__main__':
    main()
