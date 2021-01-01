import argparse
import os
import pickle
import numpy as np
import torch
from torch import cuda
import dialog0.Utils as utils
from dialog0.Sampler.Tester import Tester
from dialog0.Optim import Optim
from dialog0.Sampler.ModelHelper import create_sampler_model
from dialog0.Sampler.Vocab_conference import *
from dialog0.Sampler.Dataset_Loader import setup_latent_sentence_vocab, setup_vocab, Plain_Text_to_Train_Data, load_format, make_data
from dialog0.Sampler import config


parser = argparse.ArgumentParser()
parser.add_argument('-test_corpus', type=str)
parser.add_argument('-test_data', type=str)
parser.add_argument("-latent_sentence_dict", type=str)
parser.add_argument("-vocab_data", type=str)
parser.add_argument("-report", type=str)
parser.add_argument('-gpuid', default=[], nargs='+', type=int)

args = parser.parse_args()
if args.gpuid:
    cuda.set_device(args.gpuid[0])
    device = torch.device('cuda:{}'.format(args.gpuid[0]))


test_sampler_report = open(os.path.join(args.report, 'test_sampler_report.tsv'), 'w')
test_sampler_report_writer = csv.writer(test_sampler_report, delimiter='\t')

def report_test_func(results):
    for result in results:
        test_sampler_report_writer.writerow([result[0], result[1]])


def setup_data(vocab, latent_sentence_2_index):
    # sampler part
    if not os.path.exists(args.test_data):
        print('\ntesting data not found ...')
        print('set up testing data ...')
        Plain_Text_to_Train_Data(args.test_corpus, args.test_data, vocab, latent_sentence_2_index, train_or_valid=False)
    else:
        print('testing data found !')


def load_vocab(vocab_file_path):
    word2index_file_path = vocab_file_path + 'word2index.pkl'
    index2word_file_path = vocab_file_path + 'index2word.pkl'
    wordcount_file_path = vocab_file_path + 'wordcount.pkl'

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


def load_latent_sentence_vocab(latent_sentence_vocab_filepath):
    latent_sentence_2_index, index_2_latent_sentence = setup_latent_sentence_vocab(latent_sentence_vocab_filepath)

    return latent_sentence_2_index, index_2_latent_sentence


def build_or_load_model(vocab, index_2_latent_sentence, device):
    print('Building model...')
    model = create_sampler_model(vocab=vocab, index_2_latent_sentence=index_2_latent_sentence, device=device)

    latest_ckpt = utils.latest_checkpoint(config.out_dir)
    start_epoch_at = 0

    start_epoch_at = model.load_checkpoint(latest_ckpt)

    print('\n')
    print(model)
    print('\n')

    return model


def test_model(model, test_data, vocab, index_2_latent_sentence, device):
    tester = Tester(model, test_data, vocab, index_2_latent_sentence, use_cuda=True, device=device)

    print('start testing...')
    tester.test(make_data, report_test_func)


def main():
    # set up vocabulary for testing corpus
    print('setting up data & vocabulary ...')
    vocab = load_vocab(args.vocab_data)
    print('Sampler dictionary containing {} words'.format(vocab.n_words))

    # load meteor pos sequence dictionary
    latent_sentence_2_index, index_2_latent_sentence = load_latent_sentence_vocab(args.latent_sentence_dict)

    # set up testing data
    setup_data(vocab, latent_sentence_2_index)

    # load training, validation, testing data
    test_data = load_format(args.test_data, train_or_valid=False)

    # construct model architecture
    print('Sampler model building ...')
    model = build_or_load_model(vocab, index_2_latent_sentence, device)

    # distribute model into GPU
    if config.use_cuda:
        model = model.cuda()

    # start testing
    test_model(model, test_data, vocab, index_2_latent_sentence, device)



if __name__ == '__main__':
    main()
