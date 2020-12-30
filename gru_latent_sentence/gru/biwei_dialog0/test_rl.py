import argparse
import os
import csv
import pickle
import shutil
import torch
from torch import cuda
import dialog0.Utils as utils
from dialog0.Optim import Optim
from dialog0.Seq2SeqWithRL.Infer import Infer
from dialog0.Sampler.ModelHelper import create_sampler_model
from dialog0.Seq2Seq.ModelHelper import create_seq2seq_model
from dialog0.Seq2SeqWithRL.ModelHelper import create_rl_model
from dialog0.Seq2SeqWithRL.Dataset_Loader import setup_vocab, setup_latent_sentence_vocab, Plain_Text_to_Train_Data, load_format, make_data, make_2stage_data
from dialog0.Seq2SeqWithRL.Vocab_conference import Vocab


parser = argparse.ArgumentParser()
parser.add_argument('-test_corpus', type=str)
parser.add_argument('-test_data', type=str)
parser.add_argument("-latent_sentence_dict", type=str)
parser.add_argument("-vocab_data", type=str)
parser.add_argument("-RL_config", type=str)
parser.add_argument("-RL_model", type=str)
parser.add_argument("-report", type=str)
parser.add_argument('-gpuid', default=[], nargs='+', type=int)

args = parser.parse_args()
config = utils.load_config(args.RL_config)
if args.gpuid:
    cuda.set_device(args.gpuid[0])
    device = torch.device('cuda:{}'.format(args.gpuid[0]))


test_rl_report = open(os.path.join(args.report, 'test_rl_report.tsv'), 'w')
tsv_writer = csv.writer(test_rl_report, delimiter='\t')

def report_test_func(results):
    for result in results:
        tsv_writer.writerow(result[0], result[1], result[2], result[3])


def setup_data(vocab, latent_sentence_2_index):
    if not os.path.exists(args.test_data):
        print('testing data not found ...')
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


def test_model(rl_model, test_data, vocab, index_2_latent_sentence, device=0):
    print('Start RL Testing!\n')

    infer_machine = Infer(rl_model,
                          test_data, vocab, index_2_latent_sentence,
                          config['Misc']['use_cuda'], device)

    print('start testing...\n')
    infer_machine.infer(make_data=make_data, make_2stage_data=make_2stage_data, report_func=report_test_func)


def main():
    # set up vocabulary
    print('setting up vocabulary...')
    vocab = load_vocab(args.vocab_data)
    print('vocab dictionary containing {} words'.format(vocab.n_words))

    # load latent sentence dictionary
    latent_sentence_2_index, index_2_latent_sentence = load_latent_sentence_vocab(args.latent_sentence_dict)

    # set up training & validation data
    setup_data(vocab, latent_sentence_2_index)

    # load prepared training data
    test_data = load_format(args.test_data, train_or_valid=False)

    # construct Sampler model architecture
    sampler_model = create_sampler_model(vocab=vocab, index_2_latent_sentence=index_2_latent_sentence, device=device)

    # construct Seq2Seq model architecture
    seq2seq_model = create_seq2seq_model(vocab=vocab, device=device)

    # construct RL model
    rl_model = create_rl_model(sampler_model=sampler_model, seq2seq_model=seq2seq_model)
    if args.RL_model:
        rl_model.load_checkpoint(os.path.join(config['Seq2SeqWithRL']['Trainer']['out_dir'], args.RL_model))

    if config['Misc']['use_cuda']:
        rl_model = rl_model.cuda()

    print("\nfinish loading model\n")

    # train it!
    test_model(rl_model, test_data, vocab, index_2_latent_sentence, device=device)



if __name__ == '__main__':
    main()
