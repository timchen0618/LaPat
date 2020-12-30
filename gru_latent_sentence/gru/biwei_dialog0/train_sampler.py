import argparse
import os
import pickle
import numpy as np
import torch
from torch import cuda
import dialog0.Utils as utils
from dialog0.Sampler.Trainer import Statistics, Trainer
from dialog0.Optim import Optim
from dialog0.Sampler.ModelHelper import create_sampler_model
from dialog0.Sampler.Vocab_conference import *
from dialog0.Sampler.Dataset_Loader import setup_latent_sentence_vocab, setup_vocab, Plain_Text_to_Train_Data, load_format, make_data
from dialog0.Sampler import config


parser = argparse.ArgumentParser()
parser.add_argument('-train_corpus', type=str)
parser.add_argument('-valid_corpus', type=str)
parser.add_argument('-train_data', type=str)
parser.add_argument('-valid_data', type=str)
parser.add_argument("-latent_sentence_dict", type=str)
parser.add_argument("-vocab_data", type=str)
parser.add_argument("-report", type=str)
parser.add_argument('-gpuid', default=[], nargs='+', type=int)

args = parser.parse_args()
if args.gpuid:
    cuda.set_device(args.gpuid[0])
    device = torch.device('cuda:{}'.format(args.gpuid[0]))


train_sampler_report = open(os.path.join(args.report, 'train_sampler_report.tsv'), 'w')
train_sampler_report_writer = csv.writer(train_sampler_report, delimiter='\t')

valid_sampler_report = open(os.path.join(args.report, 'valid_sampler_report.tsv'), 'w')
valid_sampler_report_writer = csv.writer(valid_sampler_report, delimiter='\t')

def report_train_func(epoch, avg_loss, pred_acc, zoom_in):

    print('\nsource: {}'.format(zoom_in[0]))
    print('sample: {}\n'.format(zoom_in[1]))

    pred_acc_demonstrate = str((pred_acc*100)) + '%'

    train_sampler_report_writer.writerow([epoch, zoom_in[0], zoom_in[1], avg_loss, pred_acc_demonstrate])


def report_valid_func(epoch, avg_loss, pred_acc, zoom_in):

    print('\nsource: {}'.format(zoom_in[0]))
    print('sample: {}\n'.format(zoom_in[1]))

    pred_acc_demonstrate = str((pred_acc*100)) + '%'

    valid_sampler_report_writer.writerow([epoch, zoom_in[0], zoom_in[1], avg_loss, pred_acc_demonstrate])


def setup_data(vocab, latent_sentence_2_index):
    # sampler part
    if not os.path.exists(args.train_data):
        print('\ntraining data not found ...')
        print('set up training data ...')
        Plain_Text_to_Train_Data(args.train_corpus, args.train_data, vocab, latent_sentence_2_index)
    else:
        print('training data found !')

    if not os.path.exists(args.valid_data):
        print('\nvalidation data not found ...')
        print('set up validation data ...')
        Plain_Text_to_Train_Data(args.valid_corpus, args.valid_data, vocab, latent_sentence_2_index)
    else:
        print('validation data found !')


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

    print('\n')
    print(model)
    print('\n')

    return model


def build_optim(model):
    optim = Optim(config.optim_method,
                  config.lr,
                  config.max_grad_norm,
                  config.learning_rate_decay,
                  config.weight_decay,
                  config.start_decay_at)
    optim.set_parameters(model.parameters())
    return optim


def build_lr_scheduler(optimizer):
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
    lambda2 = lambda epoch: config.learning_rate_decay ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                                  lr_lambda=[lambda2])
    return scheduler


def check_save_model_path():
    if not os.path.exists(config.out_dir):
        os.makedirs(config.out_dir)


def train_model(model, train_data, valid_data, optim, lr_scheduler, vocab, index_2_latent_sentence, device):
    trainer = Trainer(model, train_data, valid_data, vocab, index_2_latent_sentence, optim, lr_scheduler, use_cuda=True, device=device)

    num_train_epochs = config.num_train_epochs
    print('start training ...')
    for step_epoch in range(num_train_epochs):
        # 1. Train for one epoch on the training set.
        avg_train_loss, pred_train_acc, elapsed_train_time = trainer.train(step_epoch+1, make_data, report_train_func)

        # 2. Valid for one epoch on the validation set.
        avg_dev_loss, pred_dev_acc, elapsed_dev_time = trainer.valid(step_epoch+1, make_data, report_valid_func)

        trainer.lr_scheduler.step()

        # save model
        trainer.save_per_epoch(step_epoch, out_dir=config.out_dir)


def main():
    # set up vocabulary for training corpus
    print('setting up data & vocabulary ...')
    vocab = load_vocab(args.vocab_data)
    print('Sampler dictionary containing {} words'.format(vocab.n_words))

    # load meteor pos sequence dictionary
    latent_sentence_2_index, index_2_latent_sentence = load_latent_sentence_vocab(args.latent_sentence_dict)

    # set up training data
    setup_data(vocab, latent_sentence_2_index)

    # load training, validation, testing data
    train_data = load_format(args.train_data)
    valid_data = load_format(args.valid_data)

    # construct model architecture
    print('Sampler model building ...')
    model = build_or_load_model(vocab, index_2_latent_sentence, device)
    check_save_model_path()

    # build optimizer
    optim = build_optim(model)
    lr_scheduler = build_lr_scheduler(optim.optimizer)

    # distribute model into GPU
    if config.use_cuda:
        model = model.cuda()

    # start training
    train_model(model, train_data, valid_data, optim, lr_scheduler, vocab, index_2_latent_sentence, device)



if __name__ == '__main__':
    main()
