import argparse
import os
import pickle
from gensim.models.wrappers import FastText
import numpy as np
import torch
from torch import cuda
import dialog0.Utils as utils
from dialog0.Seq2Seq.Trainer import Statistics, Trainer
from dialog0.Optim import Optim
from dialog0.Seq2Seq.ModelHelper import create_seq2seq_model
from dialog0.Seq2Seq.Vocab_conference import *
from dialog0.Seq2Seq.Dataset_Loader import setup_vocab, Plain_Text_to_Train_Data, load_format, make_data
from dialog0.Seq2Seq import config


#parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-train_corpus", type=str)
parser.add_argument("-valid_corpus", type=str)
parser.add_argument("-train_data", type=str)
parser.add_argument("-valid_data", type=str)
parser.add_argument("-vocab_data", type=str)
parser.add_argument("-gpuid", default=[], nargs='+', type=int)


args = parser.parse_args()
if args.gpuid:
    cuda.set_device(args.gpuid[0])
    device = torch.device('cuda:{}'.format(args.gpuid[0]))


train_seq2seq_report = open('./report/train_seq2seq_report.txt', 'w')
valid_seq2seq_report = open('./report/valid_seq2seq_report.txt', 'w')



def report_train_func(epoch, step_batch, step_batches, report_stats, zoom_in, probabilities):

    print('\ninput post: {}'.format(zoom_in[0]))
    print('latent sentence: {}'.format(zoom_in[1]))
    print('target response: {}'.format(zoom_in[2]))
    print('prediction: {}\n'.format(zoom_in[3]))

    report_stats.print_out(state='train', step_batch=step_batch, step_batches=step_batches, epoch=epoch+1)
    train_seq2seq_report.write('=============================================================================\n')
    train_seq2seq_report.write('input post: {}\n'.format(zoom_in[0]))
    train_seq2seq_report.write('latent word: {}\n'.format(zoom_in[1]))
    train_seq2seq_report.write('target response: {}\n'.format(zoom_in[2]))
    train_seq2seq_report.write('predict response: {}\n'.format(zoom_in[3]))
    train_seq2seq_report.write('p_gen: {} | l_copy: {}\n'.format(probabilities[0],probabilities[1]))
    train_seq2seq_report.write('epoch: {} | avg_loss: {}\n'.format(epoch,report_stats.avg_loss))
    train_seq2seq_report.write('=============================================================================\n\n')



def report_valid_func(epoch, report_stats, zoom_in, probabilities):

    print('\ninput post: {}'.format(zoom_in[0]))
    print('latent sentence: {}'.format(zoom_in[1]))
    print('target response: {}'.format(zoom_in[2]))
    print('prediction: {}\n'.format(zoom_in[3]))

    avg_loss = report_stats.print_out(state='valid', step_batch=1, step_batches=1, epoch=epoch+1)
    valid_seq2seq_report.write('=============================================================================\n')
    valid_seq2seq_report.write('input post: {}\n'.format(zoom_in[0]))
    valid_seq2seq_report.write('latent word: {}\n'.format(zoom_in[1]))
    valid_seq2seq_report.write('target response: {}\n'.format(zoom_in[2]))
    valid_seq2seq_report.write('predict response: {}\n'.format(zoom_in[3]))
    valid_seq2seq_report.write('p_gen: {} | l_copy: {}\n'.format(probabilities[0],probabilities[1]))
    valid_seq2seq_report.write('avg_loss: {}\n'.format(report_stats.avg_loss))
    valid_seq2seq_report.write('=============================================================================\n\n')

    return avg_loss



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



def setup_data(vocab):
    if not os.path.exists(args.train_data):
        print('training data not found ...')
        print('set up training data ...')
        Plain_Text_to_Train_Data(args.train_corpus, args.train_data, vocab)
    else:
        print('training data found !')

    if not os.path.exists(args.valid_data):
        print('\nvalidation data not found ...')
        print('set up validation data ...')
        Plain_Text_to_Train_Data(args.valid_corpus, args.valid_data, vocab)
    else:
        print('validation data found !')



def build_or_load_model(vocab, device):

    model = create_seq2seq_model(vocab=vocab, device=device)
    print('Building model...')

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
        print('saving config file to %s ...'%(config.out_dir))
        # save config.yml



def train_model(model, train_data, valid_data, optim, lr_scheduler, vocab):
    trainer = Trainer(model, train_data, valid_data, vocab, optim, lr_scheduler)

    num_train_epochs = config.num_train_epochs
    print('start training ...')
    for step_epoch in range(num_train_epochs):

        # 1. Train for one epoch on the training set.
        trainer.train(step_epoch, make_data, report_train_func)

        # 2. Valid for one epoch on the validation set.
        avg_loss = trainer.valid(step_epoch, make_data, report_valid_func)

        trainer.lr_scheduler.step()

        # save model
        trainer.save_per_epoch(step_epoch, out_dir=config.out_dir)



def main():
    # set up vocabulary for training corpus
    print('setting up data & vocabulary ...')
    vocab = load_vocab(args.vocab_data)
    print('Sampler dictionary containing {} words'.format(vocab.n_words))

    # set up training data
    setup_data(vocab)

    # load training, validation, testing data
    train_data = load_format(args.train_data)
    valid_data = load_format(args.valid_data)

    # construct model architecture
    print('seq2seq attention model building ...')
    model = build_or_load_model(vocab, device)
    check_save_model_path()

    # build optimizer & learning rate scheduler
    optim = build_optim(model)
    lr_scheduler = build_lr_scheduler(optim.optimizer)

    # distribute model into GPU
    if config.use_cuda:
        model = model.cuda()

    # Do training
    train_model(model, train_data, valid_data, optim, lr_scheduler, vocab)


if __name__ == '__main__':
    main()
