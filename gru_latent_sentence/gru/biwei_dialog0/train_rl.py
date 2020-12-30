import argparse
import os
import pickle
import shutil
import torch
from torch import cuda
import dialog0.Utils as utils
from dialog0.Optim import Optim
from dialog0.Seq2SeqWithRL.Loss import RLLossCompute
from dialog0.Seq2SeqWithRL.Trainer import Statistics, Trainer
from dialog0.Sampler.ModelHelper import create_sampler_model
from dialog0.Seq2Seq.ModelHelper import create_seq2seq_model
from dialog0.Seq2SeqWithRL.ModelHelper import create_rl_model
from dialog0.Seq2SeqWithRL.Dataset_Loader import setup_vocab, setup_latent_sentence_vocab, Plain_Text_to_Train_Data, load_format, make_data, make_2stage_data
from dialog0.Seq2SeqWithRL.Vocab_conference import Vocab


parser = argparse.ArgumentParser()
parser.add_argument('-train_corpus', type=str)
parser.add_argument('-valid_corpus', type=str)
parser.add_argument('-train_data', type=str)
parser.add_argument('-valid_data', type=str)
parser.add_argument("-latent_sentence_dict", type=str)
parser.add_argument("-vocab_data", type=str)
parser.add_argument("-RL_config", type=str)
parser.add_argument("-sampler_model", type=str)
parser.add_argument("-seq2seq_model", type=str)
parser.add_argument("-report", type=str)
parser.add_argument('-gpuid', default=[], nargs='+', type=int)

args = parser.parse_args()
config = utils.load_config(args.RL_config)
if args.gpuid:
    cuda.set_device(args.gpuid[0])
    device = torch.device('cuda:{}'.format(args.gpuid[0]))


train_rl_report = open(os.path.join(args.report, 'train_rl_report.txt'), 'w')
valid_rl_report = open(os.path.join(args.report, 'valid_rl_report.txt'), 'w')

def report_train_func(epoch, step_batch, step_batches, report_stats, zoom_in):

    print('\ninput post: {}'.format(zoom_in[0]))
    print('selected latent sentence: {}'.format(zoom_in[1]))
    print('target response: {}'.format(zoom_in[2]))
    print('prediction: {}\n'.format(zoom_in[3]))

    report_stats.print_out(epoch=epoch, step_batch=step_batch, step_batches=step_batches)
    train_rl_report.write('=============================================================================\n')
    train_rl_report.write('epoch: {}\n'.format(epoch))
    train_rl_report.write('input post: {}\n'.format(zoom_in[0]))
    train_rl_report.write('selected latent sentence: {}\n'.format(zoom_in[1]))
    train_rl_report.write('target response: {}\n'.format(zoom_in[2]))
    train_rl_report.write('predict response: {}\n'.format(zoom_in[3]))
    train_rl_report.write('sampler_loss: {} | seq2seq_loss: {}\n'.format(report_stats.sampler_loss/report_stats.batches, report_stats.seq2seq_loss/report_stats.batches))
    train_rl_report.write('total_reward: {} | reward1: {} | reword2: {} | reward3: {}\n'.format(report_stats.rl_reward/report_stats.batches, report_stats.reward1/report_stats.batches, report_stats.reward2/report_stats.batches, report_stats.reward3/report_stats.batches))
    train_rl_report.write('avg_f1: {}\n'.format(report_stats.avg_f1/report_stats.batches))
    train_rl_report.write('=============================================================================\n\n')


def report_valid_func(epoch, report_stats, zoom_in):

    print('\ninput post: {}'.format(zoom_in[0]))
    print('selected latent sentence: {}'.format(zoom_in[1]))
    print('target response: {}'.format(zoom_in[2]))
    print('prediction: {}\n'.format(zoom_in[3]))

    report_stats.print_out(epoch=epoch)
    train_rl_report.write('=============================================================================\n')
    train_rl_report.write('epoch: {}\n'.format(epoch))
    train_rl_report.write('input post: {}\n'.format(zoom_in[0]))
    train_rl_report.write('selected latent sentence: {}\n'.format(zoom_in[1]))
    train_rl_report.write('target response: {}\n'.format(zoom_in[2]))
    train_rl_report.write('predict response: {}\n'.format(zoom_in[3]))
    train_rl_report.write('sampler_loss: {} | seq2seq_loss: {}\n'.format(report_stats.sampler_loss/report_stats.batches, report_stats.seq2seq_loss/report_stats.batches))
    train_rl_report.write('total_reward: {} | reward1: {} | reword2: {} | reward3: {}\n'.format(report_stats.rl_reward/report_stats.batches, report_stats.reward1/report_stats.batches, report_stats.reward2/report_stats.batches, report_stats.reward3/report_stats.batches))
    train_rl_report.write('avg_f1: {}\n'.format(report_stats.avg_f1/report_stats.batches))
    train_rl_report.write('=============================================================================\n\n')


def setup_data(vocab, latent_sentence_2_index):
    if not os.path.exists(args.train_data):
        print('training data not found ...')
        print('set up training data ...')
        Plain_Text_to_Train_Data(args.train_corpus, args.train_data, vocab, latent_sentence_2_index, train_or_valid=True)
    else:
        print('training data found !')

    if not os.path.exists(args.valid_data):
        print('\nvalidation data not found ...')
        print('set up validation data ...')
        Plain_Text_to_Train_Data(args.valid_corpus, args.valid_data, vocab, latent_sentence_2_index, train_or_valid=True)
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


def build_sampler_optim(model, config):
    optim = Optim(config['Seq2SeqWithRL']['Trainer']['optim_method'],
                  config['Seq2SeqWithRL']['Trainer']['sampler_lr'],
                  config['Seq2SeqWithRL']['Trainer']['max_grad_norm'],
                  config['Seq2SeqWithRL']['Trainer']['learning_rate_decay'],
                  config['Seq2SeqWithRL']['Trainer']['weight_decay'],
                  config['Seq2SeqWithRL']['Trainer']['start_decay_at'])
    optim.set_parameters(model.parameters())
    return optim


def build_seq2seq_optim(model, config):
    optim = Optim(config['Seq2SeqWithRL']['Trainer']['optim_method'],
                  config['Seq2SeqWithRL']['Trainer']['seq2seq_lr'],
                  config['Seq2SeqWithRL']['Trainer']['max_grad_norm'],
                  config['Seq2SeqWithRL']['Trainer']['learning_rate_decay'],
                  config['Seq2SeqWithRL']['Trainer']['weight_decay'],
                  config['Seq2SeqWithRL']['Trainer']['start_decay_at'])
    optim.set_parameters(model.parameters())
    return optim


def build_lr_scheduler(optimizer):
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
    lambda2 = lambda epoch: config['Seq2SeqWithRL']['Trainer']['learning_rate_decay'] ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                                  lr_lambda=[lambda2])
    return scheduler


def check_save_model_path(config):
    if not os.path.exists(config['Seq2SeqWithRL']['Trainer']['out_dir']):
        os.makedirs(config['Seq2SeqWithRL']['Trainer']['out_dir'])
        print('saving config file to %s ...'%(config['Seq2SeqWithRL']['Trainer']['out_dir']))


def train_model(rl_model, train_data, valid_data, vocab, index_2_latent_sentence, sampler_optim, seq2seq_optim, lr_scheduler, start_epoch_at=0, device=0):
    print('Start Reinforcement Learning!\n')
    num_train_epochs = config['Seq2SeqWithRL']['Trainer']['num_train_epochs']

    # RL loss reward function
    rl_loss = RLLossCompute(vocab=vocab)
    if config['Misc']['use_cuda']:
        rl_loss = rl_loss.cuda()

    trainer = Trainer(rl_model,
                      train_data, valid_data, vocab, index_2_latent_sentence,
                      rl_loss, sampler_optim, seq2seq_optim, lr_scheduler,
                      config['Misc']['use_cuda'], device)

    print('start training...\n')
    for step_epoch in range(start_epoch_at, num_train_epochs):
        print(step_epoch)

        # training part
        trainer.train(epoch=step_epoch+1, make_data=make_data, make_2stage_data=make_2stage_data, report_func=report_train_func)

        # validation part
        trainer.valid(epoch=step_epoch+1, make_data=make_data, make_2stage_data=make_2stage_data, report_func=report_valid_func)

        # learning rate scheduler
        trainer.lr_scheduler.step()
        # save model
        trainer.save_per_epoch(step_epoch, out_dir=config['Seq2SeqWithRL']['Trainer']['out_dir'])


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
    train_data = load_format(args.train_data)
    valid_data = load_format(args.valid_data)

    # construct Sampler model architecture
    sampler_model = create_sampler_model(vocab=vocab, index_2_latent_sentence=index_2_latent_sentence, device=device)
    if args.sampler_model is not None:
        sampler_model.load_checkpoint(args.sampler_model)

    # construct Seq2Seq model architecture
    seq2seq_model = create_seq2seq_model(vocab=vocab, device=device)
    if args.seq2seq_model is not None:
        seq2seq_model.load_checkpoint(args.seq2seq_model)

    # construct RL model
    rl_model = create_rl_model(sampler_model=sampler_model, seq2seq_model=seq2seq_model)
    check_save_model_path(config)

    # Build optimizer.
    sampler_optim = build_sampler_optim(rl_model.sampler_model, config)
    seq2seq_optim = build_seq2seq_optim(rl_model.seq2seq_model, config)
    lr_scheduler = build_lr_scheduler(seq2seq_optim.optimizer)

    if config['Misc']['use_cuda']:
        rl_model = rl_model.cuda()

    print("\nfinish loading model\n")

    # train it!
    train_model(rl_model, train_data, valid_data, vocab, index_2_latent_sentence, sampler_optim, seq2seq_optim, lr_scheduler, start_epoch_at=0, device=device)



if __name__ == '__main__':
    main()
