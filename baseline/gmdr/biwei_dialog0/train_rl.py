import argparse
import os
import shutil
import torch
from torch import cuda
import dialog0.Utils as utils
from dialog0.Seq2SeqWithRL.Loss import RLLossCompute
from dialog0.Seq2Seq.Loss import LossCompute
import dialog0.Seq2SeqWithRL.IO as IO
from dialog0.Seq2SeqWithRL.Trainer import Statistics,Trainer
from dialog0.Optim import Optim
from dialog0.Seq2SeqWithRL.ModelHelper import create_seq2seq_rl_model
from dialog0.Seq2Seq.ModelHelper import create_seq2seq_tag_model
from dialog0.TagSampler.ModelHelper import create_tag_sampler
from dialog0.Seq2SeqWithRL.Dataset import RLDataset
from tensorboardX import SummaryWriter
parser = argparse.ArgumentParser()
parser.add_argument('-train_data', type=str)
parser.add_argument("-config", type=str)
parser.add_argument("-vocab", type=str)
parser.add_argument("-seq2seq", type=str)
parser.add_argument("-sampler", type=str)
parser.add_argument("-log_dir", type=str)
parser.add_argument('-gpuid', default=[], nargs='+', type=int)


args = parser.parse_args()
config = utils.load_config(args.config)
if args.gpuid:
    cuda.set_device(args.gpuid[0])

summery_writer = SummaryWriter(args.log_dir)

def report_func(global_step, epoch, batch, num_batches,
                start_time, lr, report_stats):
    """
    This is the user-defined batch-level traing progress
    report function.
    Args:
        epoch(int): current epoch count.
        batch(int): current batch count.
        num_batches(int): total number of batches.
        start_time(float): last report time.
        lr(float): current learning rate.
        report_stats(Statistics): old Statistics instance.
    Returns:
        report_stats(Statistics): updated Statistics instance.
    """
    if batch % config['Seq2SeqWithRL']['Trainer']['steps_per_stats'] == \
            -1 % config['Seq2SeqWithRL']['Trainer']['steps_per_stats']:
        report_stats.print_out(epoch, batch+1, num_batches, start_time)
        summery_writer.add_scalars('progress/reward',{'total_reward': report_stats.sampler_reward/100,
                                             'reward1': report_stats.reward1/100,
                                             'reward2': report_stats.reward2/100,
                                             'reward3': report_stats.reward3/100,},global_step)
        summery_writer.add_scalar('progress/seq2seq_loss',report_stats.seq2seq_loss/100,global_step)
        summery_writer.add_scalar('progress/sampler_loss',report_stats.sampler_loss/100,global_step)
        summery_writer.add_text('progress/infer_words',report_stats.get_infer_words(), global_step)
        report_stats = Statistics()

    return report_stats





def make_train_data_iter(train_data, config):

    return IO.OrderedIterator(
                dataset=train_data, batch_size=config['Seq2SeqWithRL']['Trainer']['batch_size'],
                device=args.gpuid[0] if args.gpuid else -1,
                train=True,
                repeat=False, 
                shuffle=False, 
                sort=False,
                sort_within_batch=False)

def load_fields():
    fields = IO.load_fields(
                torch.load(args.vocab))
    # fields = dict([(k, f) for (k, f) in fields.items()
    #               if k in train.examples[0].__dict__])
    # train.fields = fields

    print(' * vocabulary size. source = %d; target = %d; tag = %d' %
          (len(fields['src'].vocab), len(fields['tgt'].vocab), len(fields['tag'].vocab)))

    return fields

def build_seq2seq_optim(model, config):
    optim = Optim(config['Seq2SeqWithRL']['Trainer']['optim_method'],
                  config['Seq2SeqWithRL']['Trainer']['seq2seq_lr'],
                  config['Seq2SeqWithRL']['Trainer']['max_grad_norm'],
                  config['Seq2SeqWithRL']['Trainer']['learning_rate_decay'],
                  config['Seq2SeqWithRL']['Trainer']['weight_decay'],
                  config['Seq2SeqWithRL']['Trainer']['start_decay_at'])
    optim.set_parameters(model.parameters())
    return optim

def build_sampler_optim(model, config):
    optim = Optim(config['Seq2SeqWithRL']['Trainer']['optim_method'],
                  config['Seq2SeqWithRL']['Trainer']['sampler_lr'],
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
        # save config.yml
        shutil.copy(args.config, os.path.join(config['Seq2SeqWithRL']['Trainer']['out_dir'],'config.yml'))

def train_model(model, train_data, fields, seq2seq_optim, sampler_optim, lr_scheduler, start_epoch_at=0):

    train_iter = make_train_data_iter(train_data, config)

    seq2seq_loss = LossCompute(model.seq2seq.generator,fields['tgt'].vocab)
    sampler_loss = RLLossCompute(model.seq2seq.generator,fields['tgt'].vocab)

    if config['Misc']['use_cuda']:
        seq2seq_loss = seq2seq_loss.cuda()
        sampler_loss = sampler_loss.cuda()

    trainer = Trainer(model,
                        train_iter,
                        seq2seq_loss,
                        sampler_loss,
                        seq2seq_optim,
                        sampler_optim,
                        lr_scheduler)

    num_train_epochs = config['Seq2SeqWithRL']['Trainer']['num_train_epochs']
    print('start training...')
    for step_epoch in  range(start_epoch_at+1, num_train_epochs):
        # trainer.lr_scheduler.step()
        # 1. Train for one epoch on the training set.
        train_stats = trainer.train(step_epoch, fields, report_func)
        trainer.lr_scheduler.step()
        #print('Train perplexity: %g' % train_stats.ppl())

        trainer.save_per_epoch(step_epoch, out_dir=config['Seq2SeqWithRL']['Trainer']['out_dir'])

def main():

    # Load train and validate data.
    print("Loading train data")
    # Load fields generated from preprocess phase.
    fields = load_fields()
    train = RLDataset(
        data_path=args.train_data,
        fields=[('src', fields["src"]),
                ('tgt', fields["tgt"]),
                ('tag', fields["tag"])])
 


    # Build model.

    seq2seq_model = create_seq2seq_tag_model(config, fields)
    if args.seq2seq is not None:
        # seq2seq_model.load_checkpoint(torch.load(args.seq2seq,map_location=lambda storage, loc: storage))
        seq2seq_model.load_checkpoint(args.seq2seq)

    tag_sampler = create_tag_sampler(config, fields)
    if args.sampler is not None:
        # tag_sampler.load_checkpoint(torch.load(args.sampler,map_location=lambda storage, loc: storage))
        tag_sampler.load_checkpoint(args.sampler)
    model = create_seq2seq_rl_model(config, fields, tag_sampler, seq2seq_model)
    check_save_model_path(config)

    # Build optimizer.
    seq2seq_optim = build_seq2seq_optim(model.seq2seq, config)
    sampler_optim = build_sampler_optim(model.tag_sampler, config)
    lr_scheduler = build_lr_scheduler(seq2seq_optim.optimizer)

    if config['Misc']['use_cuda']:
        model = model.cuda()

    # Do training.
    train_model(model, train, fields, seq2seq_optim, sampler_optim, lr_scheduler)

if __name__ == '__main__':
    main()
