import argparse
import os
import shutil
import torch
from torch import cuda
import dialog0.Utils as utils
from dialog0.TagSampler.Loss import SamplerLossCompute
import dialog0.TagSampler.IO as IO
from dialog0.TagSampler.Trainer import Statistics,Trainer
from dialog0.Optim import Optim
from dialog0.TagSampler.ModelHelper import create_tag_sampler
from dialog0.TagSampler.Dataset import SamplerDataset
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument("-config", type=str)
parser.add_argument("-vocab", type=str)
parser.add_argument('-train_data', type=str)
parser.add_argument("-log_dir", type=str)
parser.add_argument('-gpuid', default=[], nargs='+', type=int)


args = parser.parse_args()
config = utils.load_config(args.config)
#Jay edit
device = None
if args.gpuid:
    cuda.set_device(args.gpuid[0])
    device = torch.device('cuda:{}'.format(args.gpuid[0]))

summery_writer = SummaryWriter(args.log_dir)


def report_func(global_step, epoch, batch, num_batches,
                lr, report_stats):

    if batch % config['TagSampler']['Trainer']['steps_per_stats'] == -1 % config['TagSampler']['Trainer']['steps_per_stats']:
        report_stats.print_out(epoch, batch+1, num_batches)
        summery_writer.add_scalar('progress/bow_loss',report_stats.loss/100,global_step)
        report_stats = Statistics()

    return report_stats


def make_train_data_iter(train_data, config):
    return IO.OrderedIterator(
                dataset=train_data, batch_size=config['TagSampler']['Trainer']['batch_size'],
                device=args.gpuid[0] if args.gpuid else -1,
                repeat=False)

def load_fields():
    fields = IO.load_fields(
                torch.load(args.vocab))
    # fields = dict([(k, f) for (k, f) in fields.items()
    #               if k in train.examples[0].__dict__])
    # train.fields = fields

    print(' * vocabulary size. source = %d; tag = %d' %
            (len(fields['src'].vocab), len(fields['tag'].vocab)))

    return fields


def build_or_load_model(config, fields):

    #Jay edit
    model = create_tag_sampler(config,fields).to(device)
    latest_ckpt = utils.latest_checkpoint(config['TagSampler']['Trainer']['out_dir'])
    start_epoch_at = 0
    if config['TagSampler']['Trainer']['start_epoch_at'] is not None:
        ckpt = 'checkpoint_epoch%d.pkl'%(config['TagSampler']['Trainer']['start_epoch_at'])
        ckpt = os.path.join(config['TagSampler']['Trainer']['out_dir'],ckpt)
    else:
        ckpt = latest_ckpt
    # latest_ckpt = nmt.misc_utils.latest_checkpoint(model_dir)
    if ckpt:
        print('Loading model from %s...'%(ckpt))
        start_epoch_at = model.load_checkpoint(ckpt)
    else:
        print('Building model...')
    print(model)

    return model, start_epoch_at
def build_optim(model, config):
    optim = Optim(config['TagSampler']['Trainer']['optim_method'],
                  config['TagSampler']['Trainer']['learning_rate'],
                  config['TagSampler']['Trainer']['max_grad_norm'],
                  config['TagSampler']['Trainer']['learning_rate_decay'],
                  config['TagSampler']['Trainer']['weight_decay'],
                  config['TagSampler']['Trainer']['start_decay_at'])
    optim.set_parameters(model.parameters())
    return optim

def build_lr_scheduler(optimizer):
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
    lambda2 = lambda epoch: config['TagSampler']['Trainer']['learning_rate_decay'] ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                                  lr_lambda=[lambda2])
    return scheduler

def check_save_model_path(config):
    if not os.path.exists(config['TagSampler']['Trainer']['out_dir']):
        os.makedirs(config['TagSampler']['Trainer']['out_dir'])
        print('saving config file to %s ...'%(config['TagSampler']['Trainer']['out_dir']))
        # save config.yml
        shutil.copy(args.config, os.path.join(config['TagSampler']['Trainer']['out_dir'],'config.yml'))

#Jay edit
def train_model(model, train_data, fields, optim, lr_scheduler, start_epoch_at, device):

    train_iter = make_train_data_iter(train_data, config)

    train_loss = SamplerLossCompute(len(fields['tag'].vocab),fields['tag'].vocab.stoi[IO.PAD_WORD], device)

    if config['Misc']['use_cuda']:
        train_loss = train_loss.cuda()

    trainer = Trainer(  config,
                        model,
                        train_iter,
                        train_loss,
                        optim,
                        lr_scheduler,
                        device)

    num_train_epochs = config['TagSampler']['Trainer']['num_train_epochs']
    print('start training...')
    for step_epoch in  range(start_epoch_at+1, num_train_epochs):
        trainer.lr_scheduler.step()
        # 1. Train for one epoch on the training set.
        train_stats = trainer.train(step_epoch, report_func)

        trainer.save_per_epoch(step_epoch, out_dir=config['TagSampler']['Trainer']['out_dir'])

def main():

    # Load train and validate data.
    print("Loading train data")
    # Load fields generated from preprocess phase.
    fields = load_fields()


    train = SamplerDataset(
        data_path=args.train_data,
        fields=[('src', fields["src"]),
                ('tag', fields["tag"])])

    # Build model.
    model, start_epoch_at = build_or_load_model(config, fields)
    check_save_model_path(config)

    # Build optimizer.
    optim = build_optim(model, config)
    lr_scheduler = build_lr_scheduler(optim.optimizer)

    if config['Misc']['use_cuda']:
        model = model.cuda()

    # Do training.

    #Jay edit
    train_model(model, train, fields, optim, lr_scheduler, start_epoch_at, device)

if __name__ == '__main__':
    main()
