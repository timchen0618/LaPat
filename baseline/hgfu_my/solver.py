from comet_ml import Experiment
import time
import os
import yaml

import torch
import torch.nn as nn
from data import make_save_dir, read_json, DataUtils, tens2np
from model_builder import build_model
# from models import model



def cal_bleu(out_file, tgt_file):
    "Calculate BLEU score for the purpose of validation."
    from nltk.translate.bleu_score import corpus_bleu
    weights = []
    weights.append((1.0/1.0, ))
    weights.append((1.0/2.0, 1.0/2.0, ))
    weights.append((1.0/3.0, 1.0/3.0, 1.0/3.0, ))

    o_sen = (open(out_file, 'r')).readlines()
    t_sen = (open(tgt_file, 'r')).readlines()

    outcorpus = []
    for s in o_sen:
        outcorpus.append(s.split())
    tgtcorpus = []
    for s in t_sen:
        tgtcorpus.append([s.split()])

    bleus = []
    bleus.append(corpus_bleu(tgtcorpus, outcorpus, weights[0]))
    bleus.append(corpus_bleu(tgtcorpus, outcorpus, weights[1]))
    bleus.append(corpus_bleu(tgtcorpus, outcorpus, weights[2]))
    bleus.append(corpus_bleu(tgtcorpus, outcorpus, (1.0/4.0, 1.0/4.0, 1.0/4.0, 1.0/4.0, )))

    for i in range(4):
        print('bleu-%d: %6.6f'%(i+1, bleus[i]))

    return bleus


class Solver(object):
    """docstring for Solver"""
    def __init__(self, args):
        super(Solver, self).__init__()
        self.args = args
        self._save_checkpoints = args.save_checkpoints
        if args.train:
            self.model_dir = make_save_dir(os.path.join(args.model_path, args.exp_name))

        self._disable_comet = args.disable_comet
        self._model_name = args.model_name
        self._cuda = not args.no_cuda
        self.load_path = args.load_path

        stream = open(args.config, 'r') 
        config = yaml.load(stream, Loader=yaml.SafeLoader)
        self.config = config
        self.data_utils = DataUtils(config, args.train)
        self._prepare_model(config)
        self.batch_size = config['batch_size']
        self.num_epoch = config['num_epoch']
        self._print_every_step = config['print_every_step']
        self._valid_every_step = config['valid_every_step']
   
        # print('[Logging Info] Finish preparing model...')
        if self.args.test:
            self.outfile = open(os.path.join(args.pred_dir, args.model_name, args.prediction), 'w')

        
        
    def _prepare_model(self, config):
        print('[Logging] preparing model...')
        self.model = build_model(self._model_name, config['models'], self._cuda, self.data_utils)
        self.lr = config['lr']

    def load_checkpoint(self, path):
        state_dict = torch.load(path)['state_dict']
        self.model.load_state_dict(state_dict)

    def loss_compute(self, out, target, padding_idx):
        true_dist = out.data.clone()
        true_dist.fill_(0.)
        true_dist.scatter_(2, target.unsqueeze(2), 1.)
        true_dist[:,:,padding_idx] *= 0

        total = (target != padding_idx).sum(dim=1)
        
        # return -(true_dist*out).sum(dim=2).mean()
        return (-(true_dist*out).sum(dim=2).sum(dim=1)/total).mean()

    def _run_one_step(self, batch, step=None):
        batch['step'] = step
        ## Forwarding ## 
        out = self.model(batch).transpose(0, 1).contiguous()
        target = batch['y'].long()
        if self._cuda:
            target = target.cuda()
        loss = self.loss_compute(out, target, self.data_utils.pad)
        
        return loss

    def _generate_one_step(self, batch):
        outputs = self.model.generate(batch, max_length=self.data_utils._max_len, bos_token=self.data_utils.bos)
        return outputs

    def _first_instance_in_batch(self, batch):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v[:1]

        return batch

    def train(self):
        if not self._disable_comet:
            # logging
            COMET_PROJECT_NAME = 'weibo-baseline'
            COMET_WORKSPACE = 'timchen0618'

            self.exp = Experiment(project_name=COMET_PROJECT_NAME,
                                  workspace=COMET_WORKSPACE,
                                  auto_output_logging='simple',
                                  auto_metric_logging=None,
                                  display_summary=False,
                                 )
            self.exp.set_name(self.args.exp_name)
            self.exp.log_parameters(self.config)
            self.exp.log_parameters(self.config['models'][self._model_name])

        optim = torch.optim.Adam(self.model.parameters(), lr = self.lr, betas=(0.9, 0.98), eps=1e-9)
        print('[Logging Info] Finish loading data, start training...')

        step = 0
        losses = []
        for epoch in range(self.num_epoch):
            train_loader = self.data_utils.data_yielder()

            for batch in train_loader:
                self.model.train()
                optim.zero_grad()

                loss = self._run_one_step(batch, step)
                loss.backward()
                optim.step()
                losses.append(loss)

                if step % self._print_every_step == 0:
                    print('Logging...')
                    print('Step: %d | Loss: %f'%(step, sum(losses)/len(losses)))
                    print('Src: ', self.data_utils.id2sent(tens2np(batch['src'][0])))
                    print('length: ', batch['lengths'][0])
                    print('Tgt: ', self.data_utils.id2sent(tens2np(batch['y'][0])))
                    # print(tens2np(self._generate_one_step(self._first_instance_in_batch(batch))).shape)
                    print('Pred: ', self.data_utils.id2sent(tens2np(self._generate_one_step(self._first_instance_in_batch(batch)))))
                    if not self._disable_comet:
                        self.exp.log_metric('Train Loss', tens2np(sum(losses)/len(losses)), step=step)
                    losses = []

                if step % self._valid_every_step == self._valid_every_step - 1:
                    self.validate(step)

                step += 1

    @torch.no_grad()
    def validate(self, step):
        print('='*33)
        print('========== Validation ==========')
        print('='*33)
        fw = open(self.args.w_valid_file, 'w')

        self.model.eval()
        valid_loader = self.data_utils.data_yielder(valid=True)

        losses = []

        for batch in valid_loader:
            loss = self._run_one_step(batch, step)
            losses.append(loss)
            outputs = self._generate_one_step(batch)
            outputs = outputs.transpose(0, 1)

            # Writing sentences to hypothesis file
            for l in outputs:
                sentence = self.data_utils.id2sent(l[1:], True)
                fw.write(sentence)
                fw.write("\n")
        fw.close()

        print('Valid Loss: %4.6f'%(sum(losses)/len(losses)))

        # Calculate BLEU score and log to comet if needed
        bleus = cal_bleu(self.args.w_valid_file, self.args.w_valid_tgt_file)

        if not self._disable_comet:
            self.exp.log_metric('BLEU-1', bleus[0], step=step)
            self.exp.log_metric('BLEU-2', bleus[1], step=step)
            self.exp.log_metric('BLEU-3', bleus[2], step=step)
            self.exp.log_metric('BLEU-4', bleus[3], step=step)
            self.exp.log_metric('Valid Loss', sum(losses)/len(losses), step=step)

        if self._save_checkpoints:
            print('saving!!!!')

            model_name = str(int(step/1000)) + 'k_' + '%6.6f_'%(sum(losses)/len(losses)) + 'model.pth'
            state = {'step': step, 'state_dict': self.model.state_dict()}
            torch.save(state, os.path.join(self.model_dir, model_name))



    @torch.no_grad()
    def test(self):
        print('='*30)
        print('========== Testing ==========')
        print('='*30)

        self.load_checkpoint(self.load_path)
        self.model.eval()
        test_loader = self.data_utils.data_yielder()

        losses = []

        for i, batch in enumerate(test_loader):
            outputs = self._generate_one_step(batch)
            outputs = outputs.transpose(0, 1)
            print('outputs', outputs.size())
            if i % 20 == 0:
                print('step %d'%i)
            for l in outputs:
                self.outfile.write(self.data_utils.id2sent(l, test=True))
                self.outfile.write('\n')

        self.outfile.close()



