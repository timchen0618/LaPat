from __future__ import print_function
from comet_ml import Experiment
import os
import time


# # Suppress as many warnings as possible
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# from tensorflow.python.util import deprecation
# deprecation._PRINT_DEPRECATION_WARNINGS = False
# import tensorflow as tf
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# # Suppress as many warnings as possible

import torch
import torch.nn as nn

import yaml

from models import EncoderOut, make_transformer_model
from utils import DataUtils, make_save_dir, tens2np
from bert_utils import *
# from transformers import BertTokenizer, BertModel
from sequence_generator import SequenceGenerator



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

class Solver():
    '''
    Do training, validation and testing.
    '''
    def __init__(self, args):
        self.args = args
        with open(args.config, 'r') as stream:
            config = yaml.load(stream, Loader=yaml.SafeLoader)
            self.config = config[self.args.task]
        if args.model_type == 'transformer':
            self.data_utils = DataUtils(self.config, args.train, args.task)
        elif args.model_type == 'bert':
            assert args.task == 'seq2seq'
            self.data_utils = bert_utils(self.config, args.train, args.task)
        if args.train and args.save_checkpoints:
            self.model_dir = make_save_dir(os.path.join(args.model_dir, args.task, args.exp_name))
        self._disable_comet = args.disable_comet
        self._model_type = args.model_type
        self._save_checkpoints = args.save_checkpoints

        ###### loading .... ######
        print("====================")
        print("start to build model")
        print('====================')
        vocab_size = self.data_utils.vocab_size
        print("Vocab Size: %d"%(vocab_size))
        self.model = self.make_model(src_vocab=vocab_size, 
                                     tgt_vocab=vocab_size, 
                                     config=self.config['model'])

    def make_model(self, src_vocab, tgt_vocab, config):

        "Helper: Construct a model from hyperparameters."
        if self._model_type == 'transformer':
            model = make_transformer_model(src_vocab, tgt_vocab,  config)
        elif self._model_type == 'bert':
            tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', padding_side='left')
            num_added_tokens = tokenizer.add_tokens(self.data_utils.all_tokens)
            print('We have added %d tokens to the bert tokenizer.'%num_added_tokens)
            self.data_utils.set_tokenizer(tokenizer)
            model = BERT(BertModel.from_pretrained('bert-base-chinese'), self.config['max_len'], config['d_bert'], self.data_utils.vocab_size)

        return model.cuda()


    def train(self):
        if not self._disable_comet:
            # logging
            COMET_PROJECT_NAME = 'weibo-stc'
            COMET_WORKSPACE = 'timchen0618'

            self.exp = Experiment(project_name=COMET_PROJECT_NAME,
                                  workspace=COMET_WORKSPACE,
                                  auto_output_logging='simple',
                                  auto_metric_logging=None,
                                  display_summary=False,
                                 )

            self.exp.add_tag(self.args.task)
            if self.args.task != 'pure_seq2seq':
                if self.args.processed:
                    self.exp.add_tag('processed')
                else:
                    self.exp.add_tag('unprocessed')
            if self.args.sampler_label != 'none':
                self.exp.add_tag(self.args.sampler_label)
            if self._model_type == 'bert':
                self.exp.add_tag('BERT')

            self.exp.set_name(self.args.exp_name)
            self.exp.log_parameters(self.config)
            self.exp.log_parameters(self.config['model'])


        # if finetune, load pretrain
        if self.args.task == 'finetune':
            lr = 5e-7
            state_dict = torch.load(self.args.load_model)['state_dict']
            print('loading model from %s ...'%self.args.load_model)
            self.model.load_state_dict(state_dict)
        else:
            lr = self.config['lr_init']
            if self.args.load_model is not None:
                state_dict = torch.load(self.args.load_model, map_location='cuda:%d'%self.args.gpuid)['state_dict']
                print('loading model from %s ...'%self.args.load_model)
                self.model.load_state_dict(state_dict)

        if self.args.pretrain_embedding:
            self.model.load_embedding(self.args.pretrain_embedding)


        # Optimizer and some info for logging.
        if self.config['optimizer'] == 'adam':
            optim = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=0)
        elif self.config['optimizer'] == 'adamw':
            optim = torch.optim.AdamW(self.model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
        else:
            raise NotImplementedError

        total_loss = []
        p_gen_list = []
        start = time.time()
        step = self.args.start_step
        print('starting from step %d'%step)


        for epoch in range(self.config['num_epoch']):
            self.model.train()
            train_data = self.data_utils.data_yielder(valid=False)

            for batch in train_data:
                # print('-'*30)
                # Whether do noam learning rate scheduling
                if self.config['noam_decay']:
                    if step % 5 == 1:
                        lr = self.config['lr'] * (1/(self.config['model']['d_model']**0.5))*min((1/(step)**0.5), (step) * (1/(self.config['warmup_steps']**1.5)))
                        if self.args.task == 'finetune':
                            lr /= self.config['lr_decay']
                        for param_group in optim.param_groups:
                            param_group['lr'] = lr

                tgt_mask = batch['tgt_mask'].long()
                y = batch['y'].long()

                if self._model_type == 'bert':
                    inp = batch['src']['input_ids'].cuda()
                    out = self.model.forward(inp)
                    pred = tens2np(out.topk(1, dim=-1)[1].squeeze())
                    p_gen_list.append(0.0)
                else:
                    tgt = batch['tgt'].long()
                    src = batch['src'].long()
                    src_mask = batch['src_mask'].long()

                    # Forwarding (with mask or not)
                    if self.config['pos_masking']:
                        out, p_gen = self.model.forward_with_mask(src, tgt, src_mask, tgt_mask, batch['posmask'])
                    elif self.args.task == 'joint_gen' and self.config['greedy']:
                        out = self.model.forward_with_ss(src, src_mask, tgt, self.config['max_decode_step'], self.data_utils.bos)
                        # print('out', out.size())
                        p_gen = torch.zeros((1,1))
                    else:
                        out, p_gen = self.model.forward(src, tgt, src_mask, tgt_mask)

                    # Info for printing
                    pred = tens2np(out.topk(1, dim=-1)[1].squeeze())
                    p_gen = p_gen.mean()
                    p_gen_list.append(p_gen.item())


                loss = self.model.loss_compute(out, y, self.data_utils.pad)
                loss.backward()

                optim.step()
                optim.zero_grad()
                total_loss.append(tens2np(loss))

                # print out info
                if step % self.config['print_every_step'] == 0:
                    elapsed = time.time() - start
                    print("Epoch Step: %d Loss: %f  P_gen:%f Time: %f Lr: %4.6f"
                         % (step, np.mean(total_loss),
                         sum(p_gen_list)/len(p_gen_list), elapsed, lr))

                    if self._model_type == 'bert':
                        source_text = tens2np(inp.long())
                        target_text = tens2np(batch['y'].long())
                    elif self._model_type == 'transformer':
                        source_text = tens2np(batch['src'].long())
                        target_text = tens2np(batch['tgt'].long())

                    print('src:', self.data_utils.id2sent(source_text[0]))
                    print('tgt:', self.data_utils.id2sent(target_text[0]))
                    print('pred:', self.data_utils.id2sent(pred[0]))

                    # If using transformer, we want to see greedy decoding result
                    if self._model_type == 'transformer':
                        if self.config['pos_masking']:
                            greedy_text = self.model.greedy_decode(
                                                                    src.long()[:1],
                                                                    src_mask[:1],
                                                                    self.config['max_len'],
                                                                    self.data_utils.bos,
                                                                    batch['posmask'][:1]
                                                                  )
                        else:
                            greedy_text = self.model.greedy_decode(
                                                                    src.long()[:1],
                                                                    src_mask[:1],
                                                                    self.config['max_len'],
                                                                    self.data_utils.bos
                                                                  )
                        greedy_text = tens2np(greedy_text)
                        print('pred_greedy:',self.data_utils.id2sent(greedy_text[0]))

                    # logging statistics
                    if not self._disable_comet:
                        self.exp.log_metric('Train Loss', np.mean(total_loss), step=step)
                        self.exp.log_metric('Lr', lr, step=step)
                    print()
                    start = time.time()
                    total_loss = []
                    p_gen_list = []
                
                # Do validation 
                if step % self.config['valid_every_step'] == self.config['valid_every_step']-1:
                    self.validate(step)

                step += 1

    @torch.no_grad()
    def validate(self, step):
        print('*********************************')
        print('            Validation           ')
        print('*********************************')
        fw = open(self.args.w_valid_file, 'w')
        val_yielder = self.data_utils.data_yielder(valid=True)
        self.model.eval()
        total_loss = []

        # Validate one batch, writing valid hypothesis to file
        for batch in val_yielder:
            if self._model_type == 'bert':
                inp = batch['src']['input_ids'].cuda()
                out = self.model.forward(inp)
            else:
                # model is transformer
                batch['src'] = batch['src'].long()
                batch['tgt'] = batch['tgt'].long()

                if self.config['pos_masking']:
                    out, _ = self.model.forward_with_mask(batch['src'], batch['tgt'], batch['src_mask'],
                                                          batch['tgt_mask'], batch['posmask'])
                else:
                    out, _ = self.model.forward(batch['src'], batch['tgt'],
                                                batch['src_mask'], batch['tgt_mask'])

            loss = self.model.loss_compute(out, batch['y'].long(), self.data_utils.pad)
            total_loss.append(loss.item())

            if self.config['pos_masking']:
                out = self.model.greedy_decode(batch['src'].long(), batch['src_mask'],
                                self.config['max_len'], self.data_utils.bos, batch['posmask'])
            else:
                out = self.model.greedy_decode(batch['src'].long(), batch['src_mask'],
                                self.config['max_len'], self.data_utils.bos)

            # Writing sentences to hypothesis file
            for l in out:
                sentence = self.data_utils.id2sent(l[1:], True)
                fw.write(sentence)
                fw.write("\n")

        fw.close()

        # Calculate BLEU score and log to comet if needed
        bleus = cal_bleu(self.args.w_valid_file, self.args.w_valid_tgt_file)
        if not self._disable_comet:
            self.exp.log_metric('BLEU-1', bleus[0], step=step)
            self.exp.log_metric('BLEU-2', bleus[1], step=step)
            self.exp.log_metric('BLEU-3', bleus[2], step=step)
            self.exp.log_metric('BLEU-4', bleus[3], step=step)

            self.exp.log_metric('Valid Loss', sum(total_loss)/ len(total_loss), step=step)

        print('=============================================')
        print('Validation Result -> Loss : %6.6f' %(sum(total_loss)/len(total_loss)))
        print('=============================================')
        self.model.train()

        # Saving model checkpoints
        if self._save_checkpoints:
            print('saving!!!!')

            model_name = str(int(step/1000)) + 'k_' + '%6.6f__%4.4f_%4.4f_'%(sum(total_loss)/len(total_loss), bleus[0], bleus[3]) + 'model.pth'
            state = {'step': step, 'state_dict': self.model.state_dict()}
            torch.save(state, os.path.join(self.model_dir, model_name))


    @torch.no_grad()
    def test(self):
        # Prepare model
        path = self.args.load_model
        state_dict = torch.load(path)['state_dict']

        self.model.load_state_dict(state_dict)

        # file path for prediction
        pred_dir = make_save_dir(self.args.pred_dir)
        filename = self.args.filename
        outfile = open(os.path.join(pred_dir, self.args.task, filename), 'w')

        # Start decoding
        data_yielder = self.data_utils.data_yielder()
        total_loss = []
        start = time.time()


        # If beam search, create sequence generator object
        self._beam_search = self.config['eval']['beam_size'] > 1
        # self._beam_search = True
        if self._beam_search:
            seq_gen = SequenceGenerator(self.model, self.data_utils, beam_size=self.config['eval']['beam_size'], no_repeat_ngram_size=self.config['eval']['block_ngram'])

        self.model.eval()
        step = 0

        # Run one batch
        for batch in data_yielder:
            step += 1
            if step % 10 == 1:
                print('Step ', step)

            # Decoding according to scheme
            if self._beam_search:
                out = seq_gen.generate(batch, pos_masking=self.config['pos_masking'], bos_token=self.data_utils.bos)
            else:
                max_length = self.config['max_len']
                if self.config['pos_masking']:
                    out = self.model.greedy_decode(batch['src'].long(), batch['src_mask'],
                                max_length, self.data_utils.bos, batch['posmask'])
                else:
                    if self.args.task == 'joint_gen':
                        max_length = self.config['max_decode_step']
                    out = self.model.greedy_decode(batch['src'].long(), batch['src_mask'],
                                max_length, self.data_utils.bos)

            # Write sentences to file
            for l in out:
                if self._beam_search:
                    sentence = self.data_utils.id2sent(l[0]['tokens'][:-1], True)
                else:
                    sentence = self.data_utils.id2sent(l[1:], True)
                outfile.write(sentence)
                outfile.write("\n")

        outfile.close()
