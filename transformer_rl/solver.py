from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import subprocess
from models import *
from utils import *
from attention import *
from torch.distributions.categorical import Categorical
import logging
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from predpos_utils import POSDataUtils
from sequence_generator import SequenceGenerator
# from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.ERROR)

def cal_bleu(out_file, tgt_file):

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
    bleus.append(corpus_bleu(tgtcorpus, outcorpus,(1.0/4.0, 1.0/4.0, 1.0/4.0, 1.0/4.0, )))

    for i in range(4):
        print('bleu-%d: %6.6f'%(i+1, bleus[i]))

    return bleus

def cal_sent_bleu(pred, tgt):
    weights = []
    weights.append((1.0/1.0, ))
    weights.append((1.0/2.0, 1.0/2.0, ))
    weights.append((1.0/3.0, 1.0/3.0, 1.0/3.0, ))
    weights.append((1.0/4.0, 1.0/4.0, 1.0/4.0, 1.0/4.0, ))

    # return (sentence_bleu(tgt, pred, weights[0]), sentence_bleu(tgt, pred, weights[1]), sentence_bleu(tgt, pred, weights[2]), sentence_bleu(tgt, pred, weights[3]))
    return (sentence_bleu(tgt, pred, weights[0]), sentence_bleu(tgt, pred, weights[3]))

class Solver():
    def __init__(self, args):
        self.args = args
        if self.args.pred_pos:
            self.data_utils = POSDataUtils(args)
        else:
            self.data_utils = data_utils(args)
        if args.train:
            self.model_dir = make_save_dir(os.path.join(args.model_dir, args.exp_name))
        self.vocab = None
        self.mean_reward = 0.0

    def make_model(self, src_vocab, tgt_vocab, N=6, 
        d_model=512, d_ff=2048, h=8, dropout=0.1, pointer_gen = True, num_classes=498, gdrop = 0.1, pred_pos_drop=0.3, g_hidden=512):
        
        "Helper: Construct a model from hyperparameters."
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        word_embed = nn.Sequential(Embeddings(d_model, src_vocab), c(position))
        model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                                 c(ff), dropout), N, d_model, tgt_vocab, pointer_gen),
            word_embed,
            word_embed)

        # sampler model
        if self.args.pred_pos:
            sampler = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), pred_pos_drop), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                                 c(ff), pred_pos_drop), N, d_model, tgt_vocab, pointer_gen),
            word_embed,
            word_embed)
        else:
            sampler = Sampler(Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N), word_embed, Generator(d_model, self.args.num_classes, gdrop, g_hidden))
        # This was important from their code. 
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        sampler_dict = torch.load(self.args.sampler)['state_dict'] 
        transformer_dict = torch.load(self.args.transformer)['state_dict'] 
        sampler.load_state_dict(sampler_dict)
        model.load_state_dict(transformer_dict)
        # print('loading state_dict...')
        return model.cuda(), sampler.cuda() 

    def compute_f1_reward(self, out, y, infer, padding_idx):
        rewards = []
        f1 = []
        acc = []
        alpha = 1

        for i in range(infer.size(0)):
            inf = infer[i].tolist()
            gt = y[i].tolist()
            reward = self.get_f1_score(inf, gt) + alpha * self.get_accuracy(out[i],y[i],padding_idx)
            # reward = self.get_f1_score(inf, gt)
            rewards.append(reward)
            f1.append(self.get_f1_score(inf, gt))
            acc.append(self.get_accuracy(out[i],y[i],padding_idx).detach().cpu().numpy())

        final_reward = max(rewards)
        if final_reward < 0.08:
            final_reward = -0.1

        return torch.Tensor([final_reward]).cuda(), sum(f1)/ len(f1), sum(acc)/len(acc)


    def compute_bleu_reward(self, out, y, infer, padding_idx):
        bleu = []

        assert infer.size(0) != 0
        for i in range(infer.size(0)):
            inf = infer[i].tolist()
            gt = y[i].tolist()

            reward = cal_sent_bleu(inf, [gt])
            reward = reward[0] + reward[1]
            bleu.append(reward)

        final_reward = max(bleu)
        if final_reward < 0.1:
            final_reward = -0.1

        return torch.Tensor([final_reward]).cuda(), sum(bleu)/len(bleu)

    def get_f1_score(self, infer_words, ground_words):
        infer_set = set(infer_words)
        ground_set = set(ground_words)

        intersect = ground_set.intersection(infer_set)
        precision = len(intersect)/len(infer_set)
        recall = len(intersect)/len(ground_set)
        if precision == 0.0 or recall == 0.0:
            f1_score = 0.0
        else:
            f1_score = 2.0*(precision*recall) / (precision+recall)

        return f1_score

    def get_accuracy(self,out,target, padding_idx):
        pred = out.argmax(dim=-1)

        mask = (target != padding_idx)
        corr = (pred == target) * mask 
        # print('pred', pred.tolist())
        # print('target', target.tolist())
        corr = corr.sum().float()
        total = mask.sum()
        # print('acc', corr/total)
        # pred = output.max(1)[1]
        # non_padding = target.ne(self.padding_idx).view(batch_size,time_step)
        # num_correct = pred.eq(target)
        # num_correct = num_correct.view(batch_size,time_step) * mask
        return corr/total
        
        # return num_correct.sum(1).float()/non_padding.sum(1).float()

    def train(self):
        # logging
        COMET_PROJECT_NAME='weibo-stc'
        COMET_WORKSPACE='timchen0618'

        if not self.args.disable_comet:
            self.exp = Experiment(project_name=COMET_PROJECT_NAME,
                                  workspace=COMET_WORKSPACE,
                                  auto_output_logging='simple',
                                  auto_metric_logging=None,
                                  display_summary=False,
                                  )

            self.exp.add_tag(self.args.sampler_label)
            self.exp.add_tag('RL')
            if self.args.processed:
                self.exp.add_tag('processed')
            else:
                self.exp.add_tag('unprocessed')
            self.exp.set_name(self.args.exp_name)

        ###### loading .... ######
        train_data = self.data_utils.data_yielder(valid=False)
        vocab_size = self.data_utils.vocab_size
        print("============================")  
        print("=======start to build=======")
        print("============================") 
        print("Vocab Size: %d"%(vocab_size))

        #make model
        self.transformer, self.sampler = self.make_model(src_vocab = vocab_size, 
                                                        tgt_vocab = vocab_size, 
                                                        N = self.args.num_layer, 
                                                        dropout = self.args.dropout, 
                                                        pointer_gen = self.args.pointer_gen,
                                                        gdrop = self.args.gdrop,
                                                        pred_pos_drop = self.args.pred_pos_drop,
                                                        g_hidden = self.args.g_hidden
                                                        )

        
        opt_sampler = torch.optim.Adam(self.sampler.parameters(), lr=5e-6, betas=(0.9, 0.98), eps=1e-9)
        opt_trans = torch.optim.Adam(self.transformer.parameters(), lr=5e-6, betas=(0.9, 0.98), eps=1e-9)#get_std_opt(self.transformer)
        total_loss = []
        sampler_losses = []
        p_gen_list = []
        rewards = []
        sampler_accs = []
        reward_buffer = []
        reward_epoch = []
        f1s = []
        accs = []
        bleus = []
        start = time.time()

        step = self.args.start_step
        print('starting from step %d...'%step)
        for epoch in range(self.args.num_epoch):
            self.transformer.train()
            self.sampler.train()
            train_data = self.data_utils.data_yielder(valid=False)
            for batch in train_data:
                with torch.autograd.set_detect_anomaly(True):
                    opt_trans.zero_grad()
                    opt_sampler.zero_grad()

                    # convert data type
                    src = batch['src'].long()
                    src_pos = src.clone()
                    tgt = batch['tgt'].long()
                    src_mask = batch['src_mask'].long()
                    tgt_mask = batch['tgt_mask'].long()
                    y = batch['y'].long()

                    if self.args.pred_pos:
                        pos_tgt = batch['pos_tgt'].long()
                        pos_tgt_mask = batch['pos_tgt_mask'].long()

                        # predict POS
                        pos_out, _ = self.sampler.forward(src, pos_tgt, src_mask, pos_tgt_mask)
                        b_size, src_len, dim = pos_out.size()
                        sampled = torch.multinomial(torch.exp(pos_out).view(-1, pos_out.size(-1)), 1)
                        sampled = sampled.view(b_size, src_len)
                        # m = Categorical(probs=torch.exp(out).view(-1, out.size(-1)))
                        # infer = (m.sample()).view(b_size, src_len)

                        # create pos-informed data
                        src_pos = torch.cat((src, sampled), dim = 1).detach()


                    else:
                        sampler_label = batch['sampler_label'].long()
                        # forward sampler model (create POS)
                        pos_out = self.sampler.forward(src, src_mask)
                        sampled_idx = torch.multinomial(torch.exp(pos_out), 1)
                        # print('sample_idx', sampled_idx.size())
                        # selected_tag_logprob = pos_out[sampled_idx]
                        selected_tag_logprob = torch.index_select(pos_out, 1, sampled_idx.squeeze())
                        # selected_tag_logprob, sampled_idx = torch.max(pos_out, dim=-1)
                        
                        sample_acc = (sampler_label == sampled_idx).sum()/ sampler_label.fill_(1).sum().float()
                        sampler_accs.append(sample_acc.detach().cpu().numpy())

                        # create pos-informed data
                        for i in range(sampled_idx.size(0)):
                            pos = self.data_utils.pos_dict['idx2structure'][int(sampled_idx[i])]
                            pos = ['<' + l + '>' for l in pos.strip().split()]
                            src_pos[i] = self.data_utils.addpos2src(pos, src_pos[i], self.data_utils.src_max_len)
                            

                    if self.args.pos_masking:
                        posmask = torch.zeros((sampled_idx.size(0), self.args.max_len, self.data_utils.vocab_size)).cuda()
                        posmask[:, :, self.data_utils.pad] = 1
                        for i in range(sampled_idx.size(0)):
                            if len(pos) > self.args.max_len:
                                pos = pos[:self.args.max_len]
                            ### pos_masking ###
                            for j, p in enumerate(pos):
                                posmask[i, j] = self.data_utils.pos2mask[p]
                            batch['posmask'] = posmask
                            ### pos_masking ###

                    rl_src_mask = (src_pos != self.data_utils.pad).unsqueeze(1)
                    if self.args.pos_masking:
                        out, p_gen = self.transformer.forward_with_mask(src_pos.detach(), tgt, rl_src_mask, tgt_mask, batch['posmask'])
                    else:
                        out, p_gen = self.transformer.forward(src_pos.detach(), tgt, rl_src_mask, tgt_mask)

                    #print out info
                    p_gen = p_gen.mean()
                    p_gen_list.append(p_gen.item())
                    pred = out.topk(1, dim=-1)[1].squeeze(-1).detach().cpu().numpy()
                    gg = batch['src'].long().detach().cpu().numpy()
                    tt = batch['tgt'].long().detach().cpu().numpy()
                    yy = batch['y'].long().detach().cpu().numpy()


                    # infer = out.topk(1, dim=-1)[1].squeeze()
                    # infer = self.transformer.greedy_decode(src_pos, src_mask, self.data_utils.max_len, 1)
                    b_size, src_len, emb_dim = out.size()
                    m = Categorical(probs=out.view(-1, out.size(-1)))
                    infer = (m.sample()).view(b_size, src_len)
                    inf = infer.long().detach().cpu().numpy()
                    
                    # compute loss
                    seq2seq_loss = self.transformer.loss_compute(out, y, self.data_utils.pad)
                    if self.args.reward_type == 'bleu':
                        final_reward, bleu = self.compute_bleu_reward(out, y, infer, self.data_utils.pad)
                    elif self.args.reward_type == 'f1':
                        final_reward, f1, acc = self.compute_f1_reward(out, y, infer, self.data_utils.pad)
                    else:
                        raise NotImplementedError

                    final_reward.requires_grad=False

                    if self.args.pred_pos:
                        true_dist = pos_out.data.clone()
                        true_dist.fill_(0.)
                        true_dist.scatter_(2, y.unsqueeze(2), 1.)
                        true_dist[:,:,self.data_utils.pad] *= 0
                        logprob = (true_dist*pos_out).sum(dim=2).sum(dim=1)
                        # print('logprob', logprob.size())
                        # print('final_reward', final_reward.size())
                        # print(logprob)
                        sampler_loss = -logprob * (final_reward - self.mean_reward)
                    else:
                        sampler_loss = -selected_tag_logprob * (final_reward - self.mean_reward)

                    sampler_loss = sampler_loss.mean()

                    sampler_loss.backward(retain_graph=True)
                    opt_sampler.step()
                    seq2seq_loss.backward(retain_graph=True)
                    opt_trans.step()

                    rewards.append(final_reward.mean().detach().cpu().numpy())
                    total_loss.append(seq2seq_loss.detach().cpu().numpy())
                    sampler_losses.append(sampler_loss.detach().cpu().numpy())
                    reward_epoch.append(final_reward.mean().detach().cpu().numpy())
                    if self.args.reward_type == 'bleu':
                        bleus.append(bleu)
                    elif self.args.reward_type == 'f1':
                        f1s.append(f1)
                        accs.append(acc)

                if step % self.args.print_every_steps == 1:
                    elapsed = time.time() - start
                    if self.args.reward_type == 'bleu':
                        print("[Step: %d] Loss: %f Sampler_Loss: %f Reward: %f BLEU: %4.4f P_gen: %f Time: %f MeanR: %4.5f" %
                                (step, np.mean(total_loss), np.mean(sampler_losses), np.mean(rewards), sum(bleus)/len(bleus), sum(p_gen_list)/len(p_gen_list), elapsed, self.mean_reward))
                    elif self.args.reward_type == 'f1':
                        print("[Step: %d] Loss: %f Sampler_Loss: %f Reward: %f F1: %4.4f Acc: %4.4f P_gen: %f Time: %f MeanR: %4.5f" %
                                (step, np.mean(total_loss), np.mean(sampler_losses), np.mean(rewards), sum(f1s)/len(f1s), sum(accs)/len(accs), sum(p_gen_list)/len(p_gen_list), elapsed, self.mean_reward))
                    print('src:',self.id2sent(self.data_utils, gg[0]))
                    print('tgt:',self.id2sent(self.data_utils, yy[0]))
                    print('pred:',self.id2sent(self.data_utils, pred[0]))
                    print('sampled:', self.id2sent(self.data_utils, inf[0]))

                    if self.args.pos_masking:
                        pp =  self.transformer.greedy_decode(src_pos.long()[:1], rl_src_mask[:1], 14, self.data_utils.bos, batch['posmask'][:1])
                    else:
                        pp =  self.transformer.greedy_decode(src_pos.long()[:1], rl_src_mask[:1], 14, self.data_utils.bos)
                    pp = pp.detach().cpu().numpy()
                    print('pred_greedy:',self.id2sent(self.data_utils, pp[0]))
     
                    if not self.args.disable_comet:
                        self.exp.log_metric('Train Loss', np.mean(total_loss) ,step=step)
                        self.exp.log_metric('Sampler Loss', np.mean(sampler_losses) ,step=step)
                        self.exp.log_metric('Reward', np.mean(rewards) ,step=step)
                        if not self.args.pred_pos:
                            self.exp.log_metric('Sampler Accuracy', np.mean(sampler_accs), step=step)
                        self.exp.log_metric('F1', sum(f1s)/ len(f1s))
                        self.exp.log_metric('Acc', sum(accs)/len(accs))
                    print()
                    start = time.time()
                    total_loss = []
                    sampler_losses = []
                    rewards = []
                    p_gen_list = []
                    sampler_accs = []
                    f1s = []
                    accs = []
                    bleus = []

                if step % self.args.valid_every_steps == self.args.valid_every_steps - 1:
                    self.validate(step)
                
                    reward_buffer.append(sum(reward_epoch)/len(reward_epoch))
                    if len(reward_buffer) > 10:
                        reward_buffer = reward_buffer[1:]
                        assert len(reward_buffer) == 10
                    self.mean_reward = sum(reward_buffer)/len(reward_buffer)
                    print('='*50)
                    print('Loading mean reward, value: %f'%self.mean_reward)
                    print('='*50)
                    reward_epoch = []

                step += 1

    @torch.no_grad()
    def validate(self, step):
        print('-'*30)
        print('--------Validation--------')
        print('-'*30)
        val_yielder = self.data_utils.data_yielder(valid=True)
        self.transformer.eval()
        self.sampler.eval()
        total_loss = []

        fw = open(self.args.w_valid_file, 'w')
        for batch in val_yielder:
            src = batch['src'].long()
            tgt = batch['tgt'].long()
            src_pos = batch['src'].long().data.clone().cuda()
            with torch.no_grad():
                # sampler model
                if self.args.pred_pos:
                    pos_tgt = batch['pos_tgt'].long()
                    # predict POS
                    pos_out, _ = self.sampler.forward(src, pos_tgt, batch['src_mask'], batch['pos_tgt_mask'])
                    pos_out = self.sampler.greedy_decode(src, batch['src_mask'], self.data_utils.max_len, self.data_utils.bos)

                    print('pos', pos_out.size())
                    # create pos-informed data
                    print('src', src.size())
                    src_pos = torch.cat((src, pos_out[:,1:]), dim = 1).detach()
                    print('src_pos', src_pos.size())
                else:
                    out = self.sampler.forward(src, batch['src_mask'])
                    selected_tag_logprob, sampled_idx = torch.max(out, dim=-1)

                if self.args.pos_masking:
                    posmask = torch.zeros((sampled_idx.size(0), self.args.max_len, self.data_utils.vocab_size)).cuda()
                    posmask[:, :, self.data_utils.pad] = 1
                
                if not self.args.pred_pos:
                    # create pos-informed data
                    for i in range(sampled_idx.size(0)):
                        pos = self.data_utils.pos_dict['idx2structure'][int(sampled_idx[i])]
                        pos = ['<' + l + '>' for l in pos.strip().split()]
                        src_pos[i] = self.data_utils.addpos2src(pos, src[i], self.data_utils.src_max_len)

                        if self.args.pos_masking:
                            if len(pos) > self.args.max_len:
                                pos = pos[:self.args.max_len]
                            ### pos_masking ###
                            for j, p in enumerate(pos):
                                posmask[i, j] = self.data_utils.pos2mask[p]
                            batch['posmask'] = posmask
                            ### pos_masking ###

                rl_src_mask = (src_pos != self.data_utils.pad).unsqueeze(1)
                out, _ = self.transformer.forward(src_pos, tgt, 
                        rl_src_mask, batch['tgt_mask'])
                loss = self.transformer.loss_compute(out, batch['y'].long(), self.data_utils.pad)
                total_loss.append(loss.item())

                if self.args.pos_masking:
                    out = self.transformer.greedy_decode(src_pos, rl_src_mask, self.data_utils.max_len, self.data_utils.bos, batch['posmask'])
                else:
                    out = self.transformer.greedy_decode(src_pos, rl_src_mask, self.data_utils.max_len, self.data_utils.bos)

                for i, l in enumerate(out):
                    sentence = self.data_utils.id2sent(l[1:], True)
                    fw.write(sentence)
                    fw.write("\n")
        fw.close()

        logging.debug('comparing %s and %s'%(self.args.w_valid_file, self.args.w_valid_tgt_file))
        bleus = cal_bleu(self.args.w_valid_file, self.args.w_valid_tgt_file)
        if not self.args.disable_comet:
            self.exp.log_metric('BLEU-1', bleus[0], step=step)
            self.exp.log_metric('BLEU-2', bleus[1], step=step)
            self.exp.log_metric('BLEU-3', bleus[2], step=step)
            self.exp.log_metric('BLEU-4', bleus[3], step=step)


        print('=============================================')
        print('Validation Result -> Loss : %6.6f' %(sum(total_loss)/len(total_loss)))
        print('=============================================')
        self.transformer.train()
        self.sampler.train()
        # self.log.add_scalar('Loss/valid', sum(total_loss)/len(total_loss), step)
        if not self.args.disable_comet:
            self.exp.log_metric('Valid Loss', sum(total_loss)/len(total_loss) ,step=step)

        if self.args.save_checkpoints:
            print('saving!!!!')
            if self.args.save_best_only:
                # sampler_name = str(int(step/10000)) + 'w_' + '%6.6f_%4.4f_%4.4f_'%(sum(total_loss)/len(total_loss), bleus[0], bleus[3]) + 'sampler.pth'
                state = {'step': step, 'state_dict': self.sampler.state_dict()}
                torch.save(state, os.path.join(self.model_dir, 'sampler.best.pth'))
                # transformer_name = str(int(step/10000)) + 'w_' + '%6.6f_%4.4f_%4.4f_'%(sum(total_loss)/len(total_loss), bleus[0], bleus[3]) + 'transformer.pth'
                state = {'step': step, 'state_dict': self.transformer.state_dict()}
                torch.save(state, os.path.join(self.model_dir, 'transformer.best.pth'))
            else:
                sampler_name = str(int(step/1000)) + 'k_' + '%6.6f_%4.4f_%4.4f_'%(sum(total_loss)/len(total_loss), bleus[0], bleus[3]) + 'sampler.pth'
                state = {'step': step, 'state_dict': self.sampler.state_dict()}
                torch.save(state, os.path.join(self.model_dir, sampler_name))

                transformer_name = str(int(step/1000)) + 'k_' + '%6.6f_%4.4f_%4.4f_'%(sum(total_loss)/len(total_loss), bleus[0], bleus[3]) + 'transformer.pth'
                state = {'step': step, 'state_dict': self.transformer.state_dict()}
                torch.save(state, os.path.join(self.model_dir, transformer_name))

    @torch.no_grad()
    def test(self):
        #prepare model
        path = self.args.transformer
        max_len = self.args.max_len
        state_dict = torch.load(path)['state_dict']
        vocab_size = self.data_utils.vocab_size

        self.transformer, self.sampler = self.make_model(src_vocab = vocab_size, 
                                     tgt_vocab = vocab_size, 
                                     N = self.args.num_layer, 
                                     dropout = self.args.dropout, 
                                     pointer_gen = self.args.pointer_gen,
                                     gdrop = self.args.gdrop,
                                     pred_pos_drop = self.args.pred_pos_drop,
                                     g_hidden = self.args.g_hidden
                                     )
        self.transformer.load_state_dict(state_dict)

        sampler_path = self.args.sampler
        sampler_state_dict = torch.load(sampler_path)['state_dict']
        self.sampler.load_state_dict(sampler_state_dict)

        pred_dir = make_save_dir(self.args.pred_dir)
        filename = self.args.filename

        # print('-----------')
        #start decoding
        data_yielder = self.data_utils.data_yielder(valid=False)
        total_loss = []
        start = time.time()

        #file
        f = open(os.path.join(pred_dir, filename), 'w')
        f_pos = open(os.path.join(pred_dir, self.args.pos_file), 'w')

        self.transformer.eval()
        self.sampler.eval()
        step = 0
        sampler_accs = []
        total_loss = []
        for batch in data_yielder:
            #print(batch['src'].data.size())
            pos_list = []
            step += 1
            # if step % 10 == 1:
            #     print('Step ', step)
            src = batch['src'].long()
            src_pos = src.clone()
            src_mask = batch['src_mask'].long()
            if not self.args.pred_pos:
                sampler_label = batch['sampler_label'].long()
                # tgt = batch['tgt'].long()
            pos = None

            # sampler model
            if self.args.pred_pos:
                pos_out = self.sampler.greedy_decode(src, src_mask, self.data_utils.max_len, self.data_utils.bos)
                src_pos = torch.cat((src, pos_out[:,1:]), dim = 1).detach()

            else:
                out = self.sampler.forward(src, src_mask)
                # choose top10 or top1
                selected_tag_logprob, sampled_idx = torch.max(out, dim=-1)
                # selected_tag_logprob, sampled_idx = torch.topk(out, 10, dim=-1)

                sampled_idx = sampled_idx.unsqueeze(1)
                # print('sampled_idx', sampled_idx.size())
                sample_acc = (sampler_label == sampled_idx[:,0]).sum()/ sampler_label.fill_(1).sum().float()
                sampler_accs.append(sample_acc.detach().cpu().numpy())

                # for j in range(sampled_idx.size(1)):
                sampled_idx = sampled_idx.squeeze(1)
                # create pos-informed data
                s = sampled_idx
                # print('s %d'%j, s)
                for i in range(sampled_idx.size(0)):
                    # pos = self.data_utils.pos_dict['idx2structure'][int(sampled_idx[i])]
                    pos = self.data_utils.pos_dict['idx2structure'][int(s[i])]
                    # print(pos)
                    f_pos.write(pos)
                    f_pos.write('\n')
                    pos = ['<' + l + '>' for l in pos.strip().split()]

                    pos_list.append(pos)
                    # print('pos', pos)
                    src_pos[i] = self.data_utils.addpos2src(pos, src_pos[i], self.data_utils.src_max_len)
                
                if self.args.pos_masking:
                    posmask = torch.zeros((sampled_idx.size(0), self.args.max_len, self.data_utils.vocab_size)).cuda()
                    posmask[:, :, self.data_utils.pad] = 1

                    if len(pos) > self.args.max_len:
                        pos = pos[:self.args.max_len]
                    ### pos_masking ###
                    for j, p in enumerate(pos):
                        posmask[i, j] = self.data_utils.pos2mask[p]
                    batch['posmask'] = posmask

            rl_src_mask = (src_pos != self.data_utils.pad).unsqueeze(1)

            if self.args.beam_size > 1:
                seq_gen = SequenceGenerator(self.transformer, self.data_utils, beam_size=self.args.beam_size, no_repeat_ngram_size=self.args.block_ngram)
                out = seq_gen._generate(batch, src_pos, rl_src_mask, pos_masking=self.args.pos_masking, bos_token=self.data_utils.bos)
            else:
                if self.args.pos_masking:
                    out = self.transformer.greedy_decode(src_pos, rl_src_mask, self.data_utils.max_len, self.data_utils.bos, batch['posmask'])
                else:
                    out = self.transformer.greedy_decode(src_pos, rl_src_mask, self.data_utils.max_len, self.data_utils.bos)

            # out = self.transformer.greedy_decode(batch['src'].long(), batch['src_mask'], max_len, self.data_utils.bos)
            for i, l in enumerate(out):
                # print(l)
                if self.args.beam_size > 1:
                    sentence = self.data_utils.id2sent(l[0]['tokens'][:-1], True)
                else:
                    sentence = self.data_utils.id2sent(l[1:], True)
                # pos = pos_list[i]
                #print(l[1:])
                f.write(sentence)
                # f.write('\t')
                # pos_str = ""
                # for p in pos:
                #     pos_str += p 
                #     pos_str += " "
                # f.write(pos_str.strip())
                f.write("\n")
            src_pos = src.clone()
            pos_list = []
            # out, _ = self.transformer.forward(src_pos, tgt, 
            #                         batch['src_mask'], batch['tgt_mask'])
            # loss = self.transformer.loss_compute(out, batch['y'].long(), self.data_utils.pad)
            # total_loss.append(loss.item())
        if not self.args.pred_pos:
            print('sampler accs %f'%(sum(sampler_accs)/ len(sampler_accs)))
        # print('total_loss: %f'%(sum(total_loss)/len(total_loss)))

    def id2sent(self, data_utils, indices, test=False):
        #print(indices)
        sent = []
        word_dict={}
        for index in indices:
            if test and (index == data_utils.word2id['</s>'] or index in word_dict):
                continue
            sent.append(data_utils.index2word[index])
            word_dict[index] = 1

        return ' '.join(sent)
