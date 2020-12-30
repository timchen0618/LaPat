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
import colorama
from colorama import Fore, Style, Back

# from torch.utils.tensorboard import SummaryWriter


class Solver():
    def __init__(self, args):
        self.args = args
        self.data_utils = DataUtils(args)
        if args.save_checkpoints:
            self.model_dir = make_save_dir(os.path.join(args.model_dir, args.sampler_label, args.exp_name))
        self.disable_comet = args.disable_comet
        colorama.init(autoreset=True) 

    def make_model(self, src_vocab, tgt_vocab, N=6, 
        d_model=512, d_ff=2048, h=8, dropout=0.1, g_drop=0.1):
        
        "Helper: Construct a model from hyperparameters."
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        word_embed = nn.Sequential(Embeddings(d_model, src_vocab), c(position))

        model = Sampler(Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N), word_embed, Generator(d_model, self.args.num_classes, g_drop))
        print(model)
        # This was important from their code. 
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model.cuda()

    def train(self):
        # logging
        if not self.disable_comet:
            COMET_PROJECT_NAME = 'weibo-stc'
            COMET_WORKSPACE = 'timchen0618'

            self.exp = Experiment(project_name=COMET_PROJECT_NAME,
                                  workspace=COMET_WORKSPACE,
                                  auto_output_logging='simple',
                                  auto_metric_logging=None,
                                  display_summary=False,
                                  )

            self.exp.add_tag(self.args.sampler_label)
            self.exp.add_tag('Sampler')
            if self.args.processed:
                self.exp.add_tag('processed')
            else:
                self.exp.add_tag('unprocessed')
            self.exp.set_name(self.args.exp_name)

        ###### loading .... ######
        vocab_size = self.data_utils.vocab_size
        print("============================")  
        print("=======start to build=======")
        print("============================") 
        print("Vocab Size: %d"%(vocab_size))

        #make model
        self.model = self.make_model(src_vocab=vocab_size, 
                                     tgt_vocab=vocab_size, 
                                     N=self.args.num_layer, 
                                     dropout=self.args.dropout,
                                     g_drop=self.args.generator_drop
                                     )
        self.model.load_embedding(self.args.pretrain_model)
        # sys.exit(0)

        lr = 1e-7
        generator_lr = 1e-4
        d_model = 512
        warmup_steps = self.args.warmup_steps
        # optim = torch.optim.Adam([
        #                           {'params':list(self.model.encoder.parameters())+list(self.model.src_embed.parameters())},
        #                           {'params':self.model.generator.parameters(), 'lr':generator_lr} 
        #                           ], lr=lr, betas=(0.9, 0.98), eps=1e-9)
        optim = torch.optim.AdamW([
                                  {'params':list(self.model.encoder.parameters())+list(self.model.src_embed.parameters())},
                                  {'params':self.model.generator.parameters(), 'lr':generator_lr} 
                                  ], lr=lr, betas=(0.9, 0.98), eps=1e-9)

        # optim = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
        total_loss = []
        train_accs = []
        start = time.time()

        step = 0
        for epoch in range(self.args.num_epoch):
            self.model.train()
            train_data = self.data_utils.data_yielder(epo=epoch)
            

            for batch in train_data:
                optim.zero_grad()
                if step % 20 == 1:
                    lr = self.args.lr * (1/(d_model**0.5))*min((1/(step)**0.5), step * (1/(warmup_steps**1.5)))
                    optim.param_groups[0]['lr'] = lr
                    lr2 = self.args.g_lr * (1/(d_model**0.5))*min((1/(step)**0.5), step * (1/(warmup_steps**1.5)))
                    optim.param_groups[1]['lr'] = lr2
                    # for param_group in optim.param_groups:
                    #     param_group['lr'] = lr

                src = batch['src'].long()
                src_mask = batch['src_mask'].long()
                y = batch['y']

                #forward model
                out = self.model.forward(src, src_mask)

                # print out
                pred = out.topk(5, dim=-1)[1].squeeze().detach().cpu().numpy()
                gg = batch['src'].long().detach().cpu().numpy()
                yy = batch['y']

                #compute loss
                
                loss = self.model.loss_compute(out, y, self.args.multi)
                # else:
                #     loss = self.model.loss_compute(out, y.long().unsqueeze(1), self.args.multi)
                #compute acc
                acc = self.model.compute_acc(out, batch['y'], self.args.multi)

                loss.backward()
                # print('emb_size', self.model.src_embed[0].lut.weight.size())
                # print('emb', self.model.src_embed[0].lut.weight.grad.sum())
                # print('enc_out', self.model.encoder.layers[0].feed_forward.w_1.weight.grad)
                optim.step()
                total_loss.append(loss.detach().cpu().numpy())
                train_accs.append(acc)
                
                if step % self.args.print_every_step == 1:
                    elapsed = time.time() - start
                    print(Fore.GREEN + "[Step: %d]"%step + 
                          Fore.WHITE + " Loss: %f | Time: %f | Acc: %4.4f | Lr: %4.6f" %(np.mean(total_loss), elapsed, sum(train_accs)/len(train_accs), optim.param_groups[0]['lr']))
                    print(Fore.RED + 'src:', Style.RESET_ALL, self.id2sent(self.data_utils, gg[0][:150]))
                    print(Fore.RED + 'y:', Style.RESET_ALL, yy[0])
                    print(Fore.RED + 'pred:', Style.RESET_ALL, pred[0])
                    # print(train_accs)
                    # self.log.add_scalar('Loss/train', np.mean(total_loss), step)
                    if not self.disable_comet:
                        self.exp.log_metric('Train Loss', np.mean(total_loss), step=step)
                        self.exp.log_metric('Train Acc', sum(train_accs)/len(train_accs), step=step)
                        self.exp.log_metric('Learning Rate', lr, step=step)
                    # print('grad', self.model.src_embed.grad)
                    print(Style.RESET_ALL)
                    start = time.time()
                    total_loss = []
                    train_accs = []

                if step % self.args.valid_every_step == self.args.valid_every_step-1:
                    val_yielder = self.data_utils.data_yielder(epo=0, valid=True)

                    self.model.eval()
                    valid_losses = []
                    valid_accs = []
                    for batch in val_yielder:
                        with torch.no_grad():
                            out = self.model.forward(batch['src'].long(), batch['src_mask'].long())
                            loss = self.model.loss_compute(out, batch['y'], self.args.multi)
                                
                            acc = self.model.compute_acc(out, batch['y'], self.args.multi)
                            valid_accs.append(acc)
                            valid_losses.append(loss.item())

                    print('=============================================')
                    print('Validation Result -> Loss : %6.6f | Acc : %6.6f' %(sum(valid_losses)/len(valid_losses), sum(valid_accs)/ len(valid_accs)))
                    print('=============================================')
                    self.model.train()
                    # self.log.add_scalar('Loss/valid', sum(valid_losses)/len(valid_losses), step)
                    if not self.disable_comet:
                        self.exp.log_metric('Valid Loss', sum(valid_losses)/ len(valid_losses), step=step)
                        self.exp.log_metric('Valid Acc', sum(valid_accs)/ len(valid_accs), step=step)

                    if self.args.save_checkpoints:
                        print('saving!!!!')
                        model_name = str(int(step/1000)) + 'k_' + '%6.6f_%6.6f'%(sum(valid_losses)/len(valid_losses), sum(valid_accs)/ len(valid_accs)) + 'model.pth'
                        state = {'step': step, 'state_dict': self.model.state_dict()}
                        #state = {'step': step, 'state_dict': self.model.state_dict(),
                        #    'optimizer' : optim_topic_gen.state_dict()}
                        torch.save(state, os.path.join(self.model_dir, model_name))
                step += 1

    def test(self):
        #prepare model
        path = self.args.load_model
        max_len = self.args.max_len
        state_dict = torch.load(path)['state_dict']
        vocab_size = self.data_utils.vocab_size
        self.model = self.make_model(src_vocab = vocab_size, 
                                     tgt_vocab = vocab_size, 
                                     N = self.args.num_layer, 
                                     dropout = self.args.dropout,
                                     g_drop = self.args.generator_drop
                                     )
        model = self.model
        model.load_state_dict(state_dict)
        pred_dir = make_save_dir(self.args.pred_dir)
        filename = self.args.filename

        #start decoding
        data_yielder = self.data_utils.data_yielder()
        total_loss = []
        start = time.time()

        #file
        f = open(os.path.join(pred_dir, filename), 'w')

        self.model.eval()
        step = 0
        total_loss = []  
        corr, total = 0.0, 0.0
        for batch in data_yielder:
            step += 1
            if step % 10 == 1:
                print('Step ', step)
            # out = self.model.greedy_decode(batch['src'].long(), batch['src_mask'], max_len, self.data_utils.bos)
            with torch.no_grad():
                out = self.model.forward(batch['src'].long(), batch['src_mask'].long())
                if self.args.multi:
                # if True:
                    print('out', out.size())
                    print('y', batch['y'])
                    loss = self.model.loss_compute(out, batch['y'], self.args.multi)
                    c = self.model.compute_acc(out, batch['y'], self.args.multi)
                    corr += c*len(batch['y'])
                    total += len(batch['y'])

                    preds = out.argmax(dim=-1)
                    for x in preds:
                        f.write(str(x.item()))
                        f.write('\n')
                        # f.write(str(p.item()))
                        # f.write('\n')
                else:
                    loss = self.model.loss_compute(out, batch['y'], self.args.multi)
                    # preds = out.argmax(dim = -1)
                    _, preds = out.topk(dim = -1, k=10)
                    print(preds.size())

                    for p in preds:
                        for x in p:
                            f.write(str(x.item()))
                            f.write(' ')
                        f.write('\n')
                        # f.write(str(p.item()))
                        # f.write('\n')
                total_loss.append(loss.item())
        print(total_loss)
        print(sum(total_loss)/ len(total_loss))
        print('acc: ', corr/float(total))
        print(corr, ' ', total)

    def compute_acc(self, out, y):
        corr = 0.
        pred = out.argmax(dim = -1)
        for i, target in enumerate(y):
            # print(target)
            # print(pred[i])
            if pred[i] in target:
                corr += 1
        return corr, len(y)

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
