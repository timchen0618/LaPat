import torch
import pdb
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import dialog0.Seq2SeqWithRL.IO as IO
from sklearn import feature_extraction    
from sklearn.feature_extraction.text import TfidfTransformer    
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.cluster import KMeans 
class Seq2SeqWithRL(nn.Module):
    def __init__(self, seq2seq, tag_sampler):
        super(Seq2SeqWithRL, self).__init__()
        self.seq2seq = seq2seq
        self.tag_sampler = tag_sampler
    def forward(self, input, fields):
        src_inputs = input[0]
        src_lengths = input[1]
        tgt_inputs = input[2]
        selected_tag_input = input[3]
        
        # tag_inputs are repeated
        # src_input = src_inputs[:,0].unsqueeze(-1)
        # src_length = [src_lengths[0]]
        # tag_input = tag_inputs[:,0]
        encoder_outputs, encoder_hidden = self.encode(src_inputs, src_lengths, None)
        # sampler_output = self.sample_tag(src_input,src_length)

        # select top 3
        
        # tag_log_probs = torch.gather(sampler_output,-1,tag_input)
        # tag_probs = tag_log_probs.exp()
        # np_tag_probs = tag_probs.data.cpu().numpy()
        # np_tag_probs /= np_tag_probs.sum()
        # selected_tag_index = np.random.choice(a=tag_input.size(0),size=1,p=np_tag_probs)[0]
        # selected_tag = tag_input[selected_tag_index].unsqueeze(-1)

        # selected_tag_logprob = tag_log_probs[selected_tag_index]
        selected_tag_input = selected_tag_input.expand(1,tgt_inputs.size(1))

        tag_hidden = self.tag_encode(selected_tag_input)
        decoder_init_hidden = self.seq2seq.init_decoder_state((encoder_hidden,tag_hidden))

        decoder_outputs , decoder_hiddens, attn_scores \
                = self.decode(
                tgt_inputs, tag_hidden, encoder_outputs, decoder_init_hidden
            )
        # there is a bug here
        decoded_indices,decoded_words=self.greddy_decoding(fields,
                                                    decoder_init_hidden[:,0].unsqueeze(1),
                                                    tag_hidden[:,0].unsqueeze(1),
                                                    encoder_outputs[:,0].unsqueeze(1),
                                                    20)   
        return decoder_outputs, decoded_indices, decoded_words

    def select_tag(self, src_inputs):
        return src_inputs

    def tag_encode(self, input):
        tag_embeddings = self.seq2seq.shared_embeddings(input)
        tag_hidden = self.seq2seq.tag_encoder(tag_embeddings) #bow
        return tag_hidden

    def sample_tag(self, src_inputs, src_lengths):
        # pdb.set_trace()
        outputs = self.tag_sampler(src_inputs, src_lengths)
        return outputs

    def sample_tag_with_kmeans(self, input, sampler_output):
        src_inputs = input[0]
        src_lengths = input[1]
        tgt_inputs = input[2]
        tag_inputs = input[3]
        # tag_inputs are repeated
        src_input = src_inputs[:,0].unsqueeze(-1)
        src_length = [src_lengths[0]]
        tag_input = tag_inputs[:,0]

        
        sampler_output = self.sample_tag(src_input,src_length)
        tag_log_probs = torch.gather(sampler_output,-1,tag_input)
        np_tag_input = tag_input.data.tolist()
        np_tag_log_probs = tag_log_probs.data.tolist()

        tag_hidden = self.tag_encode(tag_input.unsqueeze(0)).squeeze(0)
        clf = KMeans(n_clusters=3,max_iter=300)
        clf.fit(tag_hidden.data.cpu().numpy())
        clusters = [[] for _ in range(3)]
        selected_tag = []
        selected_tag_logprob = []
        for i,(data,log_prob) in enumerate(zip(np_tag_input, np_tag_log_probs)):
            clusters[clf.labels_[i]].append((data,log_prob,i))
        if len(clusters[0]) > 0:
            clusters[0] = sorted(clusters[0],key=lambda x:-(x[1]))
            selected_tag.append(clusters[0][0][0])
            selected_tag_logprob.append(tag_log_probs[clusters[0][0][2]])
        if len(clusters[1]) > 0:
            clusters[1] = sorted(clusters[1],key=lambda x:-(x[1]))
            selected_tag.append(clusters[1][0][0])
            selected_tag_logprob.append(tag_log_probs[clusters[1][0][2]])

        if len(clusters[2]) > 0:
            clusters[2] = sorted(clusters[2],key=lambda x:-(x[1]))
            selected_tag.append(clusters[2][0][0])        
            selected_tag_logprob.append(tag_log_probs[clusters[2][0][2]])
        selected_tag_logprob = torch.stack(selected_tag_logprob)
        return selected_tag, selected_tag_logprob
        # select top 3
        # tag_probs = tag_log_probs.exp()
        # np_tag_probs = tag_probs.data.cpu().numpy()
        # np_tag_probs /= np_tag_probs.sum()
        # selected_tag_index = np.random.choice(a=tag_input.size(0),size=1,p=np_tag_probs)[0]
        # selected_tag = tag_input[selected_tag_index].unsqueeze(-1)

        # selected_tag_logprob = tag_log_probs[selected_tag_index]



    def encode(self, input, lengths=None, hidden=None):
        encoder_input = input
        encoder_outputs, encoder_hidden = self.seq2seq.encode(encoder_input, lengths, None)

        return encoder_outputs, encoder_hidden



    def decode(self, input, tag_hidden, context, state):

        decoder_input = input
        decoder_outputs , decoder_hiddens, attn_scores \
                = self.seq2seq.decode(
                decoder_input, tag_hidden, context, state
            )
        return decoder_outputs,decoder_hiddens,attn_scores

    def greedy_infer(self):
        pass

    def drop_checkpoint(self, epoch, fname):
        torch.save({'seq2seq_dict': self.seq2seq.state_dict(),
                    'tag_sapmler_dict': self.tag_sampler.state_dict(),
                    'epoch': epoch,
                    },
                   fname)


    def load_checkpoint(self, fname):
        cpnt = torch.load(fname, map_location=lambda storage, loc: storage)
        self.seq2seq.load_state_dict(cpnt['seq2seq_dict'])
        self.tag_sampler.load_state_dict(cpnt['tag_sapmler_dict'])
        epoch = cpnt['epoch']
        return epoch

    def greddy_decoding(self, fields, decoder_init_hidden, tag_hidden, encoder_outputs, max_length):
         # Store output words and attention states
        decoded_indices = []        
        decoded_words = []    
        # Run through decoder
        decoder_hidden = decoder_init_hidden
        decoder_input = Variable(torch.LongTensor([fields['tgt'].vocab.stoi[IO.BOS_WORD]])).cuda()
        self.seq2seq.eval()
        for di in range(max_length):

            decoder_input = decoder_input.unsqueeze(0)
            # pdb.set_trace()
            decoder_output, decoder_hidden, attn_scores = self.seq2seq.decode(
                decoder_input, tag_hidden, encoder_outputs, decoder_hidden
            )

            # Choose top word from output
            word_prob = self.seq2seq.generator(decoder_output)

            topv, topi = word_prob.data.topk(1)


            if  (topi == fields['tgt'].vocab.stoi[IO.EOS_WORD]).all():
                decoded_indices.append(topi)
                break

                
            # Next input is chosen word
            # pdb.set_trace()
            ni = topi.tolist()[0][0][0]

            decoder_input = Variable(topi)
            

            decoder_input = decoder_input.squeeze()
            # 下面這一行我加的
            decoder_input = decoder_input.unsqueeze(0)
            
            decoded_indices.append(topi)
            decoded_words.append(fields['tgt'].vocab.itos[ni])
        decoded_indices = torch.stack(decoded_indices).squeeze()
        self.seq2seq.train()
        return decoded_indices, decoded_words


