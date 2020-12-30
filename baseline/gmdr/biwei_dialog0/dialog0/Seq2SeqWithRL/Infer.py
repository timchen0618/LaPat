import torch
from torch.autograd import Variable
from dialog0.modules.Beam import Beam
import dialog0.Seq2SeqWithRL.IO as IO
from sklearn.feature_extraction.text import TfidfTransformer    
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.cluster import KMeans 
class Infer(object):

    def __init__(self, model, fields,
                 beam_size, n_best=1,
                 max_length=100,
                 global_scorer=None, cuda=False,
                 beam_trace=False, min_length=0):
        self.model = model.eval()
        self.fields = fields
        self.n_best = n_best
        self.max_length = max_length
        self.global_scorer = global_scorer
        self.beam_size = beam_size
        self.cuda = cuda
        self.min_length = min_length

        # for debugging
        self.beam_accum = None
        if beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": []}

    def inference_batch(self, batch, tag_input, tag_score):
        src_inputs = batch.src[0]
        src_lengths = batch.src[1].tolist()
        # tag_inputs = batch.tag.squeeze()
        
        # (0) Prep each of the components of the search.
        # And helper method for reducing verbosity.
        beam_size = self.beam_size
        batch_size = len(src_lengths)
        vocab = self.fields["tgt"].vocab
        beam = [Beam(beam_size, n_best=self.n_best,
                                    cuda=self.cuda,
                                    global_scorer=self.global_scorer,
                                    pad=vocab.stoi[IO.PAD_WORD],
                                    eos=vocab.stoi[IO.EOS_WORD],
                                    bos=vocab.stoi[IO.BOS_WORD],
                                    min_length=self.min_length)
                for __ in range(batch_size)]

        # Help functions for working with beams and batches
        def var(a): return Variable(a, volatile=True)

        def rvar(a): return var(a.repeat(1, beam_size, 1))

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)

        # (1) Run the encoder on the src.
        context, enc_states = self.model.encode(src_inputs, src_lengths)

        
        selected_tag = Variable(torch.LongTensor([tag_input])).unsqueeze(-1).cuda()
        selected_tag_idx = tag_input
        selected_tag_score = tag_score
        tag_hidden = self.model.tag_encode(selected_tag)
        dec_states = self.model.seq2seq.init_decoder_state((enc_states,tag_hidden))

        # (2) Repeat src objects `beam_size` times.

        context = rvar(context.data)
        src_lengths = Variable(torch.LongTensor(src_lengths)).cuda()
        if not isinstance(dec_states, tuple): # GRU
            dec_states = Variable(dec_states.data.repeat(1, beam_size, 1))
        else: # LSTM
            dec_states = (
                Variable(dec_states[0].data.repeat(1, beam_size, 1)),
                Variable(dec_states[1].data.repeat(1, beam_size, 1)),
                )

        # (3) run the decoder to generate sentences, using beam search.
        for i in range(self.max_length):
            if all((b.done() for b in beam)):
                break

            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            inp = var(torch.stack([b.get_current_state() for b in beam])
                      .t().contiguous().view(1, -1))


            # Temporary kludge solution to handle changed dim expectation
            # in the decoder
            # inp = inp.unsqueeze(2)

            # Run one step.
            dec_out, dec_states, attn_scores = self.model.decode(inp, tag_hidden, context, dec_states)
            if not isinstance(dec_states, tuple): # GRU
                dec_states = [ 
                    dec_states
                ]   
            else:
                dec_states = [ 
                    dec_states[0],
                    dec_states[1]
                ]

            dec_out = dec_out.squeeze(0)
            # dec_out: beam x rnn_size
            
            # (b) Compute a vector of batch*beam word scores.
            out = self.model.seq2seq.generator(dec_out).data
            out = unbottle(out)
            # beam x tgt_vocab
            beam_attns = {}
            beam_attns['tag'] = unbottle(attn_scores['tag'])
            beam_attns['ctx'] = unbottle(attn_scores['ctx'])
            tag_c = attn_scores['tag_c'][:, 0].squeeze()
            # (c) Advance each beam.
            attn_out = {}
            for j, b in enumerate(beam):
                attn_out['ctx'] = beam_attns['ctx'][:, j]
                attn_out['tag'] = beam_attns['tag'][:, j]
                b.advance(
                    out[:, j], attn_out)

                dec_states = self.beam_update(j, b.get_current_origin(), beam_size, dec_states)

            if len(dec_states) < 2: # GRU
                dec_states = dec_states[-1]
            else:
                dec_states = ( 
                    dec_states[0],
                    dec_states[1]
                ) 
        # (4) Extract sentences from beam.
        ret = self._from_beam(beam)
        selected_tag = vocab.itos[selected_tag_idx]
        ret['tag'] = selected_tag
        ret['tag_c'] = tag_c
        ret['tag_score'] = selected_tag_score
        return ret

    def beam_update(self, idx, positions, beam_size,states):
        out = []
        for e in states:
            a, br, d = e.size()
            sent_states = e.view(a, beam_size, br // beam_size, d)[:, :, idx]
            sent_states.data.copy_(
                sent_states.data.index_select(1, positions))
            out.append(sent_states)
        return tuple(out)

    def sample_tag_with_kmeans(self, batch, topk_tag=1000, n_cluster=3):
        src_inputs = batch.src[0]
        src_lengths = batch.src[1].tolist()
        context, enc_states = self.model.encode(src_inputs, src_lengths)
        sampler_output = self.model.sample_tag(src_inputs,src_lengths)
        sampler_output.data[0] = -1e20
        tag_log_probs = sampler_output

        selected_tag_score, selected_tag_pos = tag_log_probs.data.topk(topk_tag,dim=-1)

        # selected_tag = tag_inputs[selected_tag_pos[-1]].unsqueeze(-1)
        tag_hidden = self.model.tag_encode(Variable(selected_tag_pos).unsqueeze(0)).squeeze(0)
        tag_c = self.model.seq2seq.decoder.cal_tag_atten(tag_hidden, context)
        tag_c = tag_c.data.cpu().numpy()
        clf = KMeans(n_clusters=n_cluster,init='k-means++', max_iter=300)
        clf.fit(tag_c)
        distance = clf.transform(tag_c)
        np_topk_tag_idx = selected_tag_pos.tolist()
        np_topk_tag_score = selected_tag_score.tolist()
        clusters = [[] for _ in range(n_cluster)]
        for i,(data,log_prob,c) in enumerate(zip(np_topk_tag_idx, np_topk_tag_score,tag_c)):
            
            clusters[clf.labels_[i]].append((data,log_prob,i,distance[i][clf.labels_[i]]))

        for idx,cluster in enumerate(clusters):
            clusters[idx] = sorted(cluster,key=lambda x:-(x[3]))
        return clusters
    def _from_beam(self, beam):
        ret = {"predictions": [],
               "scores": [],
               "tag_attns": [],
               "ctx_attns": []}
        for b in beam:
            if self.beam_accum:
                self.beam_accum['predicted_ids'].append(torch.stack(b.next_ys[1:]).tolist())
                self.beam_accum['beam_parent_ids'].append(torch.stack(b.prev_ks).tolist())
                self.beam_accum['scores'].append(torch.stack(b.all_scores).tolist())

            n_best = self.n_best
            scores, ks = b.sort_finished(minimum=n_best)
            hyps, ctx_attns, tag_attns = [], [], []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, ctx_attn, tag_attn = b.get_hyp(times, k)
                hyps.append(hyp)
                ctx_attns.append(ctx_attn)
                tag_attns.append(tag_attn)
            ret["predictions"].append(hyps)
            ret["scores"].append(scores)
            ret["ctx_attns"].append(ctx_attns)
            ret["tag_attns"].append(tag_attns)

        return ret