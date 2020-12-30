import torch
import argparse
import codecs
import dialog0.Seq2SeqWithRL.IO as IO
import json
from torch import cuda
from dialog0.Seq2Seq.ModelHelper import create_seq2seq_tag_model
from dialog0.TagSampler.ModelHelper import create_tag_sampler
from dialog0.Seq2SeqWithRL.ModelHelper import create_seq2seq_rl_model
import dialog0.Utils as utils

import torch
from torch.autograd import Variable
from dialog0.modules.Beam import Beam
import dialog0.Seq2SeqWithRL.IO as IO

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


    def inference_batch(self, batch, topk_tag):
        src_inputs = batch.src[0]
        src_lengths = batch.src[1].tolist()
        tag_inputs = batch.tag.squeeze()
        
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
        sampler_output = self.model.sample_tag(src_inputs,src_lengths)
        tag_log_probs = sampler_output
        # tag_log_probs = torch.gather(sampler_output,-1,tag_inputs)
        # if tag_inputs.size(0) < topk_tag:
        #     topk_tag = 1
        selected_tag_score, selected_tag_pos = tag_log_probs.data.topk(topk_tag,dim=-1)
        # selected_tag = tag_inputs[selected_tag_pos[-1]].unsqueeze(-1)
        selected_tag = Variable(torch.LongTensor([selected_tag_pos[-1]])).unsqueeze(-1).cuda()
        selected_tag_idx = selected_tag_pos[-1]
        selected_tag_score = selected_tag_score[-1]
        tag_hidden = self.model.tag_encode(selected_tag)

        selected_tag = vocab.itos[selected_tag_idx]
        ret['tag'] = selected_tag
        ret['tag_score'] = selected_tag_score
        return ret


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-vocab", type=str)
    parser.add_argument("-model", type=str)
    parser.add_argument("-config", type=str)

    args = parser.parse_args()
    config = utils.load_config(args.config)

    use_cuda = False
    if args.gpuid:
        cuda.set_device(args.gpuid[0])
        use_cuda = True
    fields = IO.load_fields(
                torch.load(args.vocab))


    # Build model.

    seq2seq_model = create_seq2seq_tag_model(config, fields)
    sampler_model = create_tag_sampler(config, fields)
    model = create_seq2seq_rl_model(config, fields, sampler_model, seq2seq_model)


    print('Loading parameters ...')
    if args.model:
        model.load_checkpoint(args.model)
    if use_cuda:
        model = model.cuda()
    

if __name__ == '__main__':
    main()

