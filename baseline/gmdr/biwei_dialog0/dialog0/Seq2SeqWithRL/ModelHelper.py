from dialog0.modules.RNNEncoder import RNNEncoder
from dialog0.modules.TagSampler import TagSampler
from dialog0.Seq2SeqWithRL.Model import Seq2SeqWithRL
from dialog0.modules.TagEmbeddings import TagEmbeddings
import dialog0.Seq2SeqWithRL.IO as IO
def create_seq2seq_rl_model(config, fields, tag_sampler, seq2seq_model):
    seq2seq_with_rl = Seq2SeqWithRL(seq2seq_model, tag_sampler)
    print(seq2seq_with_rl)
    return seq2seq_with_rl