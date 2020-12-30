from dialog0.TagSampler.modules.RNNEncoder import RNNEncoder
from dialog0.TagSampler.Model import TagSampler
from dialog0.modules.Embeddings import SharedEmbeddings
import dialog0.TagSampler.IO as IO
def create_tag_sampler(config, fields):
    shared_embeddings = SharedEmbeddings(len(fields['tag'].vocab),
                                   config['TagSampler']['tag_embedding_size'],
                                   fields['src'].vocab.stoi[IO.PAD_WORD])
    rnn_encoder = RNNEncoder(config['TagSampler']['rnn_type'], config['Seq2Seq']['shared_embedding_size'],
                         config['TagSampler']['enc_hidden_size'],
                         config['TagSampler']['num_layers'],
                         config['TagSampler']['dropout'],
                         config['TagSampler']['bidirectional'])
    tag_sampler = TagSampler(shared_embeddings,rnn_encoder,len(fields['tag'].vocab))
    return tag_sampler
