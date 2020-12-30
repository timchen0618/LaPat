import torch.nn as nn
from dialog0.modules.Embeddings import SharedEmbeddings
from dialog0.modules.TagEncoder import TagEncoder
from dialog0.modules.RNNEncoder import RNNEncoder
from dialog0.modules.TagSampler import TagSampler
from dialog0.Seq2Seq.Model import Seq2SeqWithTag
from dialog0.Seq2Seq.modules.JointAttention import JointAttention
from dialog0.Seq2Seq.modules.JointAttnDecoder import JointAttnDecoder
import dialog0.Seq2Seq.IO as IO
def create_seq2seq_tag_model(config,fields,use_tag=True):
    shared_embeddings = SharedEmbeddings(len(fields['src'].vocab), 
                                         config['Seq2Seq']['shared_embedding_size'],
                                         fields['src'].vocab.stoi[IO.PAD_WORD])

    tag_encoder = TagEncoder(shared_embeddings.embedding_size,
                             config['Seq2Seq']['tag_hidden_size'],
                             fields['src'].vocab.stoi[IO.PAD_WORD])

    encoder = RNNEncoder(config['Seq2Seq']['rnn_type'], shared_embeddings.embedding_size,
                         config['Seq2Seq']['enc_hidden_size'],
                         config['Seq2Seq']['num_layers'],
                         config['Seq2Seq']['dropout'],
                         config['Seq2Seq']['bidirectional'])
    attention = JointAttention(config['Seq2Seq']['enc_hidden_size'],
                               config['Seq2Seq']['dec_hidden_size'],
                               config['Seq2Seq']['tag_hidden_size'])

    decoder = JointAttnDecoder(config['Seq2Seq']['rnn_type'],
                         attention,
                         shared_embeddings.embedding_size,
                         config['Seq2Seq']['dec_hidden_size'],
                         config['Seq2Seq']['num_layers'],
                         config['Seq2Seq']['dropout']
                         )
    generator = nn.Sequential(nn.Linear(decoder.hidden_size,len(fields['tgt'].vocab)),
                              nn.LogSoftmax(dim=-1))

    if config['Seq2Seq']['tie_weights']:
        generator[0].weight = shared_embeddings.embedding.weight

    seq2seq_with_tag = Seq2SeqWithTag(shared_embeddings,
                                   tag_encoder,
                                   encoder,decoder,
                                   generator,
                                   config['Seq2Seq']['feat_merge'],
                                   use_tag)

    return seq2seq_with_tag