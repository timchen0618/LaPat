import torch.nn as nn
from dialog0.Seq2Seq import config
from dialog0.Seq2Seq.modules.SharedEmbeddings import Embeddings
from dialog0.Seq2Seq.modules.Model_architecture import Model
from dialog0.Seq2Seq.Model import Seq2SeqWithTag

def create_seq2seq_model(vocab, device=0):
    shared_embeddings = Embeddings(input_size=vocab.n_words, embedding_dim=config.emb_dim)

    model = Model(model_file_path=None, is_eval=False, vocab_size=vocab.n_words)

    seq2seq_with_tag = Seq2SeqWithTag(shared_embeddings=shared_embeddings,
                                      model=model,
                                      vocab=vocab,
                                      vocab_size=vocab.n_words,
                                      device=device)

    return seq2seq_with_tag
