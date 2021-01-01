from dialog0.Sampler.modules.SharedEmbeddings import Embeddings
from dialog0.Sampler.modules.Model_architecture import Model
from dialog0.Sampler.Model import Sampler
from dialog0.Sampler import config


def create_sampler_model(vocab, index_2_latent_sentence, device=0):
    shared_embeddings = Embeddings(input_size=vocab.n_words, embedding_dim=config.emb_dim)

    model = Model(model_file_path=None, is_eval=False, hidden_dim=config.hidden_dim, output_size=len(index_2_latent_sentence))

    sampler = Sampler(shared_embeddings=shared_embeddings,
                      model=model,
                      vocab=vocab,
                      device=device)

    return sampler
