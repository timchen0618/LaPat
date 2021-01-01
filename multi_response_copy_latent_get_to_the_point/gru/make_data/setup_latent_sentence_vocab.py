import csv
import pickle

#filepath
latent_sentence_text_filepath = "../../../alexchao2007/code/multi-response/data/5w_latent_sentences_seg.txt"
latent_sentence_text_out_filepath = "./data/sampler_data/5w_latent.pkl"


def read_corpus(filepath):
    corpus = []
    with open(filepath, 'r') as tsv_in:
        tsv_reader = csv.reader(tsv_in, delimiter='\t')

        for line in tsv_reader:
            corpus.append(line[0])

    return corpus


def make_vocab(corpus):
    vocab = {'latent_sentence_2_index':None,
             'index_2_latent_sentence': None}

    latent_sentence_2_index = {}
    index_2_latent_sentence = {}
    for pointer, sentence in enumerate(corpus):
        latent_sentence_2_index[sentence] = pointer
        index_2_latent_sentence[pointer] = sentence

    vocab['latent_sentence_2_index'] = latent_sentence_2_index
    vocab['index_2_latent_sentence'] = index_2_latent_sentence

    return vocab


def write_vocab(vocab, filepath):
    with open(filepath, 'wb') as pkl_out:
        pickle.dump(vocab, pkl_out)



if __name__ == '__main__':
    corpus = read_corpus(latent_sentence_text_filepath)
    vocab = make_vocab(corpus)
    write_vocab(vocab, latent_sentence_text_out_filepath)
