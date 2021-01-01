import csv

# filepath
corpus_2stage_filepath = "../data/2stage_data/2stage_testing_data.tsv"
corpus_predict_filepath = "../report/test_seq2seq_report_epoch0.tsv"

corpus_combine_filepath = "../data/2stage_data/latent_pred.tsv"


def read_2stage_corpus(filepath):
    corpus = []
    with open(filepath, 'r') as tsv_in:
        tsv_reader = csv.reader(tsv_in, delimiter='\t')

        for line in tsv_reader:
            post = line[0]
            response = line[1]
            latent_sentence = line[2]
            corpus.append([latent_sentence])

    return corpus


def read_testing_corpus(filepath):
    corpus = []
    with open(filepath, 'r') as tsv_in:
        tsv_reader = csv.reader(tsv_in, delimiter='\t')

        for line in tsv_reader:
            prediction = line[0]
            corpus.append([prediction])

    return corpus


def combine(corpus_2stage, corpus_testing):
    corpus = []
    for line1, line2 in zip(corpus_2stage, corpus_testing):
        corpus.append(line1+line2)

    return corpus


def write_file(corpus, filepath):
    with open(filepath, 'w') as tsv_out:
        tsv_writer = csv.writer(tsv_out, delimiter='\t')

        for line in corpus:
            tsv_writer.writerow(line)



if __name__ == '__main__':
    corpus_2stage = read_2stage_corpus(corpus_2stage_filepath)
    corpus_testing = read_testing_corpus(corpus_predict_filepath)
    corpus_combine = combine(corpus_2stage, corpus_testing)
    write_file(corpus_combine, corpus_combine_filepath)
