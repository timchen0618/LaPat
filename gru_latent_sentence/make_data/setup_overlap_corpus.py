import csv

# filepath
corpus_2stage_filepath = "../data/2stage_data/2stage_testing_data.tsv"
testing_report_filepath = "../report/test_seq2seq_report_epoch9.tsv"


def read_2stage_corpus(filepath):
    corpus = []
    with open(filepath, 'r') as tsv_in:
        tsv_reader = csv.reader(tsv_in, delimiter='\t')

        for line in tsv_reader:
            corpus.append(line)

    return corpus


def read_testing_corpus(filepath):
    corpus = []
    with open(filepath, 'r') as tsv_in:
        tsv_reader = csv.reader(tsv_in, delimiter='\t')

        for line in tsv_reader:
            corpus.append(line)

    return corpus


def combine(corpus_2stage, corpus_testing):
    corpus = []
    for line_1, line_2 in zip(corpus_2stage, corpus_testing):
        corpus.append(line_1+line_2)

    return corpus


def calculate_overlap(corpus):
    f1_score_lst = []
    for line in corpus:
        post = line[0]
        response = line[1]
        latent_sentence = line[2]
        prediction = line[3]

        f1_score_lst.append(get_f1_score(latent_sentence.split(), prediction.split()))

    return sum(f1_score_lst) / len(f1_score_lst)


def get_f1_score(infer_words, ground_words):
    infer_set = set(infer_words)
    ground_set = set(ground_words)
    intersect = ground_set.intersection(infer_set)
    precision = len(intersect)/len(infer_set)
    recall = len(intersect)/len(ground_set)
    if precision == 0.0 or recall == 0.0:
        f1_score = 0.0
    else:
        f1_score = 2.0*(precision*recall) / (precision+recall)
    return f1_score





if __name__ == '__main__':
    corpus_2stage = read_2stage_corpus(corpus_2stage_filepath)
    corpus_testing = read_testing_corpus(testing_report_filepath)
    corpus_combine = combine(corpus_2stage, corpus_testing)
    overlap = calculate_overlap(corpus_combine)
    print(overlap)
