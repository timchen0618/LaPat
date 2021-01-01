import csv

# filepath
sampler_output_filepath = "../report/sampler_pretrain/test_sampler_report.tsv"
testing_filepath = "../../../../alexchao2007/code/weibo_data_final/10000/new_data_test_10000_meteor.tsv"

testing_cc_filepath = "../data/2stage_data/2stage_testing_data.tsv"


def read_testing_corpus(filepath):
    corpus = []
    with open(filepath, 'r') as tsv_in:
        tsv_reader = csv.reader(tsv_in, delimiter='\t')

        for line in tsv_reader:
            input_post = line[0]
            target_response = line[1]
            complete_postags = line[2]
            processed_postags = line[3]
            pos_id = line[4]

            corpus.append([input_post, target_response])

    return corpus


def read_sampler_corpus(corpus, filepath):
    with open(filepath, 'r') as tsv_in:
        tsv_reader = csv.reader(tsv_in, delimiter='\t')

        for id, line in enumerate(tsv_reader):
            post = line[0]
            latent_sentence = line[1]

            corpus[id].append(latent_sentence)

    return corpus


def write_file(corpus, filepath):
    with open(filepath, 'w') as tsv_out:
        tsv_writer = csv.writer(tsv_out, delimiter='\t')

        for line in corpus:
            tsv_writer.writerow(line)



if __name__ == '__main__':
    testing_corpus = read_testing_corpus(testing_filepath)
    sampler_corpus = read_sampler_corpus(testing_corpus, sampler_output_filepath)
    write_file(sampler_corpus, testing_cc_filepath)
