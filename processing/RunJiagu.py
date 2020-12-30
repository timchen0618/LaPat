import csv
import jiagu


# corpus filepath
seq2seq_train_filepath = '../multi_response/data/weibo/weibo_official/weibo_src_tgt_utf8.train'
seq2seq_pos_train_filepath = '../multi_response/data/weibo/weibo_official/weibo_src_tgt_pos_utf8.train'

seq2seq_dev_filepath = '../multi_response/data/weibo/weibo_official/weibo_src_tgt_utf8.dev'
seq2seq_pos_dev_filepath = '../multi_response/data/weibo/weibo_official/weibo_src_tgt_pos_utf8.dev'

seq2seq_test_filepath = '../multi_response/data/weibo/weibo_official/weibo_src_tgt.test'
seq2seq_pos_test_filepath = '../multi_response/data/weibo/weibo_official/weibo_src_tgt_pos.test'


def read_corpus(filepath):
    print('Reading weibo corpus...')
    corpus = []
    with open(filepath, 'r') as tsv_in:
        tsv_reader = csv.reader(tsv_in, delimiter='\t')

        for line in tsv_reader:
            post = line[0]
            response = line[1]

            corpus.append([post, response])

    return corpus


def execute(corpus):
    print('Start Pos processing...')
    for id in range(len(corpus)):
        post = corpus[id][0].strip()
        response = corpus[id][1].strip()

        response_lst = response.split(' ')
        while '' in response_lst:
            response_lst.remove('')

        response_pos = jiagu.pos(response_lst)
        response_pos = ' '.join(response_pos)

        corpus[id].append(response_pos)

        if ((id+1) % 50000) == 0:
            print("already process {} instances".format(id+1))

    return corpus


def write_file(filepath, corpus):
    print('Writing data into file...\n')
    with open(filepath, 'w') as tsv_out:
        tsv_writer = csv.writer(tsv_out, delimiter='\t')

        for line in corpus:
            tsv_writer.writerow(line)



if __name__ == '__main__':
    # train
    corpus_train = read_corpus(seq2seq_train_filepath)
    corpus_pos_train = execute(corpus_train)
    write_file(seq2seq_pos_train_filepath, corpus_pos_train)

    # development
    corpus_dev = read_corpus(seq2seq_dev_filepath)
    corpus_pos_dev = execute(corpus_dev)
    write_file(seq2seq_pos_dev_filepath, corpus_pos_dev)

    # test
    corpus_test = read_corpus(seq2seq_test_filepath)
    corpus_pos_test = execute(corpus_test)
    write_file(seq2seq_pos_test_filepath, corpus_pos_test)
