import numpy as np
import pdb
import pickle
def main():
    # input_file = '../seq2seq_bm25_sentence_pos_processed_train_corpus.tsv'
    input_file_path = 'unprocessed_files/unprocessed_file_'
    output_file = 'pos_unprocessed_1000_latent_align_train.tsv'
    with open(output_file, 'w') as w_file:
        for i in range(142):
            input_file = input_file_path + str(i) + '.tsv'
            with open(input_file, 'r') as readlines:
                for idx, lines in enumerate(readlines):
                    print(i, ', ', idx)
                    w_file.write(lines)
if  __name__ == '__main__':
    main()
