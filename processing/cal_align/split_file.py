import numpy as np
import pickle
import pdb

def main():
    input_file = '/share/home/alexchao2007/code/weibo_data_final/1000/pos_unprocessed_1000_train_meteor.tsv'
    cut_num = 30000
    tmp_lines = []

    total_size = 0
    with open(input_file, 'r') as readlines:
        for lines in enumerate(readlines):
            total_size = total_size + 1

    with open(input_file, 'r') as readlines:
        idx = 1
        file_idx = 0
        for lines in enumerate(readlines):
            tmp_lines.append(lines[1])
            if idx%cut_num == 0 or idx == total_size-1:
                print('file_idx: ', file_idx, ' idx: ', idx)
                with open('files/unprocessed_strct_' + str(file_idx) + '.tsv', 'w') as w_file:
                    for tmp_line in tmp_lines:
                        w_file.write(tmp_line)
                tmp_lines = []
                file_idx = file_idx + 1
            idx = idx + 1

if __name__ == '__main__':
    main()
