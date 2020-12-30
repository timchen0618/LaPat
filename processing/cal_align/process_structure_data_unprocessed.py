import numpy as np
import pickle
from Bio import pairwise2
import pdb
import sys
import nltk.translate.meteor_score as meteor_score
import nltk as nltk
def main():
    # start from tmux window 21
    # 30-36: alphalady
    # 36-42 alphaman
    # 10-29 amethyst + 8,9
    # next:21
    input_idx = int(sys.argv[1])
    input_file = 'files/unprocessed_strct_' + str(input_idx) + '.tsv'
    output_file = 'unprocessed_files/unprocessed_file_' + str(input_idx) + '.tsv'

    # freq_dict_file = '../seq2seq_bm25_postags_freq_original_dict.pkl'

    # freq_dict = pickle.load(open(freq_dict_file, 'rb'))
    dict_file = '/share/home/alexchao2007/code/weibo_data_final/1000/pos_unprocessed_dict_meteor1000.pkl'

    # rearrange struct from frequency
    '''
    struct_dict_array = []
    for k, v in freq_dict.items():
        struct_dict_array.append((k, v))
    sorted_struct_dict_array = sorted(struct_dict_array, key=lambda x: x[1], reverse=True)
    struct_list = sorted_struct_dict_array[:1000]

    structure_dict = {}
    idx2structure = {}
    structure2idx = {}

    idx = 0
    for each_struct, _ in struct_list:
            idx2structure[idx] = each_struct
            structure2idx[each_struct] = idx
            idx = idx + 1
    structure_dict['idx2structure'] = idx2structure
    structure_dict['structure2idx'] = structure2idx
    '''
    
    '''
    with open(dict_file, 'wb') as w_file:
        pickle.dump(structure_dict, w_file)
    '''

    structure_dict = pickle.load(open(dict_file, 'rb'))
    
    struct_list = []
    for k, v in structure_dict['structure2idx'].items():
        struct_list.append(k)

    with open(output_file, 'w') as struct_idx_file:
        with open(input_file, 'r') as readlines:
            for idx, line in enumerate(readlines):
                print('processing idx: ', input_idx, 'unprocessed: ', idx, '/30000')
                _, _, _, structure, _, _ = line.split('\t')

                max_struct_score = -100
                max_struct_idx = 0
            
                for idx, latent_struct in enumerate(struct_list):
                    # max_score = meteor_score.single_meteor_score(latent_struct, structure) 
                    # if latent_struct=='v n n n n n n n n n n n n n n n n n n n n n n n n n n n' or structure=='v n n n n n n n n n n n n n n n n n n n n n n n n n n n':
                    #     continue           
                    max_score = pairwise2.align.globalxx(latent_struct, structure)[0][2]
                    if max_score > max_struct_score:
                        max_struct_score = max_score
                        max_struct_idx = structure_dict['structure2idx'][latent_struct]

                line = '\t'.join(line[:-1].split('\t')[:-1])
                struct_idx_file.write(line)
                struct_idx_file.write('\t')
                struct_idx_file.write(str(max_struct_idx))
                struct_idx_file.write('\n')

if __name__ == '__main__':
    main()