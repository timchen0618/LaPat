#!/bin/bash
python3 main.py -test \
                -exp_name test \
                -filename pred_2stage_genpos_final.txt \
                -task seq2seq \
                -load ../transformer/train_model/seq2seq/base_0619_drop02/99_w2.047761_model.pth 
                #-load train_model/seq2seq/post_pos/89k_2.709873__0.1104_0.0034_model.pth
                #-load train_model/seq2seq/base_0619_drop02/109_w2.045436_model.pth
                #-load /share/home/timchen0618/transformer/train_model/seq2seq/base_0826_lr1_batch512_adamw/24k_1.384933__0.2954_0.0693_model.pth \
                #-test_corpus ../multi-response/data/weibo_utf8/data_with_pos/align_data/pos_processed_498_latent_test.tsv \
                #-pos_dict_path ../multi-response/data/weibo_utf8/data_with_pos/align_data/pos_processed_structure_dict.pkl \
                #-load train_model/seq2seq/base_0619_nodrop/34_w2.048403_model.pth \
                #-load train_model/seq2seq/base_0619_drop02/129_w2.037903_model.pth \
                #-load train_model/seq2seq/base_0619_drop02/109_w2.045436_model.pth
