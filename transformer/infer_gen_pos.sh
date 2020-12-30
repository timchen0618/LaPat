#!/bin/bash
python3 main.py -test \
                -exp_name test \
                -load train_model/pred_pos/pred_pos_lr01_pre_drop03/22_w1.618957__0.4327_0.0646_model.pth \
                -filename pred_pos_on_new_test_2stage.txt \
                -task pred_pos 
                #-test_corpus ../multi-response/data/weibo_utf8/data_with_pos/align_data/pos_processed_498_latent_test.tsv \
                #-pos_dict_path ../multi-response/data/weibo_utf8/data_with_pos/align_data/pos_processed_structure_dict.pkl \
                #-load train_model/seq2seq/base_0619_nodrop/34_w2.048403_model.pth \
