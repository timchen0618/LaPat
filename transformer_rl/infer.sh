#!/bin/bash
python3 main.py -test -batch_size 16 \
		-exp test0522 \
		-dropout 0.2 \
        -gdrop 0.2 \
        -sampler_label meteor \
		-filename pred_rl_meteor_10000_41k.txt \
		-test_corpus /share/home/timchen0618/data/weibo-stc/weibo_utf8/final_data_with_pos/meteor_data/10000/new_data_test_10000_meteor.tsv \
        -sampler ./train_model/rl_meteor10000_last/41k_3.174658_0.1032_0.0025_sampler.pth   \
		-transformer ./train_model/rl_meteor10000_last/41k_3.174658_0.1032_0.0025_transformer.pth   \
        -g_hidden 512 \
        -num_classes 10000 \
        -beam_size 3 \
        -block_ngram 3 \
        -pos_dict_path /share/home/timchen0618/data/weibo-stc/weibo_utf8/final_data_with_pos/meteor_data/10000/structure_dict_unprocessed_10000.pkl
		#-transformer ./train_model/rl0709_addbase/10w_3.725040_0.0926_0.0014_transformer.pth \
		#-sampler ./train_model/rl0709_addbase/10w_3.725040_0.0926_0.0014_sampler.pth \
