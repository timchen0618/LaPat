#!/bin/bash
python3 main.py -test -batch_size 32 \
		-exp test0522 \
		-dropout 0.2 \
        -gdrop 0.2 \
        -sampler_label align \
        -filename pred_rl_meteor_genpos_fast_0907_51k_beam3_block3.txt \
		-test_corpus /share/home/timchen0618/data/weibo-stc/weibo_utf8/final_data_with_pos/meteor_data/1000/new_data_test_1000_meteor.tsv \
		-sampler ./train_model/rl_meteor1000_0907_genpos_fast/51k_3.266715_0.1094_0.0024_sampler.pth \
		-transformer ./train_model/rl_meteor1000_0907_genpos_fast/51k_3.266715_0.1094_0.0024_transformer.pth \
        -pred_pos \
        -beam_size 3 \
        -block_ngram 3 \
        -pos_dict_path /share/home/timchen0618/data/weibo-stc/weibo_utf8/final_data_with_pos/meteor_data/1000/pos_unprocessed_dict_meteor1000.pkl
		#-transformer ./train_model/rl0709_addbase/10w_3.725040_0.0926_0.0014_transformer.pth \
		#-sampler ./train_model/rl0709_addbase/10w_3.725040_0.0926_0.0014_sampler.pth \
