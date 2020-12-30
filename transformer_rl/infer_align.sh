#!/bin/bash
python3 main.py -test -batch_size 32 \
		-exp test0522 \
		-dropout 0.2 \
        -gdrop 0.2 \
        -sampler_label align \
		-filename pred_rl_align_0909.txt \
		-test_corpus /share/home/timchen0618/data/weibo-stc/weibo_utf8/final_data_with_pos/align_data/post_ref_800.tsv \
        -g_hidden 512 \
        -num_classes 498 \
        -pos_dict_path /share/home/timchen0618/data/weibo-stc/weibo_utf8/final_data_with_pos/align_data/pos_unprocessed_structure_dict.pkl \
        -block_ngram 2 \
        -beam_size 3 \
		-transformer ./train_model/rl_align_0907_fast_eos/$1_transformer.pth   \
		-sampler ./train_model/rl_align_0907_fast_eos/$1_sampler.pth 
        #-sampler ../transformer_sampler/train_model/align/sampler_align_un_drop04_gdrop04_cls_preembed/59k_31.755777_0.626572model.pth \
        #-transformer ../transformer/train_model/seq2seq/base_0619_drop02/99_w2.047761_model.pth
		#-sampler ./train_model/rl0709_addbase/10w_3.725040_0.0926_0.0014_sampler.pth \
		#-transformer ./train_model/rl0709_addbase/10w_3.725040_0.0926_0.0014_transformer.pth \
