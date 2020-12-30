#!/bin/bash
python3 main.py -train \
        -exp_name rl_align_0907_fast_eos \
        -dropout 0.2 \
		-gdrop 0.2 \
        -g_hidden 512 \
		-batch_size 32 \
        -num_classes 498 \
		-sampler_label align \
        -sampler ../transformer_sampler/train_model/align/sampler_align_un_drop04_gdrop04_cls_preembed/59k_31.755777_0.626572model.pth \
        -transformer ../transformer/train_model/seq2seq/base_0619_drop02/99_w2.047761_model.pth \
		-corpus /share/home/timchen0618/data/weibo-stc/weibo_utf8/final_data_with_pos/align_data/pos_unprocessed_498_train_align.tsv  \
		-valid_corpus /share/home/timchen0618/data/weibo-stc/weibo_utf8/final_data_with_pos/align_data/pos_unprocessed_498_dev_align.tsv  \
        -print_every_steps 500 \
		-valid_every_steps 1000 \
		-w_valid_file valid.align.txt \
		-w_valid_tgt_file valid.tgt.txt \
		-reward_type f1 \
		-save_checkpoints \
		-pos_dict_path /share/home/timchen0618/data/weibo-stc/weibo_utf8/final_data_with_pos/align_data/pos_unprocessed_structure_dict.pkl
		#-pos_masking \
		#-sampler ../transformer_sampler/train_model/align/sampler_align_un_drop04_gdrop04_cls_preembed/19k_31.852326_0.717572model.pth \
		#-transformer ../transformer/train_model/seq2seq/base_0619_drop02/129_w2.037903_model.pth \
