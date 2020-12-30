#!/bin/bash
python3 main.py -train \
        -exp_name rl_enf_0903 \
        -dropout 0.2 \
		-gdrop 0.2 \
		-batch_size 32 \
		-sampler ../transformer_sampler/train_model/align/sampler_align_un_drop04_gdrop04_cls_preembed/19k_31.852326_0.717572model.pth \
		-transformer ../transformer/train_model/seq2seq/base_0619_drop02/129_w2.037903_model.pth \
		-sampler_label align \
		-corpus /share/home/timchen0618/data/weibo-stc/weibo_utf8/final_data_with_pos/align_data/pos_unprocessed_498_train_align.tsv \
		-valid_corpus /share/home/timchen0618/data/weibo-stc/weibo_utf8/final_data_with_pos/align_data/pos_unprocessed_498_dev_align.tsv \
		-print_every_steps 500 \
		-valid_every_steps 25000 \
		-w_valid_file valid.enf.txt \
		-w_valid_tgt_file valid.tgt.txt \
		-pos_masking \
		-reward_type f1 \
		-save_checkpoints
