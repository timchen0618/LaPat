#!/bin/bash
python3 main.py -train \
        -exp_name rl_meteor1000_0907_genpos_fast_eos_former \
        -dropout 0.2 \
		-gdrop 0.2 \
        -g_hidden 1024 \
		-batch_size 32 \
        -num_classes 1000 \
		-sampler_label meteor \
        -sampler ../transformer/train_model/pred_pos/pred_pos_lr01_pre_drop03/22_w1.618957__0.4327_0.0646_model.pth \
        -transformer ../transformer/train_model/seq2seq/base_0619_drop02/99_w2.047761_model.pth \
		-corpus /share/home/timchen0618/data/weibo-stc/weibo_utf8/final_data_with_pos/meteor_data/1000/pos_unprocessed_1000_train_meteor.tsv  \
		-valid_corpus /share/home/timchen0618/data/weibo-stc/weibo_utf8/final_data_with_pos/meteor_data/1000/pos_unprocessed_1000_dev_meteor.tsv  \
        -print_every_steps 500 \
		-valid_every_steps 1000 \
		-w_valid_file valid.gen1.txt \
		-w_valid_tgt_file valid.tgt.txt \
		-reward_type f1 \
		-save_checkpoints \
        -pred_pos \
		-pos_dict_path /share/home/timchen0618/data/weibo-stc/weibo_utf8/final_data_with_pos/small_data/1000/pos_unprocessed_dict_meteor1000.pkl 
		#-valid_corpus /share/home/timchen0618/data/weibo-stc/weibo_utf8/final_data_with_pos/small_data/1000/pos_unprocessed_1000_latent_meteor_valid.tsv  \
        #-transformer ../transformer/train_model/seq2seq/base_0826_lr1_batch512_adamw/199k_1.265833__0.3207_0.0919_model.pth \
		#-pos_masking \
		#-sampler ../transformer_sampler/train_model/align/sampler_align_un_drop04_gdrop04_cls_preembed/19k_31.852326_0.717572model.pth \
		#-transformer ../transformer/train_model/seq2seq/base_0619_drop02/129_w2.037903_model.pth \
