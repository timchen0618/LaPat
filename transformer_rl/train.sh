#!/bin/bash
python3 main.py -train \
        -exp_name rl_meteor1000_0831 \
        -dropout 0.2 \
		-gdrop 0.2 \
        -g_hidden 1024 \
		-batch_size 32 \
        -num_classes 1000 \
		-sampler_label meteor \
        -sampler ../transformer_sampler/train_model/meteor/sampler_meteor_1000_un_drop02_gdrop02_lr01_glr5_adamw_prefull/179k_59.328290_0.060547model.pth \
        -transformer ../transformer/train_model/seq2seq/base_0826_lr1_batch512_adamw/199k_1.265833__0.3207_0.0919_model.pth \
		-corpus /share/home/timchen0618/data/weibo-stc/weibo_utf8/final_data_with_pos/meteor_data/1000/pos_unprocessed_1000_train_meteor.tsv  \
		-valid_corpus /share/home/timchen0618/data/weibo-stc/weibo_utf8/final_data_with_pos/meteor_data/1000/pos_unprocessed_1000_dev_meteor.tsv  \
        -print_every_steps 500 \
		-valid_every_steps 25000 \
		-w_valid_file valid.txt \
		-w_valid_tgt_file valid.tgt.txt \
		-reward_type f1 \
		-save_checkpoints \
		-pos_dict_path /share/home/timchen0618/data/weibo-stc/weibo_utf8/final_data_with_pos/meteor_data/1000/pos_unprocessed_dict_meteor1000.pkl 
		#-pos_masking \
		#-sampler ../transformer_sampler/train_model/align/sampler_align_un_drop04_gdrop04_cls_preembed/19k_31.852326_0.717572model.pth \
		#-transformer ../transformer/train_model/seq2seq/base_0619_drop02/129_w2.037903_model.pth \
