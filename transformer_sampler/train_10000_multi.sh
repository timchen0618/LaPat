#!/bin/bash
python3 main.py -train \
		-exp_name sampler_meteor_10000_0908 \
		-model_dir train_model \
		-dropout 0.2 \
		-generator_drop 0.2 \
		-batch_size 512 \
		-num_classes 10000 \
		-data_path /share/home/timchen0618/data/weibo-stc/weibo_utf8/final_data_with_pos/meteor_data/10000/ \
		-pos_dict_path structure_dict_unprocessed_10000.pkl \
		-sampler_label meteor \
		-print_every_step 1000 \
		-valid_every_step 10000 \
		-corpus pos_unprocessed_10000_train_meteor_sampler_multi.tsv \
		-valid_corpus pos_unprocessed_10000_dev_meteor_sampler_multi.tsv \
		-warmup_steps 4000 \
		-multi \
		-save_checkpoints \
        -lr 1 \
        -g_lr 10 \
		-pretrain_model /share/home/timchen0618/transformer/train_model/seq2seq/base_0619_drop02/119_w2.035721_model.pth 
        #-pretrain_model ../transformer/train_model/seq2seq/base_0826_lr1_batch512_adamw/4_w1.324053__0.3056_0.0780_model.pth \
