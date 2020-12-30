#!/bin/bash
python3 main.py -train \
		-sampler_label align \
		-model_dir train_model \
		-task seq2seq \
 		-exp_name finetune_from_ptt_lrsmall \
		-model_type transformer \
		-w_valid_file valid.finetune.txt \
		-w_valid_tgt_file valid.tgt.txt \
		-start_step 0 \
		-load train_model/pretrain_seq2seq/checkpoint_epoch9.pkl \
		-save_checkpoints 
