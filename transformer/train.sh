#!/bin/bash
python3 main.py -train \
		-sampler_label align \
		-model_dir train_model \
		-task seq2seq \
 		-exp_name test \
		-model_type transformer \
		-w_valid_file valid.enforce.txt \
		-w_valid_tgt_file valid.tgt.txt \
		-start_step 1300000 \
		-load train_model/seq2seq/base_0709_nodrop_tie_enforce_fast/129_w2.699536__0.2952_0.0706_model.pth \
		-disable_comet
