#!/bin/bash
python3 main.py -train \
		-sampler_label align \
		-model_dir train_model \
		-task pure_seq2seq \
 		-exp_name lr_-3_nodecay \
		-model_type transformer \
		-w_valid_file valid.pure.3.txt \
		-w_valid_tgt_file valid.tgt.txt \
        -save_checkpoints
