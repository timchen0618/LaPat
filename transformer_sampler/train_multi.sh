#!/bin/bash
python3 main.py -train -logdir ./log/ \
	-exp_name sampler0415 \
	-dropout 0.2 \
	-batch_size 128 \
	-corpus pos_unprocessed_498_sampler_train_multi.tsv \
	-test_corpus pos_unprocessed_498_sampler_test_multi.tsv \
	-valid_corpus pos_unprocessed_498_sampler_valid_multi.tsv \
	-multi
