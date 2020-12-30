#!/bin/bash
python3 main.py -test -batch_size 32 \
		-exp test0522 \
		-dropout 0.2 \
        -gdrop 0.2 \
        -sampler_label align \
		-filename pred_rl_meteor_enf_0904.txt \
		-test_corpus /share/home/timchen0618/data/weibo-stc/weibo_utf8/final_data_with_pos/align_data/new_data_test_498_align.tsv \
		-sampler ./train_model/rl_enf_0903/2w_43.235466_0.0781_0.0009_sampler.pth \
		-transformer ./train_model/rl_enf_0903/2w_43.235466_0.0781_0.0009_transformer.pth \
        -g_hidden 512 \
        -num_classes 498 \
        -pos_dict_path /share/home/timchen0618/data/weibo-stc/weibo_utf8/final_data_with_pos/align_data/pos_unprocessed_structure_dict.pkl
		#-transformer ./train_model/rl0709_addbase/10w_3.725040_0.0926_0.0014_transformer.pth \
		#-sampler ./train_model/rl0709_addbase/10w_3.725040_0.0926_0.0014_sampler.pth \
