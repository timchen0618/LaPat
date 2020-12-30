#!/bin/bash
python3 main.py -test \
        -exp_name test0508 \
        -dropout 0.4 \
        -generator_drop 0.4 \
        -batch_size 128 \
        -test_corpus new_data_test_498_align.tsv \
        -num_classes 498 \
        -pos_dict_path pos_unprocessed_structure_dict.pkl \
        -data_path /share/home/timchen0618/data/weibo-stc/weibo_utf8/final_data_with_pos/align_data \
        -filename pred_align_0901.txt \
        -load train_model/align/sampler_align_un_drop04_gdrop04_cls_preembed/59k_31.755777_0.626572model.pth
        #-load train_model/meteor/sampler_meteor_1000_un_drop02_gdrop02_lr01_glr5_adamw_prefull/179k_59.328290_0.060547model.pth \
