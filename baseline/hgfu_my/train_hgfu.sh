#!/bin/bash
python3 main.py -train \
                -exp_name hgfu_lr1e-3 \
                -model_name HGFU \
                -save_checkpoints \
                -w_valid_file w_valid/valid.hgfu3.txt \
                -w_valid_tgt_file w_valid/valid.tgt.txt \
                -config ./config/hgfu/config.yml 
