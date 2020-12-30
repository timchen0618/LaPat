#!/bin/bash
python3 main.py -test \
                -model_name Seq2Seq \
                -pred_dir pred_dir \
                -prediction pred_pureseq2seq_lr2e-4_corr.txt \
                -load train_model/pureseq2seq_lr2e-4/99k_4.949401_model.pth \
                -config ./config/seq2seq/config.yml
