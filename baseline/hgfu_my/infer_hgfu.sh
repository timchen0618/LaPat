#!/bin/bash
python3 main.py -test \
                -model_name HGFU \
                -pred_dir pred_dir \
                -prediction pred_lr-4_0902.txt \
                -load train_model/hgfu_lr1e-4/199k_4.910521_model.pth \
                -config ./config/hgfu/config.yml
