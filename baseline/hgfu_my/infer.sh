#!/bin/bash
python3 main.py -test \
                -model_name HGFU \
                -pred_dir pred_dir \
                -prediction pred_lr-3_2.txt \
                -load train_model/lr-3/99k_2.788304_model.pth \
                -config ./config/hgfu/config.yml
