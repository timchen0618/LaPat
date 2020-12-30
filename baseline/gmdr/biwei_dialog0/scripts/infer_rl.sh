DIR=
python3 ${DIR}/infer_rl.py \
    -src_in \
    -tgt_out \
    -model \
    -data \
    -config ./config.yml \
    -gpuid \
    -beam_size 3 \
    -topk_tag 1 \
    -decode_max_length 20