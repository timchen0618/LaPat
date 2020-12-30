DIR=
python3 ${DIR}/infer_rl.py \
    -test_data \
    -test_out \
    -model \
    -vocab \
    -config ./config.yml \
    -gpuid \
    -beam_size 3 \
    -topk_tag 1 \
    -decode_max_length 20 \
    -num_worker 3