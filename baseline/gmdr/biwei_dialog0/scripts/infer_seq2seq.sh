DIR=
python3 ${DIR}/infer_seq2seq.py \
    -test_data \
    -tgt_out \
    -seq2seq \
    -sampler \
    -vocab \
    -config ./config.yml \
    -gpuid 0 \
    -beam_size 3 \
    -topk_tag 1 \
    -decode_max_length 20