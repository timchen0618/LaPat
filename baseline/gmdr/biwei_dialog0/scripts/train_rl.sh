DIR=./biwei-dialog0
python3 ${DIR}/train_rl.py \
    -train_src \
    -train_tgt \
    -train_tag \
    -config ./config.yml \
    -vocab weibo-rl.vocab.pkl \
    -seq2seq \
    -sampler \
    -log_dir ./out/log \
    -gpuid 0