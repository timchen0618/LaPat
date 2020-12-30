DIR=./biwei-dialog0
python3 ${DIR}/build_vocab.py \
    -train_data ../data/weibo/raw_src_tgt.tsv \
    -save_data weibo \
    -config ./config.yml
