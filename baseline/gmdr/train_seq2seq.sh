DIR=./biwei_dialog0
python ${DIR}/train_seq2seq.py \
    -config ./config.yml \
    -vocab ./data/weibo/weibo.vocab.pkl \
    -train_data ./data/weibo/seq2seq_data_top1.tsv \
    -log_dir ./out/log \
    -gpuid 1

