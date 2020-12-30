DIR=./biwei_dialog0
python ${DIR}/train_rl.py \
    -train_data ./data/weibo/rl_data3.tsv \
    -config ./config.yml \
    -vocab ./data/weibo/weibo.vocab.pkl \
    -seq2seq /share/home/alexchao2007/code/multi_response_original/multi_response/out/seq_pretrained/checkpoint_epoch8.pkl \
    -sampler /share/home/alexchao2007/code/multi_response_original/multi_response/out/sampler_pretrained/checkpoint_epoch6.pkl \
    -log_dir ./out/log \
    -gpuid 1
