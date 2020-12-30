DIR=./biwei_dialog0
python ${DIR}/multi_infer_rl.py \
    -test_data /share/home/alexchao2007/code/weibo_data_final/10000/new_data_test_10000_meteor.tsv \
    -test_out ./result_multi_rl_test.test \
    -model ./out/rl_original/checkpoint_epoch1.pkl \
    -vocab ./data/weibo/weibo.vocab.pkl \
    -config ./config.yml \
    -gpuid 0 \
    -beam_size 5 \
    -topk_tag 1000 \
    -decode_max_length 20 \
    -num_worker 3
