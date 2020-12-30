DIR=./biwei_dialog0
python ${DIR}/infer_seq2seq.py \
    -test_data /share/home/alexchao2007/code/weibo_data_final/10000/new_data_test_10000_meteor.tsv \
    -test_out ./seq2seq_pretrain_4_6.test \
    -vocab ./data/weibo/weibo.vocab.pkl \
    -seq2seq /share/home/alexchao2007/code/multi_response_original/multi_response/out/seq_pretrained/checkpoint_epoch6.pkl \
    -sampler /share/home/alexchao2007/code/multi_response_original/multi_response/out/sampler_pretrained/checkpoint_epoch4.pkl \
    -config ./config.yml \
    -beam_size 5 \
    -gpuid 3 \
    -topk_tag 1000 \
    -decode_max_length 20 \
    -num_cluster 3 \
    -tensorboard ./test/log/test2.json