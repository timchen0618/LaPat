DIR=./biwei_dialog0
python ${DIR}/infer_rl.py \
    -test_data /share/home/alexchao2007/code/weibo_data_final/10000/new_data_test_10000_meteor.tsv \
    -test_out ./result_rl_test_epoch6_sortinfornt.test \
    -model ./out/rl_original/checkpoint_epoch6.pkl \
    -vocab ./data/weibo/weibo.vocab.pkl \
    -config ./config.yml \
    -gpuid 2 \
    -beam_size 5 \
    -topk_tag 1000 \
    -num_cluster 3 \
    -dump_beam dump_beam.json \
    -decode_max_length 20
