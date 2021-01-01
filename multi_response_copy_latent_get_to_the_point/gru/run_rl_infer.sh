DIR=./biwei_dialog0
python3 ${DIR}/test_rl.py \
    -test_corpus ../../../alexchao2007/code/weibo_data_final/10000/new_data_test_10000_meteor.tsv \
    -test_data ./data/rl_data/10000/new_data_test_10000_meteor_index.tsv \
    -latent_sentence_dict ./data/sampler_data/5w_latent.pkl \
    -vocab_data ./data/JAT_weibo_data/ \
    -RL_config ${DIR}/config.yml \
    -RL_model checkpoint_epoch0.pkl \
    -report ./report/RL/ \
    -gpuid 3
