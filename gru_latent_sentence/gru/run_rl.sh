DIR=./biwei_dialog0
python3 ${DIR}/train_rl.py \
    -train_corpus ../../../alexchao2007/code/weibo_data_final/10000/pos_unprocessed_10000_train_meteor.tsv \
    -valid_corpus ../../../alexchao2007/code/weibo_data_final/10000/pos_unprocessed_10000_dev_meteor.tsv \
    -train_data ./data/rl_data/10000/pos_unprocessed_10000_train_meteor_index.tsv \
    -valid_data ./data/rl_data/10000/pos_unprocessed_10000_dev_meteor_index.tsv \
    -latent_sentence_dict ./data/sampler_data/5w_latent.pkl \
    -vocab_data ./data/JAT_weibo_data/ \
    -RL_config ${DIR}/config.yml \
    -sampler_model ./out/sampler/checkpoint_epoch0.pkl \
    -seq2seq_model ./out/seq2seq/checkpoint_epoch0.pkl \
    -report ./report/RL/ \
    -gpuid 3
