DIR=./biwei_dialog0
python3 ${DIR}/train_sampler.py \
        -train_corpus ../../../alexchao2007/code/weibo_data_final/10000/pos_unprocessed_10000_train_meteor.tsv \
        -valid_corpus ../../../alexchao2007/code/weibo_data_final/10000/pos_unprocessed_10000_dev_meteor.tsv \
        -train_data ./data/sampler_data/pos_unprocessed_10000_train_meteor_index.tsv \
        -valid_data ./data/sampler_data/pos_unprocessed_10000_dev_meteor_index.tsv \
        -latent_sentence_dict ./data/sampler_data/5w_latent.pkl \
        -vocab_data ./data/JAT_weibo_data/ \
        -report ./report/sampler_pretrain/ \
        -gpuid 0
