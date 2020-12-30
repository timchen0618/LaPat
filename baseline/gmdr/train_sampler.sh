DIR=./biwei_dialog0
python ${DIR}/train_sampler.py \
    -config ./config.yml \
    -train_data /share/home/vpj870331/prepare_data_for_original/data/training_sampler_data.tsv \
    -vocab ./data/weibo/weibo.vocab.pkl \
    -log_dir ./out/log_dir \
    -gpuid 3 #> log_files/pretrain_sampler_ourdata.txt
    #./data/weibo/sampler_data.tsv \
