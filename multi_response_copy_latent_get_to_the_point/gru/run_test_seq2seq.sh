DIR=./biwei_dialog0
python3 ${DIR}/test_seq2seq.py \
        -test_corpus ./data/2stage_data/2stage_testing_data.tsv \
        -test_data ./data/2stage_data/2stage_testing_data_index.tsv \
        -vocab_data ./data/JAT_weibo_data \
        -report ./report/ \
        -gpuid 0
