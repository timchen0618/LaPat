import argparse
from solver import Solver


def parse():
    parser = argparse.ArgumentParser(description="tree transformer")

    parser.add_argument('-load', default=None, help= 'load: model_dir', dest= 'load_model', type=str)
    parser.add_argument('-train', action='store_true',help='whether train whole model')
    parser.add_argument('-test', action='store_true',help='whether test')
    parser.add_argument('-task', default='seq2seq', type=str)

    parser.add_argument('-model_dir',default='train_model',help='output model weight dir')
    # parser.add_argument("-logdir", default = './log/', type = str)
    parser.add_argument('-pred_dir', default='./pred_dir/', help='prediction dir', dest='pred_dir')
    parser.add_argument('-filename', default='pred.txt', help='prediction file', dest='filename')
    parser.add_argument('-pretrain_embedding', default=None)

    parser.add_argument('-sampler_label', type=str, default='none')
    parser.add_argument('-processed', action='store_true')

    parser.add_argument('-config', type=str, default='config.yml')
    parser.add_argument('-exp_name', type=str, help='experiment name')
    parser.add_argument('-disable_comet', action='store_true')
    parser.add_argument('-model_type', default='transformer', type=str)
    parser.add_argument('-save_checkpoints', action='store_true')
    
    parser.add_argument('-w_valid_file', type=str)
    parser.add_argument('-w_valid_tgt_file', type=str, default='valid.tgt.txt')

    parser.add_argument('-start_step', type = int, default=0)
    parser.add_argument('-gpuid', type=int, default=0)
    # parser.add_argument('-gpuid', default=[], nargs='+', type=int)

    # parser.add_argument('-batch_size', type=int, default=64, help='batch size')s
    # parser.add_argument('-num_epoch', default = 20, type = int)
    
    # parser.add_argument("-pointer_gen", action='store_true')
    # parser.add_argument("-dropout", default=0.0, type=float)
    # parser.add_argument("-num_layer", default = 6, type=int)

    # parser.add_argument("-corpus", default = "../multi-response/data/weibo_utf8/seq2seq_data/train_weibo_seq2seq_1.txt", type = str)
    # parser.add_argument("-test_corpus", default = "../multi-response/data/weibo_utf8/seq2seq_data/test_weibo_seq2seq_1.txt", type = str)
    # parser.add_argument("-valid_corpus", default = "../multi-response/data/weibo_utf8/seq2seq_data/valid_weibo_seq2seq_1.txt", type = str)
    # parser.add_argument("-corpus", default = "./data/seq2seq_bm25_sentence_pos_processed_train_corpus.tsv", type = str)
    # parser.add_argument("-test_corpus", default = "./data/seq2seq_bm25_sentence_pos_processed_test_corpus.tsv", type = str)
    # parser.add_argument("-valid_corpus", default = "./data/seq2seq_bm25_sentence_pos_processed_valid_corpus.tsv", type = str)
    
    

    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    args = parse()
    solver = Solver(args)
    
    if args.train:
        solver.train()
    elif args.test:
        solver.test()











