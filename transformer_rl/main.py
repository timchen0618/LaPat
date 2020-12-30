import argparse
from solver import Solver

def parse():
    parser = argparse.ArgumentParser(description="tree transformer")
    parser.add_argument('-train', action='store_true',help='whether train whole model')
    parser.add_argument('-test', action='store_true',help='whether test')
    parser.add_argument('-model_dir',default='train_model',help='output model weight dir')
    parser.add_argument('-gpuid', default=[], nargs='+', type=int)

    # loading model path
    parser.add_argument('-rl_model', default='./train_model/model.pth', help= 'load: model_dir')
    parser.add_argument('-sampler', default='', help= 'sampler model path')
    parser.add_argument('-transformer', default='', help= 'transformer model path')

    # output
    parser.add_argument('-print_every_steps', default=500, type=int)
    parser.add_argument('-valid_every_steps', default = 50000, type=int)

    #training config
    parser.add_argument('-batch_size', type=int, default=64, help='batch size')
    parser.add_argument('-num_epoch', default = 20, type = int)
    parser.add_argument('-src_max_len', default = 50, type = int)
    parser.add_argument('-max_len', type=int, default=20, help='maximum output length', dest= 'max_len')
    parser.add_argument('-start_step', type=int, default=0)
    parser.add_argument('-pos_masking', action='store_true')
    parser.add_argument('-posmask', type=str, default='../transformer/processing/posmask.json')
    parser.add_argument('-reward_type', type=str, default='f1')

    #network config
    parser.add_argument("-pointer_gen", action='store_true')
    parser.add_argument("-dropout", default=0.1, type=float)
    parser.add_argument('-gdrop', default=0.1, type=float)
    parser.add_argument('-pred_pos_drop', default=0.3, type=float)
    parser.add_argument("-num_layer", default = 6, type=int)
    parser.add_argument("-num_classes", default = 498, type=int)
    parser.add_argument("-g_hidden", default=512, type=int)
    parser.add_argument('-disable_comet', action='store_true')
    parser.add_argument('-pred_pos', action='store_true')
    # parser.add_argument("")

    # data
    # parser.add_argument("-corpus", default = "../multi-response/data/weibo_utf8/seq2seq_data/train_weibo_seq2seq_1.txt", type = str)
    # parser.add_argument("-test_corpus", default = "../multi-response/data/weibo_utf8/seq2seq_data/test_weibo_seq2seq_1.txt", type = str)
    # parser.add_argument("-valid_corpus", default = "../multi-response/data/weibo_utf8/seq2seq_data/valid_weibo_seq2seq_1.txt", type = str)
    parser.add_argument("-corpus", default = "/share/home/timchen0618/data/weibo-stc/weibo_utf8/data_with_pos/align_data/pos_unprocessed_498_latent_train.tsv", type = str)
    parser.add_argument("-test_corpus", default = "/share/home/timchen0618/data/weibo-stc/weibo_utf8/data_with_pos/align_data/pos_unprocessed_498_latent_test.tsv", type = str)
    parser.add_argument("-valid_corpus", default = "/share/home/timchen0618/data/weibo-stc/weibo_utf8/data_with_pos/align_data/pos_unprocessed_498_latent_valid.tsv", type = str)
    parser.add_argument('-pos_dict_path', default='/share/home/timchen0618/data/weibo-stc/weibo_utf8/data_with_pos/align_data/pos_unprocessed_structure_dict.pkl', help='dict_dir')
    parser.add_argument("-vocab", type=str, default = "./5w.json")
    parser.add_argument('-w_valid_file', type=str, default='valid.txt')
    parser.add_argument('-w_valid_tgt_file', type = str, default='valid.tgt.txt')

    parser.add_argument('-sampler_label', type=str, default='align')
    parser.add_argument('-processed', action='store_true')

    parser.add_argument("-logdir", default = './log/', type = str)
    parser.add_argument("-exp_name", type = str)
    parser.add_argument('-save_best_only', action='store_true')
    parser.add_argument('-save_checkpoints', action='store_true')


    #output options
    parser.add_argument('-beam_size', default=1, type=int)
    parser.add_argument('-block_ngram', default=0, type=int)
    parser.add_argument('-pred_dir', default='./pred_dir/', help='prediction dir', dest='pred_dir')
    parser.add_argument('-filename', default='pred.txt', help='prediction file', dest='filename')
    parser.add_argument('-pos_file', default='pos.txt')

    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    args = parse()
    solver = Solver(args)
    
    if args.train:
        solver.train()
    elif args.test:
        solver.test()

