import argparse
from solver import Solver

def parse():
    parser = argparse.ArgumentParser(description="tree transformer")
    parser.add_argument('-train', action='store_true',help='whether train whole model')
    parser.add_argument('-test', action='store_true',help='whether test')
    parser.add_argument('-model_dir',default='train_model',help='output model weight dir')
    parser.add_argument('-gpuid', default=[], nargs='+', type=int)

    # loading model path
    parser.add_argument('-load', default='./train_model/model.pth', help= 'load: model_dir', dest='load_model')

    #training config
    parser.add_argument('-batch_size', type=int, default=64, help='batch size')
    parser.add_argument('-num_epoch', default = 100, type = int)
    parser.add_argument('-src_max_len', default = 50, type = int)
    parser.add_argument('-max_len', type=int, default=20, help='maximum output length', dest= 'max_len')
    parser.add_argument('-multi', action='store_true')
    parser.add_argument('-lr', default=1, type = float)
    parser.add_argument('-g_lr', default=5, type=float)
    parser.add_argument('-disable_comet', action='store_true')
    parser.add_argument('-pretrain_model', type=str, default='model.pth')
    parser.add_argument('-warmup_steps', type=int, default=8000)

    #network config
    parser.add_argument("-pointer_gen", action='store_true')
    parser.add_argument("-dropout", default=0.1, type=float)
    parser.add_argument("-num_layer", default = 6, type=int)
    parser.add_argument('-num_classes', type=int, default=498)
    parser.add_argument('-generator_drop', type=float, default=0.1)


    # data
    # parser.add_argument("-corpus", default = "../multi-response/data/weibo_utf8/seq2seq_data/train_weibo_seq2seq_1.txt", type = str)
    # parser.add_argument("-test_corpus", default = "../multi-response/data/weibo_utf8/seq2seq_data/test_weibo_seq2seq_1.txt", type = str)
    # parser.add_argument("-valid_corpus", default = "../multi-response/data/weibo_utf8/seq2seq_data/valid_weibo_seq2seq_1.txt", type = str)
    parser.add_argument('-data_path', default='/share/home/timchen0618/data/weibo-stc/weibo_utf8/data_with_pos/', type=str)
    parser.add_argument("-corpus", default = "pos_unprocessed_498_latent_train.tsv", type = str)
    parser.add_argument("-test_corpus", default = "pos_unprocessed_498_latent_test.tsv", type = str)
    parser.add_argument("-valid_corpus", default = "pos_unprocessed_498_latent_valid.tsv", type = str)
    parser.add_argument('-pos_dict_path', default='pos_unprocessed_structure_dict.pkl', help='dict_dir')
    parser.add_argument("-vocab", type=str, default = "./5w_pos.json")


    parser.add_argument("-exp_name", type = str)

    parser.add_argument('-sampler_label', default='align', type=str)
    parser.add_argument('-processed', action='store_true')
    parser.add_argument('-print_every_step', default=500, type=int)
    parser.add_argument('-valid_every_step', default=10000, type=int)
    parser.add_argument('-save_checkpoints', action='store_true')

    #output options
    parser.add_argument('-pred_dir', default='./pred_dir/', help='prediction dir', dest='pred_dir')
    parser.add_argument('-filename', default='pred.txt', help='prediction file', dest='filename')

    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    args = parse()
    solver = Solver(args)
    
    if args.train:
        solver.train()
    elif args.test:
        solver.test()

