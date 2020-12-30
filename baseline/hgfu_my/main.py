import argparse
from solver import Solver


def parse():
    parser = argparse.ArgumentParser(description="tree transformer")

    parser.add_argument('-load', default=None, help= 'load: model_dir', dest= 'load_path', type=str)
    parser.add_argument('-train', action='store_true',help='whether train whole model')
    parser.add_argument('-test', action='store_true',help='whether test')

    parser.add_argument('-disable_comet', action='store_true')
    parser.add_argument('-exp_name', type=str)
    parser.add_argument('-no_cuda', action='store_true')

    parser.add_argument('-config', type=str, default='config.yml')
    parser.add_argument('-model_name', type=str, default='HGFU')

    parser.add_argument('-model_path', type=str, default='train_model')
    parser.add_argument('-save_checkpoints', action='store_true')

    parser.add_argument('-pred_dir', type=str, default='./pred_dir/')
    parser.add_argument('-prediction', type=str, default='pred.txt')
    parser.add_argument('-w_valid_file', type=str, default='./w_valid/valid.txt')
    parser.add_argument('-w_valid_tgt_file', type=str, default='./w_valid/valid.tgt.txt')


    

    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    args = parse()
    solver = Solver(args)
    print('[Logging Info] Finish building solver...')
    if args.train:
        solver.train()
    elif args.test:
        solver.test()
    # solver.cal_number_of_templates()










