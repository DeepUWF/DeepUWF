import main
import os
from arguments import get_args_parser
import Model.util.misc as misc

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    misc.init_distributed_mode(args) 

    aucs = []
    output_dir = args.output_dir

    for seed in [0, 1, 2, 3, 4]:
        print('Model with seed:', seed)
        args.seed = seed
        args.output_dir = os.path.join(output_dir, f'seed_{seed}')
        auc = main.main(args)
        aucs.append(auc)

    print("5 AUCs:", aucs)