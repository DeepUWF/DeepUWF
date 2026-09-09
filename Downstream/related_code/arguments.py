import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('UWFound', add_help=False)
    parser.add_argument('--seed', default=0, type=int)
    # Federated learning parameters
    parser.add_argument('--server', action='store_true', help='run as server')
    parser.add_argument('--client', action='store_true', help='run as client')
    ### parser.add_argument('--local_client_num', default=1, type=int, help='the number of local federated client')
    parser.add_argument('--server_addr', type=str, default='localhost:8080', help='server address:port')
    parser.add_argument('--apfl', action='store_true', help='use APFL')
    parser.add_argument('--alpha', type=float, default=0.25, help='alpha for APFL')
    parser.set_defaults(server=False)
    parser.set_defaults(client=False)
    parser.set_defaults(apfl=False)

    # Distillation
    parser.add_argument('--distillation', action='store_true', help='use distillation')
    parser.add_argument('--teacher_model', type=str, default='resnet50', help='teacher model')
    parser.add_argument('--teacher_chkpt', type=str, default=None, help='teacher checkpoint')
    parser.add_argument('--critertion', type=str, default='bce', help='criterion for distillation')
    parser.set_defaults(distillation=False)

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')  # e.g. "vit_large_patch16"
    parser.add_argument('--num_classes', default=2, type=int, metavar='N',)
    parser.add_argument('--multi_label', action='store_true', help='Use multi-label classification')
    #parser.add_argument('--chkpt_path', default='/path/to/workspace/dataset/DeepUWF/DR/finetune_rdr224_pretrain_ffm_checkpoint-best.pth', type=str, help='train from checkpoint')
    #parser.add_argument('--chkpt_path', default='/path/to/workspace/dataset/DeepUWF/SH_DR/finetune_shrdr_pretrain_sh_checkpoint-best.pth', type=str, help='train ricd的时候要用这个分类起')
    parser.add_argument('--chkpt_path', default='/path/to/workspace/weights/DeepUWFDP/pretrain_distill_ffm_checkpoint-best.pth', type=str, help='train from checkpoint')
    #parser.add_argument('--chkpt_path', default='/path/to/workspace/weights/DeepUWFDP/SHDR_NoDPBaseline_adamw_savegrad/model_standalone_best.pt', type=str, help='train from checkpoint')
    #parser.add_argument('--chkpt_path', default='/path/to/workspace/weights/DeepUWFDP/SHDR_ep1_adamw/model_standalone_best.pt', type=str, help='train from checkpoint')

    #parser.add_argument('--chkpt_path', default='/path/to/workspace/weights/DeepUWFDP/SHDR_ep32_adamw/model_standalone_best.pt', type=str, help='train from checkpoint')
   # parser.add_argument('--chkpt_path', default='/path/to/workspace/weights/DeepUWFDP/SHDR_ep8_adamw/model_standalone_best.pt', type=str, help='train from checkpoint')


    parser.add_argument('--input_size', default=448, type=int,  
                        help='images input size')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--norm_pix_loss', action='store_true', help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)
    parser.set_defaults(multi_label=False)

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM', help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR', help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--layer_decay', type=float, default=0.75, help='layer-wise lr decay from ELECTRA/BEiT')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N', help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='', type=str, help='dataset path')
    parser.add_argument('--label', default=None, help='path of the .pkl label, train and test set will be split by 8:2')
    parser.add_argument('--train_label', default='', type=str, help='path of the .pkl label for training')
    parser.add_argument('--test_label', default='', type=str, help='path of the .pkl label for test')
    parser.add_argument('--output_dir', default='./checkpoints', help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./log', help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--task', default='MIM', help='task name for saving checkpoints')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus)')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    
    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    
    # * Finetuning params
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')
    
    # Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # Mode
    parser.add_argument('--pretrain', action='store_true', default=False,
                        help='Perform evaluation only')
    parser.add_argument('--finetune', action='store_true', default=False,
                        help='Perform evaluation only')
    parser.add_argument('--eval', action='store_true', default=True,
                    help='Perform evaluation only')


    # distributed parameters
    parser.add_argument('--distributed', action='store_true', default=False,
                        help='use distributed')

    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    #args = parser.parse_args()
    args,unknown = parser.parse_known_args()

    return args
