import numpy as np
import os
import random

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

import timm
# assert timm.__version__ == "0.3.2"  # version check

import Model.util.misc as misc
from Model.util.misc import NativeScalerWithGradNormCount as NativeScaler
from Model.util.datasets import build_dataset, split_dataset, LabeledDataset, UnlabeledDataset


def prepare_dataset(args):
    '''
    return a dict of keys: "data_loader_train", "data_loader_test", "log_writer"
        args.eval:      (None,               data_loader_test,   None        )
        args.pretrain:  (data_loader_train,  None,               log_writer  )
        args.finetune:  (data_loader_train,  data_loader_test,   log_writer  )
    '''

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    if args.eval:
        '''
        dataset_test = ...
        '''
        if args.test_label == '':
            dataset_test = build_dataset(is_train='test', args=args)
        else:
            dataset_test = LabeledDataset(args.data_path, args.test_label,
                                          is_train='test', args=args)
        data_loader_train = None
        log_writer = None
    else:  # pretrain or finetune

        if args.pretrain:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0),
                                             interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            dataset_train = UnlabeledDataset(args.data_path, transform=transform_train)
            dataset_test = None

        elif args.finetune:
            '''
            prepare labeled dataset
            '''
            if args.label is not None:
                dataset_train, dataset_test = split_dataset(args.data_path,
                                                            args.label,
                                                            args)
            elif args.train_label == '' or args.test_label == '':
                dataset_train = build_dataset(is_train='train', args=args)
                dataset_test = build_dataset(is_train='test', args=args)
            else:
                dataset_train = LabeledDataset(args.data_path, args.train_label,
                                               is_train='train', args=args)
                dataset_test = LabeledDataset(args.data_path, args.test_label,
                                              is_train='test', args=args)

        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
        # set tfevents folder
        if (global_rank == 0) and (args.log_dir is not None):
            os.makedirs(args.log_dir, exist_ok=True)
            log_writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.task))
        else:
            log_writer = None

    if dataset_test is not None:
        if args.dist_eval:
            if len(dataset_test) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test, num_replicas=num_tasks, rank=global_rank,
                shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            sampler=sampler_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_test = None

    return {
        'data_loader_train': data_loader_train,
        'data_loader_test': data_loader_test,
        'log_writer': log_writer,
    }


def prepare_model(args):
    '''
    return model, model_without_ddp, n_parameters, criterion, optimizer, loss_scaler
    '''
    from Model.util.pos_embed import interpolate_pos_embed
    import timm.optim.optim_factory as optim_factory
    from timm.models.layers import trunc_normal_
    import Model.util.lr_decay as lrd
    from Model import models_vit, models_mae
    from timm.loss import LabelSmoothingCrossEntropy
    device = torch.device(args.device)

    if args.chkpt_path is None and args.finetune:
        model = models_vit.__dict__[args.model](args)
        if args.smoothing > 0.:
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss()
    else:
        # load the checkpoint (args.chkpt_path)
        print("Load checkpoint from: %s" % args.chkpt_path)
        if args.chkpt_path is not None:
            checkpoint = torch.load(args.chkpt_path, map_location='cpu')
            checkpoint_model = checkpoint['model']
        else:
            checkpoint_model = None
        if args.pretrain:
            # pretrain
            from Model import models_mae, models_vit
            model = models_mae.__dict__[args.model](
                img_size=args.input_size,
                norm_pix_loss=args.norm_pix_loss,
            )
            if checkpoint_model is not None:
                interpolate_pos_embed(model, checkpoint_model, 'pos_embed')
                interpolate_pos_embed(model, checkpoint_model, 'decoder_pos_embed')

            ######## edited: distillation in args
            # distillation
            if args.distillation:
                teacher_model = models_vit.__dict__[args.teacher_model](
                    args,
                    img_size=args.input_size,
                    num_classes=args.num_classes
                )
                teacher_model.load_state_dict(torch.load(args.teacher_chkpt, map_location='cpu'))
                criterion = nn.BCEWithLogitsLoss()
                args.teacher_net = teacher_model.to(device)
                args.cls_head = list(teacher_model.children())[-1]
                teacher_model.eval()
            else:
                criterion = None
        else:
            # finetune / evaluate
            from Model import models_vit
            model = models_vit.__dict__[args.model]

            model = model(
                img_size=args.input_size,
                num_classes=args.num_classes,
                drop_path_rate=args.drop_path,
                global_pool=args.global_pool,
            )
            if not args.eval and checkpoint_model is not None:
                state_dict = model.state_dict()
                for k in ['head.weight', 'head.bias']:
                    if (k in checkpoint_model) and checkpoint_model[k].shape != state_dict[k].shape:
                        print(f"Removing key {k} from pretrained checkpoint")
                        del checkpoint_model[k]
                # interpolate position embedding
                interpolate_pos_embed(model, checkpoint_model)
                # initialize fc layer
                trunc_normal_(model.head.weight, std=2e-5)
            if args.multi_label:
                criterion = nn.BCEWithLogitsLoss()
            elif args.smoothing > 0.:
                criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
            else:
                criterion = torch.nn.CrossEntropyLoss()

        if checkpoint_model is not None:
            msg = model.load_state_dict(checkpoint_model, strict=False)
            print(msg)
        # # do not record gradient of attention blocks
        # for blk in model.blocks[:-1]:
        #     blk.requires_grad_(False)

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # set distributed
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # set optimizer
    if args.pretrain:
        param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
        print(optimizer)
    else:
        try:
            # build optimizer with layer-wise lr decay (lrd)
            param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
                                                no_weight_decay_list=model_without_ddp.no_weight_decay(),
                                                layer_decay=args.layer_decay
                                                )
            optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
        except:
            optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr,
                                          weight_decay=args.weight_decay)
        # optimizer = torch.optim.SGD(param_groups, lr=args.lr, momentum=0.9, nesterov=True)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    return model, model_without_ddp, n_parameters, criterion, optimizer, loss_scaler


def fix_seed(args):
    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True