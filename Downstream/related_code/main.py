import numpy as np
import os
import random
import time
import datetime
from pathlib import Path
import flwr as fl
import json

import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm
#assert timm.__version__ == "0.3.2"  # version check

from arguments import get_args_parser
from prepare import prepare_dataset, prepare_model, fix_seed
import Model.util.misc as misc
from Model.util.misc import NativeScalerWithGradNormCount as NativeScaler

import warnings
warnings.filterwarnings('ignore')
rand_id = random.randint(0, 1000000)


def prepare_client(args, model_dict, data_dict):
    '''
    data_dict: a dict that contains training-related variables
    '''
    from client import UWFClient
    if args.pretrain:
        from Model.pretrain import train_one_epoch
        evaluate = None
    else:
        from Model.finetune import train_one_epoch, evaluate

    client = UWFClient(
        model_dict=model_dict,
        data_dict=data_dict,
        is_pretrain=args.pretrain,
        train_func=train_one_epoch,
        save_model_func=misc.save_model,
        device=args.device,
        args=args,
        epoch=args.start_epoch,
        eval_func=evaluate,
        local_rounds=[9,8,7,6,5,4,3,2,1,1]
    )
    return client




def main(args):
    if args.server:
        '''
        to be modified
        '''
        from server import launch_server, weighted_average, CustomStrategy
        #launch_server(num_epochs=args.epochs, server_addr=args.server_addr)
        launch_server(
            num_epochs=args.epochs,
            strat=CustomStrategy,
            #strat=fl.server.strategy.FedAvg,
            server_addr=args.server_addr,
        )
        exit(0)

    print("pretrain = {}, finetune = {}, eval = {}".format(args.pretrain, args.finetune, args.eval))
    
    # enable benchmark
    cudnn.benchmark = True
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    if args.client:# and misc.is_main_process():
        from client import UWFClient
        cli = UWFClient(
            is_pretrain=args.pretrain,
            save_model_func=misc.save_model,
            args=args,
            epoch=args.start_epoch,
        )
        fl.client.start_numpy_client(
            #server_address="frp-fit.top:65530",
            server_address=args.server_addr,
            client=cli,
            grpc_max_message_length=0x7fffffff,
        )
        exit(0)
    # if args.client and (not args.eval):
    #     '''
    #     Only 1 process will connect to the federation server even if using DDP
    #     '''
    #     cli = prepare_client(
    #             args=args, 
    #             model_dict={
    #                 "model": model, 
    #                 "model_without_ddp": model_without_ddp, 
    #                 "optimizer": optimizer, 
    #                 "loss_scaler": loss_scaler, 
    #                 "criterion": criterion,
    #                 "n_parameters": n_parameters,
    #             }, 
    #             data_dict=data_dict,
    #         )
    #     fl.client.start_numpy_client(
    #         #server_address="frp-fit.top:65530",
    #         server_address=args.server_addr,
    #         client=cli,
    #         grpc_max_message_length=0x7fffffff,
    #     ) 


    device = torch.device(args.device)
    # print pwd and args
    print('##############################################')
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    print('##############################################')
    # prepare data and model, as well as random seed
    fix_seed(args)
    data_dict = prepare_dataset(args)
    model, model_without_ddp, n_parameters, criterion, optimizer, loss_scaler = prepare_model(args)
    
    
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    
    if args.pretrain:
        from Model.pretrain import train_one_epoch
        evaluate = None
    else:
        from Model.finetune import train_one_epoch, evaluate, save_prob
    
    if args.eval:
        # evaluation only
        test_stats, auc_roc = evaluate(model, data_dict, device, os.path.join(args.log_dir, args.task), 'val', args.num_classes, multi_label=args.multi_label)
        print("Test auc_roc = %.4f" % auc_roc)
        print(test_stats)
        exit(0)


    max_auc = -0x7fffffff
    for epoch in range(args.start_epoch, args.epochs):
        
        if args.distributed:
            data_dict["data_loader_train"].sampler.set_epoch(epoch)
        # train one epoch
        train_stats = train_one_epoch(model, optimizer, loss_scaler, data_dict, device, epoch, args,
                                        criterion=criterion)

        if evaluate is not None:
            val_stats, val_auc_roc = evaluate(model, data_dict, device, os.path.join(args.log_dir, args.task), 'val', args.num_classes, multi_label=args.multi_label)
            if max_auc<val_auc_roc:
                max_auc = val_auc_roc
                # save predicted probabilities of all samples
                # global rand_id
                # save_prob(model, data_dict, device, args.output_dir, args, rand_id)

                if args.output_dir:
                    misc.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch
                        )
            if data_dict["log_writer"] is not None:
                data_dict["log_writer"].add_scalar('perf/val_acc1', val_stats['acc1'], epoch)
                data_dict["log_writer"].add_scalar('perf/val_auc', val_auc_roc, epoch)
                data_dict["log_writer"].add_scalar('perf/val_loss', val_stats['loss'], epoch)

        elif args.output_dir and (epoch % 5 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch
                        )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters,
                }

        if args.output_dir and misc.is_main_process():
            if data_dict["log_writer"] is not None:
                data_dict["log_writer"].flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    print('Training time {}'.format(total_time_str))
    return max_auc




if __name__ == '__main__':
    argparser = get_args_parser()
    args = argparser.parse_args()
    # args.server == True: run as the federated server
    # args.client == True: run as the federated client, 
    #                   and the main process will communicate with the server
    # else: run like a local process
    
    misc.init_distributed_mode(args) 
    main(args)