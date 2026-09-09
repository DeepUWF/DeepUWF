
import sys

import torch
import timm

# assert timm.__version__ == "0.3.2"  # version check

from .util import misc as misc
from .util import lr_sched as lr_sched

import math


def train_one_epoch(model, optimizer, loss_scaler, data_dict, device, current_epoch, args=None, criterion=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(current_epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if data_dict["log_writer"] is not None:
        print('log_dir: {}'.format(data_dict["log_writer"].log_dir))
    ###### edited: distillation
    if args.distillation:
        metric_logger.add_meter('distill_loss', misc.SmoothedValue(window_size=1, fmt='{value:.6f}') )
        args.teacher_net.eval()
    ######

    for data_iter_step, data in enumerate(metric_logger.log_every(data_dict["data_loader_train"], print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_dict["data_loader_train"]) + current_epoch, args)

        samples = data.to(device, non_blocking=True)
        ##### edited: outcome
        with torch.cuda.amp.autocast():
            loss, _, _, outcome = model(samples, mask_ratio=args.mask_ratio)
            if args.distillation:
                # expand the outcome of shape from [B,1024] to [B, 2048]
                outcome = torch.cat([outcome, outcome], dim=1)
            
                # print(outcome.shape, args.cls_head)
                args.teacher_net.eval()
                target = args.teacher_net(samples)  # args.teacher_net: defined in prepare.py -> prepare_model
                pred = args.cls_head(outcome)   ##########
                pseudo_label = torch.sigmoid(target)
                distill_loss = criterion(pred, pseudo_label)
                # print(f"distill_loss: {distill_loss}")
                metric_logger.update(distill_loss=distill_loss.item())
                loss += distill_loss*0.3
            ######
        
        # MIM loss backward and model update
        loss_value = loss.item()
        
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        loss /= accum_iter

        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
    
        loss_value_reduce = misc.all_reduce_mean(loss_value)


        if data_dict["log_writer"] is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_dict["data_loader_train"]) + current_epoch) * 1000)
            data_dict["log_writer"].add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            data_dict["log_writer"].add_scalar('lr', lr, epoch_1000x)
                


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}










