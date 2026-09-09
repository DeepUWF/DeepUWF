# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import math
import sys
import csv
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions

from timm.data import Mixup
from timm.utils import accuracy

from typing import Iterable, Optional
from .util import misc as misc
from .util import lr_sched as lr_sched
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score,multilabel_confusion_matrix
# from sklearn.metrics import roc_curve, plot_roc_curve
from pycm import *
from .util import model_update
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy


def classify_threshold(prediction, threshold=0.5):
    '''
    Only for 2-class classification
    '''
    # prediction: [N, 2] with softmax
    # threshold: float \in [0, 1]
    preds = deepcopy(prediction[:, 1])
    preds[preds >= threshold] = 1
    preds[preds < threshold] = 0
    return preds


def misc_measures(confusion_matrix):
    
    acc = []
    sensitivity = []
    specificity = []
    precision = []
    G = []
    F1_score_2 = []
    mcc_ = []
    
    for i in range(1, confusion_matrix.shape[0]):
        cm1=confusion_matrix[i]
        acc.append(1.*(cm1[0,0]+cm1[1,1])/np.sum(cm1))
        sensitivity_ = 1.*cm1[1,1]/(cm1[1,0]+cm1[1,1])
        sensitivity.append(sensitivity_)
        specificity_ = 1.*cm1[0,0]/(cm1[0,1]+cm1[0,0])
        specificity.append(specificity_)
        precision_ = 1.*cm1[1,1]/(cm1[1,1]+cm1[0,1])
        precision.append(precision_)
        G.append(np.sqrt(sensitivity_*specificity_))
        F1_score_2.append(2*precision_*sensitivity_/(precision_+sensitivity_))
        mcc = (cm1[0,0]*cm1[1,1]-cm1[0,1]*cm1[1,0])/np.sqrt((cm1[0,0]+cm1[0,1])*(cm1[0,0]+cm1[1,0])*(cm1[1,1]+cm1[1,0])*(cm1[1,1]+cm1[0,1]))
        mcc_.append(mcc)
        
    acc = np.array(acc).mean()
    sensitivity = np.array(sensitivity).mean()
    specificity = np.array(specificity).mean()
    precision = np.array(precision).mean()
    G = np.array(G).mean()
    F1_score_2 = np.array(F1_score_2).mean()
    mcc_ = np.array(mcc_).mean()
    
    return acc, sensitivity, specificity, precision, G, F1_score_2, mcc_





def train_one_epoch(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                    loss_scaler, data_dict, device, current_epoch, 
                    args=None, max_norm: float = 0, 
                    criterion: torch.nn.Module=nn.CrossEntropyLoss()
                    ):
    
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    
    header = 'Epoch: [{}]'.format(current_epoch)
    print_freq = 1

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if data_dict["log_writer"] is not None:
        print('log_dir: {}'.format(data_dict["log_writer"].log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_dict["data_loader_train"], print_freq, header)):

        if args.apfl:
            lr = lr_sched.adjust_learning_rate(optimizer[0], data_iter_step / len(data_dict["data_loader_train"]) + current_epoch, args)
            lr = lr_sched.adjust_learning_rate(optimizer[1], data_iter_step / len(data_dict["data_loader_train"]) + current_epoch, args)
        # we use a per iteration (instead of per epoch) lr scheduler
        elif data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_dict["data_loader_train"]) + current_epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if args.apfl:
            optimizer[0].zero_grad()
            optimizer[1].zero_grad()
            with torch.cuda.amp.autocast():
                outputs, outputs_personal = model(samples)
                loss = criterion(outputs, targets)
                loss_personal = criterion(outputs_personal, targets)
            loss_scaler(loss, optimizer[0], clip_grad=args.clip_grad, 
                        parameters=model.parameters(), create_graph=False, update_grad=True)
            loss_scaler(loss_personal, optimizer[1], clip_grad=args.clip_grad,
                        parameters=model.parameters(), create_graph=False, update_grad=True)
            model_update.model_update(model.classifier, model.classifier_personal, args.alpha)
            args.alpha = model_update.alpha_update(model.classifier,
                                                   model.classifier_personal,
                                                   alpha=args.alpha, eta=lr)
            loss_value = loss_personal.item()

        else:
            with torch.cuda.amp.autocast():
                outputs = model(samples)
                # print(criterion, outputs.shape, targets.shape)
                # print('outputs:', outputs, 'target:', targets)
                loss = criterion(outputs, targets)
            
            # outputs = model(samples)
            # loss = criterion(outputs, targets)
            #print(samples[7], outputs[7], targets[7])
        
            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)
            
            loss /= accum_iter
            update_grad = ((data_iter_step + 1) % accum_iter == 0)
            loss_scaler(loss, optimizer, 
                        clip_grad=args.clip_grad,
                        parameters=model.parameters(), 
                        create_graph=False,
                        update_grad=update_grad,
                        )
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()

        if args.device == 'cuda':
            torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if data_dict["log_writer"] is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_dict["data_loader_train"]) + current_epoch) * 1000)
            data_dict["log_writer"].add_scalar('loss', loss_value_reduce, epoch_1000x)
            data_dict["log_writer"].add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}




@torch.no_grad()
def evaluate(model, data_dict, device, task, mode, num_class, threshold=0.5, multi_label=False):
    if multi_label:
        # criterion = torch.nn.BCEWithLogitsLoss()
        # model.eval()
        # prediction_list = []
        # true_label_list = []
        # for batch in data_dict["data_loader_test"]:
        #     images = batch[0]
        #     target = batch[-1]
        #     images = images.to(device, non_blocking=True)
        #     target = target.to(device, non_blocking=True)
        #     true_label = target
        #     output = model(images)
        #     prediction_list.extend(output.cpu().detach().numpy())
        #     true_label_list.extend(true_label.cpu().detach().numpy())
        # auc_roc = roc_auc_score(true_label_list, prediction_list,multi_class='ovr',average='macro')
        # # process prediction with threshold
        # prediction_list = classify_threshold(np.array(prediction_list), threshold=threshold)
        # # calculate accuracy mannually
        # acc = 0
        # for i in range(len(prediction_list)):
        #     print(prediction_list[i], true_label_list[i])
        #     if prediction_list[i] == true_label_list[i]:
        #         acc += 1
        # acc = acc / len(prediction_list)
        # print('AUC-roc: {:.4f}'.format(auc_roc), 'Acc: {:.4f}'.format(acc))
        return {'acc1':-1, 'loss':-1}, -1

    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    if (task is not None) and (not os.path.exists(task)):
        os.makedirs(task, exist_ok=True)

    prediction_decode_list = []
    prediction_list = []
    true_label_decode_list = []
    true_label_onehot_list = []
    
    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_dict["data_loader_test"], 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        true_label=F.one_hot(target.to(torch.int64), num_classes=num_class)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)
            prediction_softmax = nn.Softmax(dim=1)(output)
            ##########################################################
            #_,prediction_decode = torch.max(prediction_softmax, 1)
            if multi_label:
                prediction_decode = classify_threshold(prediction_softmax, threshold=threshold)
            else:
                _,prediction_decode = torch.max(prediction_softmax, 1)
            ##########################################################

            _,true_label_decode = torch.max(true_label, 1)

            prediction_decode_list.extend(prediction_decode.cpu().detach().numpy())
            true_label_decode_list.extend(true_label_decode.cpu().detach().numpy())
            true_label_onehot_list.extend(true_label.cpu().detach().numpy())
            prediction_list.extend(prediction_softmax.cpu().detach().numpy())

        acc1,_ = accuracy(output, target, topk=(1,2))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
    # gather the stats from all processes
    true_label_decode_list = np.array(true_label_decode_list)
    prediction_decode_list = np.array(prediction_decode_list)
    confusion_matrix = multilabel_confusion_matrix(true_label_decode_list, prediction_decode_list,labels=[i for i in range(num_class)])
    acc, sensitivity, specificity, precision, G, F1, mcc = misc_measures(confusion_matrix)
    
    auc_roc = roc_auc_score(true_label_onehot_list, prediction_list,multi_class='ovr',average='macro')
    auc_pr = average_precision_score(true_label_onehot_list, prediction_list,average='macro')          
            
    metric_logger.synchronize_between_processes()
    
    print('Sklearn Metrics - Acc: {:.4f} AUC-roc: {:.4f} AUC-pr: {:.4f} F1-score: {:.4f} MCC: {:.4f}, sensitivity: {:.4f} specificity: {:.4f}'.format(acc, auc_roc, auc_pr, F1, mcc, sensitivity, specificity)) 
    results_path = task+'_metrics_{}.csv'.format(mode)
    with open(results_path,mode='a',newline='',encoding='utf8') as cfa:
        wf = csv.writer(cfa)
        data2=[[acc,sensitivity,specificity,precision,auc_roc,auc_pr,F1,mcc,metric_logger.loss]]
        for i in data2:
            wf.writerow(i)
            
    
    if mode in ['test', 'val']:
        cm = ConfusionMatrix(actual_vector=true_label_decode_list, predict_vector=prediction_decode_list)
        cm.plot(cmap=plt.cm.Blues,number_label=True,normalized=True,plot_lib="matplotlib")
        plt.savefig(task+'confusion_matrix_test.jpg',dpi=600,bbox_inches ='tight')
        # # draw and save auroc curve
        # fig, ax = plt.subplots(figsize=(12, 10))
        # roc_curve = plot_roc_curve(model, data_dict["data_loader_test"], ax=ax, linewidth=1, name='ROC curve (area = %0.2f)' % auc_roc)
        # ax.legend(fontsize=12)
        # plt.savefig(task+'auroc_curve_test.jpg',dpi=600,bbox_inches ='tight')
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()},auc_roc



@torch.no_grad()
def save_prob(model, data_dict, device, output_dir, args, id):
    '''
    save the predicted probabilities of all samples as npz files
    '''
    model.eval()
    prediction_list = []
    true_label_list = []
    for batch in data_dict["data_loader_test"]:
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        true_label = target
        output = model.forward_features(images)
        prediction_list.extend(output.cpu().detach().numpy())
        true_label_list.extend(true_label.cpu().detach().numpy())

    prediction_list = np.array(prediction_list)
    true_label_list = np.array(true_label_list)

    prefix = args.chkpt_path.split('/')[-1].split('.')[0] + f'_{id}_'

    np.savez_compressed(os.path.join(output_dir, prefix+'prediction.npz'), prediction_list)
    np.savez_compressed(os.path.join(output_dir, prefix+'target.npz'), true_label_list)
