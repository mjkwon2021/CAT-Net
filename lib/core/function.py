# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------
"""
Modified by Myung-Joon Kwon
mjkwon2021@gmail.com
July 14, 2020
"""

import logging
import os
import time

import numpy as np
from tqdm import tqdm
import os

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F

from lib.utils.utils import AverageMeter
from lib.utils.utils import get_confusion_matrix
from lib.utils.utils import adjust_learning_rate
from lib.utils.utils import get_world_size, get_rank



def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that 
    process with rank 0 has the averaged results.
    """
    world_size = get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        dist.reduce(reduced_inp, dst=0)
    return reduced_inp

def train(config, epoch, num_epoch, epoch_iters, base_lr, num_iters,
         trainloader, optimizer, model, writer_dict, final_output_dir):
    
    # Training
    model.train()
    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    world_size = get_world_size()

    for i_iter, (images, labels, qtable) in enumerate(trainloader):
        # images, labels, _, _ = batch
        images = images.cuda()
        labels = labels.long().cuda()

        losses, _ = model(images, labels, qtable)  # _ : output of the model (see utils.py)
        loss = losses.mean()

        reduced_loss = reduce_tensor(loss)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(reduced_loss.item())

        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_iters,
                                  i_iter+cur_iters)

        if i_iter % config.PRINT_FREQ == 0:
            print_loss = ave_loss.average() / world_size
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {:.6f}, Loss: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters, 
                      batch_time.average(), lr, print_loss)
            logging.info(msg)
            
            writer.add_scalar('train_loss', print_loss, global_steps)
            global_steps += 1
            writer_dict['train_global_steps'] = global_steps


def validate(config, testloader, model, writer_dict, valid_set="valid"):
    
    rank = get_rank()
    world_size = get_world_size()
    model.eval()
    ave_loss = AverageMeter()
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    avg_mIoU = AverageMeter()
    avg_p_mIoU = AverageMeter()

    with torch.no_grad():
        for _, (image, label, qtable) in enumerate(tqdm(testloader)):
            size = label.size()
            image = image.cuda()
            label = label.long().cuda()

            losses, pred = model(image, label, qtable)
            pred = F.upsample(input=pred, size=(
                        size[-2], size[-1]), mode='bilinear')
            loss = losses.mean()
            reduced_loss = reduce_tensor(loss)
            ave_loss.update(reduced_loss.item())

            current_confusion_matrix = get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)
            confusion_matrix += current_confusion_matrix
            # mIoU
            pos = current_confusion_matrix.sum(1)  # ground truth label count
            res = current_confusion_matrix.sum(0)  # prediction count
            tp = np.diag(current_confusion_matrix)  # Intersection part
            IoU_array = (tp / np.maximum(1.0, pos + res - tp))  # Union part
            mean_IoU = IoU_array.mean()
            avg_mIoU.update(mean_IoU)
            TN = current_confusion_matrix[0, 0]
            FN = current_confusion_matrix[1, 0]
            FP = current_confusion_matrix[0, 1]
            TP = current_confusion_matrix[1, 1]
            p_mIoU = 0.5 * (FN / np.maximum(1.0, FN + TP + TN)) + 0.5 * (FP / np.maximum(1.0, FP + TP + TN))
            avg_p_mIoU.update(np.maximum(mean_IoU, p_mIoU))

    confusion_matrix = torch.from_numpy(confusion_matrix).cuda()
    reduced_confusion_matrix = reduce_tensor(confusion_matrix)

    confusion_matrix = reduced_confusion_matrix.cpu().numpy()
    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum() / pos.sum()
    mean_acc = (tp / np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()
    print_loss = ave_loss.average()/world_size

    if rank == 0:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar(valid_set+'_loss', print_loss, global_steps)
        writer.add_scalar(valid_set+'_mIoU', mean_IoU, global_steps)
        writer.add_scalar(valid_set+'_avg_mIoU', avg_mIoU.average(), global_steps)
        writer.add_scalar(valid_set+'_avg_p-mIoU', avg_p_mIoU.average(), global_steps)
        writer.add_scalar(valid_set+'_pixel_acc', pixel_acc, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1
    return print_loss, mean_IoU, avg_mIoU.average(), avg_p_mIoU.average(), IoU_array, pixel_acc, mean_acc, confusion_matrix

