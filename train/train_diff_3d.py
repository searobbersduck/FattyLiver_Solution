import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

import numpy as np
import sys
import scipy.ndimage as nd
import json
import pickle
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from models.resnet import *
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import time
import math
from utils.utils import AverageMeter
from datasets.FattyLiverDatasets import FattyLiverClsDatasetsDiff3D

def initial_cls_weights(cls):
    for m in cls.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
            m.weight.data.normal_(0, math.sqrt(2./n))
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()
        if isinstance(m, nn.Conv3d):
            n = m.kernel_size[0]*m.kernel_size[1]*m.kernel_size[2]*m.out_channels
            m.weight.data.normal_(0, math.sqrt(2./n))
            if m.bias is not None:
                m.bias.data.zero_()


def train(train_dataloader, model, criterion, optimizer, epoch, display):
    model.train()
    tot_pred = np.array([], dtype=int)
    tot_label = np.array([], dtype=int)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    end = time.time()
    logger = []
    for num_iter, (images, labels,_) in enumerate(train_dataloader):
        data_time.update(time.time()-end)
        output = model(Variable(images.cuda()))
        loss = criterion(output, Variable(labels.cuda()))
        _, pred = torch.max(output, 1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time()-end)
        end = time.time()
        pred = pred.cpu().data.numpy()
        labels = labels.numpy()
        tot_pred = np.append(tot_pred, pred)
        tot_label = np.append(tot_label, labels)
        losses.update(loss.data.cpu().numpy(), len(images))
        accuracy.update(np.equal(pred, labels).sum()/len(labels), len(labels))
        if (num_iter+1) % display == 0:
            correct = np.equal(tot_pred, tot_label).sum()/len(tot_pred)
            print_info = 'Epoch: [{0}][{1}/{2}]\tTime {batch_time.val:3f} ({batch_time.avg:.3f})\t'                'Data {data_time.avg:.3f}\t''Loss {loss.avg:.4f}\tAccuray {accuracy.avg:.4f}'.format(
                epoch, num_iter, len(train_dataloader),batch_time=batch_time, data_time=data_time,
                loss=losses, accuracy=accuracy
            )
            print(print_info)
            logger.append(print_info)
    print(tot_pred)
    print(tot_label)
    return accuracy.avg, logger


def val(train_dataloader, model, criterion, epoch, display):
    model.eval()
    tot_pred = np.array([], dtype=int)
    tot_label = np.array([], dtype=int)
    tot_prob = np.array([])
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    end = time.time()
    logger = []
    for num_iter, (images, labels, _) in enumerate(train_dataloader):
        data_time.update(time.time()-end)
        output = model(Variable(images.cuda()))
        loss = criterion(output, Variable(labels.cuda()))
        _, pred = torch.max(output, 1)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        batch_time.update(time.time()-end)
        end = time.time()
        pred = pred.cpu().data.numpy()
        labels = labels.numpy()
        tot_pred = np.append(tot_pred, pred)
        tot_label = np.append(tot_label, labels)
        losses.update(loss.data.cpu().numpy(), len(images))
        accuracy.update(np.equal(pred, labels).sum()/len(labels), len(labels))
        if (num_iter+1) % display == 0:
            correct = np.equal(tot_pred, tot_label).sum()/len(tot_pred)
            print_info = 'Epoch: [{0}][{1}/{2}]\tTime {batch_time.val:3f} ({batch_time.avg:.3f})\t'                'Data {data_time.avg:.3f}\t''Loss {loss.avg:.4f}\tAccuray {accuracy.avg:.4f}'.format(
                epoch, num_iter, len(train_dataloader),batch_time=batch_time, data_time=data_time,
                loss=losses, accuracy=accuracy
            )
            print(print_info)
            logger.append(print_info)
    print(tot_pred)
    print(tot_label)
    return accuracy.avg, logger




def main():

    batch_size = 4
    num_workers = 4
    phase = 'train'
    epochs = 10000
    display = 2

    config_file = '../config/config_diff_3d.json'
    config = None
    with open(config_file) as f:
        config = json.load(f)
    print('\n')
    print('====> parse options:')
    print(config)
    print('\n')

    data_root = '../data/experiment_0/0.ori'
    config_train = '../data/config/config_train.txt'
    config_val = '../data/config/config_val.txt'
    crop_size = [32, 384, 512]


    print('====> create output model path:\t')
    os.makedirs(config["model_dir"], exist_ok=True)
    time_stamp = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
    model_dir = os.path.join(config["model_dir"], 'ct_pos_recogtion_{}'.format(time_stamp))
    os.makedirs(model_dir, exist_ok=True)


    print('====> building model:\t')
    model = resnet34(num_classes=config["num_classes"], 
                     shortcut_type=True, sample_size_y=crop_size[1], sample_size_x=crop_size[2], sample_duration=crop_size[0])
    initial_cls_weights(model)
    pretrained_weights = config['weight']
    if pretrained_weights is not None:
        model.load_state_dict(torch.load(pretrained_weights))

    criterion = nn.CrossEntropyLoss().cuda()

    if phase == 'train':
        train_ds = FattyLiverClsDatasetsDiff3D(data_root, config_train, crop_size)
        val_ds = FattyLiverClsDatasetsDiff3D(data_root, config_val, crop_size)
        train_dataloader = DataLoader(train_ds, batch_size=batch_size, 
                                     shuffle=True, num_workers=num_workers, 
                                     pin_memory=True)
        val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                                   num_workers=num_workers, pin_memory=False)

        best_acc = 0.4

        for epoch in range(epochs):
            if epoch < config['fix']:
                lr = config['lr']
            else:
                lr = config['lr'] * (0.1 ** (epoch//config['step']))
            mom = config['mom']
            wd = config['wd']
            optimizer = None
            if config['optimizer'] == 'sgd':
                optimizer = optim.SGD([{'params': model.parameters()}], 
                                      lr=lr, momentum=mom, weight_decay=wd, nesterov=True)
            elif config['optimizer'] == 'adam':
                optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=lr, betas=(0.9, 0.999))

            _, _ = train(train_dataloader, nn.DataParallel(model).cuda(), criterion, optimizer, epoch, display)
            acc, logger = val(val_dataloader, nn.DataParallel(model).cuda(), criterion, epoch, display)
            print('val acc:\t{:.3f}'.format(acc))
            if acc > best_acc:
                print('\ncurrent best accuracy is: {}\n'.format(acc))
                best_acc = acc
                saved_model_name = os.path.join(model_dir, 'ct_pos_recognition_{:04d}_best.pth'.format(epoch))
                torch.save(model.cpu().state_dict(), saved_model_name)
                print('====> save model:\t{}'.format(saved_model_name))





if __name__ == '__main__':
    main()