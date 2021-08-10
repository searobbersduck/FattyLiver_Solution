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
from torchvision import transforms
from torchvision.transforms import Resize
from torch.utils.data import Dataset, DataLoader
# from models.resnet import *
from models.resnet_bn import *
import torch.optim as optim
from torchvision import models
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import time
import math
from utils.utils import AverageMeter
from datasets.Preprocess_2D import Slice2D_DataPreprocess

import torch.nn.functional as F

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
        labels[labels<2] = 0
        labels[labels>=2] = 1
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
    tot_prob = np.array([], dtype=np.float32)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    end = time.time()
    logger = []

    for num_iter, (images, labels, image_path) in enumerate(train_dataloader):
        labels[labels<2] = 0
        labels[labels>=2] = 1
        data_time.update(time.time()-end)
        output = model(Variable(images.cuda()))
        loss = criterion(output, Variable(labels.cuda()))
        _, pred = torch.max(output, 1)

        batch_time.update(time.time()-end)
        end = time.time()
        pred = pred.cpu().data.numpy()
        labels = labels.numpy()
        tot_pred = np.append(tot_pred, pred)
        tot_label = np.append(tot_label, labels)
        tot_prob = np.append(tot_prob, F.softmax(output).cpu().detach().numpy()[:,1])
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
    
    # for i in range(len(images)):
    #     print(tot_pred[i],' ', tot_label[i],' ', image_path[i].split('/')[-1])
    print(tot_pred)
    # print(image_path)
    print(tot_label)
    return accuracy.avg, logger, tot_pred, tot_label, tot_prob

def test(train_dataloader, model, criterion, epoch, display):
    return val(train_dataloader, model, criterion, epoch, display)

def main(argv): 
    #相关数据和地址设置
    mode = argv[1]      #train/test
    data_type = argv[2] #diff/phase1/phase2

    batch_size = 4
    num_workers = 4
    phase = 'train'
    epochs = 500
    display = 10
    transforms_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomCrop(size = (320,448)),
        # transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    transforms_val = transforms.Compose([
        transforms.ToTensor()
    ]) 

    data_root = '/home/zhangwd/code/work/FattyLiver_Solution/data/experiment_slice/{}/{}'.format(mode, data_type)
    val_root = '/home/zhangwd/code/work/FattyLiver_Solution/data/experiment_slice/val/{}'.format(data_type)
    config_train = '../data/config/config_train.txt'
    config_val = '../data/config/config_val.txt'

    #读入模型训练的config参数
    config_file = '/home/zhangwd/code/work/FattyLiver_Solution/config/config_2D.json'    
    config = None
    with open(config_file) as f:
        config = json.load(f)
    print('\n')
    print('====> config parse options:')
    print(config)
    print('\n')

    #建立模型存储路径
    print('====> create output model path:\t')
    model_dir = '../data/aug2D/slice_crop_{}_exp2'.format(data_type)
    os.makedirs(model_dir, exist_ok=True)
    # time_stamp = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
    # model_dir = os.path.join(model_dir,'')
    # os.makedirs(model_dir, exist_ok=True)
    
    #开始导入模型并初始化权重
    print('====> Start to build model:\t')
    model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(512, 2)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    initial_cls_weights(model)

    criterion = nn.CrossEntropyLoss().cuda()

    if phase == 'train':
        train_ds = Slice2D_DataPreprocess(data_root, data_type, config_train, transforms_train)
        val_ds = Slice2D_DataPreprocess(val_root, data_type, config_val, transforms_val)
        train_dataloader = DataLoader(train_ds, batch_size=batch_size, 
                                     shuffle=True, num_workers=num_workers, 
                                     pin_memory=True)
        val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                                   num_workers=num_workers, pin_memory=False)

        best_acc = 0.58

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
            acc, logger,tot_pred, tot_label, tot_prob = val(val_dataloader, nn.DataParallel(model).cuda(), criterion, epoch, display)
            print('val acc:\t{:.3f}'.format(acc))

            if acc > best_acc:
                print('\ncurrent best accuracy is: {}\n'.format(acc))
                # best_acc = acc
                # saved_model_name = os.path.join()#待补充
                saved_model_name = os.path.join(model_dir, '{}_{}_{}_Fattyliver.pth'.format(data_type,acc,epoch))
                torch.save(model.cpu().state_dict(), saved_model_name)
                print('====> save model:\t{}'.format(saved_model_name))

    pass

if __name__ == '__main__':
    main(sys.argv)