import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import sys

sys.path.append('../')

import numpy as np
import sys
import scipy.ndimage as nd
import json
import pickle
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torchvision import transforms
from torchvision.transforms import Resize
from torch.utils.data import Dataset, DataLoader
from models.resnet import *
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import time
import math
from utils.utils import AverageMeter
from datasets.Preprocess_2D import Slice2D_DataPreprocess
from train.train_2d_cls2 import test
from glob import glob
import torch.nn.functional as F

import scipy.ndimage as nd
import json
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

def cal_auc(y_true, y_pred):
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
    auc = metrics.auc(fpr, tpr)

    return auc


def main():
    batch_size = 4
    num_workers = 4
    data_transforms = transforms.Compose([
            transforms.ToTensor(),
        ]) 
    model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(512, 2)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    data_type = 'phase1'
    phase_root = '../data/aug2D/slice_diff_exp2'
    data_root = '/home/zhangwd/code/work/FattyLiver_Solution/data/experiment_slice/train/{}'.format(data_type)
    config_test = '../data/config/config_train.txt'    
    auc_list = []
    weight_addr = []

    for weights in glob(phase_root + '/*.pth'):
        weight_addr.append(weights)
        model.load_state_dict(torch.load(weights))

        test_ds = Slice2D_DataPreprocess(data_root, data_type, config_test, data_transforms)
        test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                            num_workers=num_workers, pin_memory=False)    

        criterion = nn.CrossEntropyLoss().cuda()
        acc, logger, tot_pred, tot_label, tot_prob = test(test_dataloader, nn.DataParallel(model).cuda(), criterion, 0, 10)

        auc = cal_auc(np.array(tot_label, dtype=np.float32), np.array(tot_prob))
        auc_list.append(auc)

    best_auc = max(auc_list)
    best_auc_index = auc_list.index(best_auc)

    print('=====>The best auc is: {}\n'.format(best_auc))
    print('The corresponding weight address is: {}'.format(weight_addr[best_auc_index]))

if __name__ == '__main__':
    main()