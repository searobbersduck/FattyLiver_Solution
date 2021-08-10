import os
import sys
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import SimpleITK as sitk
from torch.utils.data import Dataset, DataLoader
import nibabel
from torchvision import transforms
from torchvision.transforms import Resize
from scipy import ndimage
import time
import torch
from PIL import Image
import torch.nn as nn
import fire
import time
import pydicom
import shutil

class Slice2D_DataPreprocess(Dataset):
#遍历读入已经存好的npy数据
    def __init__(self, npy_root, data_type, config_file, transforms):
        self.config_file = config_file  #train.txt
        self.data_type = data_type  #diff/phase1/phase2
        self.image_files = []   #npy图像所在的文件夹们
        self.UID_list = []  #UID_list
        self.labels = []    #标签
        self.npyimage = []  #npy数据
        #self.data_path = npy_root
        self.label_map = {}
        self.img_lists = []
        self.transforms = transforms

        #train npy_root:
        #../data/experiment_slice/train
        with open(self.config_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line is None or len(line) == 0:
                    continue
                ss = line.split('\t')   #ss[0]:UID_list ss[1]:label

                image_file = os.path.join(npy_root, ss[0])  #每位病例的npy文件夹名称，其间包括多个npy文件
                # mask_file = os.path.join(mask_root,ss[0])+'.mha'
                # npyaddr = os.path.join(npy_root,ss[0])  #npy文件夹
                if os.path.isdir(image_file):                    
                    self.image_files.append(image_file) #图像所在的文件夹
                    for img_path in glob(image_file + '/*.npy'):
                        self.img_lists.append(img_path)
                    self.label_map[str(ss[0])] = int(ss[1])
                    # self.UID_list.append(str(ss[0]))
                    # self.labels.append(int(ss[1]))
                    
                
            print('====> fatty liver count is:{}'.format(len(self.img_lists)))

    def __getitem__(self, index):

        image_path = self.img_lists[index]
        uid = image_path.split('/')[-2]
        label = self.label_map[uid]
        image = np.load(image_path)     

        #全部统一成y:x = 384x512
        image_bg = np.zeros([384,512], dtype=np.float32)
        boundary_1 = min(image.shape[0], image_bg.shape[0])
        boundary_2 = min(image.shape[1], image_bg.shape[1])
        image_bg[:boundary_1,:boundary_2] = image[:boundary_1,:boundary_2]

        image_bg = Image.fromarray(image_bg)
        if self.transforms is not None:
            image_tensor = self.transforms(image_bg)
        #liver_tensor = torch.from_numpy(image).float()
        # liver_tensor = liver_tensor.unsqueeze(0)

        return image_tensor, label, image_path

    def __len__(self):
        return len(self.img_lists)


def test2Dslice():
    mode = 'train'
    data_type = 'phase1'
    npy_root = '/home/zhangwd/code/work/FattyLiver_Solution/data/experiment_slice/{}/{}'.format(mode,data_type)
    config_file = '/home/zhangwd/code/work/FattyLiver_Solution/data/config/config_train.txt'
    data_transforms = transforms.Compose([
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]) 
    test_layer = Slice2D_DataPreprocess(npy_root, data_type, config_file, data_transforms)
    loader = torch.utils.data.DataLoader(test_layer, batch_size=1, shuffle=True, num_workers=2)
    for data in loader:
        _, path, label = data
        print(path,label)

    print('\n======> Slice2D_DataPreprocess finished')

if __name__ == '__main__':
    
    test2Dslice()