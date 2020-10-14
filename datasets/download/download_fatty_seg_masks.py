import requests
import pandas as pd
import os
import sys
import csv
import numpy as np
from glob import glob
import json

import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))

from common.utils.download_utils import download_mha_with_csv

import fire

def download_masks(out_path, config_file):
    '''
    debug cmd: download_masks('../data/pulmonaryEmbolism/data_batch_1/masks', '../data/pulmonaryEmbolism/data_batch_1/image_anno_TASK_2706.csv')
    debug cmd: download_masks('../../data/seg_task/masks', '../../data/config_raw/image_anno_TASK_3491.csv')

    invoke cmd: python download_pulmonary_embolism_masks.py download_masks '../data/pulmonaryEmbolism/data_batch_1/masks' '../data/pulmonaryEmbolism/data_batch_1/image_anno_TASK_2706.csv'
    invoke cmd: python download_pulmonary_embolism_masks.py download_masks '../data/pulmonaryEmbolism/data_batch_1/masks' '../data/pulmonaryEmbolism/data_batch_1/image_anno_TASK_2899.csv'
    '''
    download_mha_with_csv(out_path, config_file)


def rename_mask_files(indir, outdir, config_file):
    '''
    debug cmd: rename_mask_files('../../data/seg_task/masks', '../../data/seg_task/renamed_masks', '../../data/config_raw/image_anno_TASK_3491.csv')
    invoke cmd: python download_fatty_seg_masks.py rename_mask_files '../../data/seg_task/masks' '../../data/seg_task/renamed_masks' '../../data/config_raw/image_anno_TASK_3491.csv'
    '''
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(config_file)
    index_dict = {}
    for index, row in df.iterrows():
        series_uid = row['序列编号']
        mask_name = row['影像结果编号']
        if series_uid in index_dict:
            cur_index = index_dict[series_uid]+1
        else:
            cur_index = 0
        index_dict[series_uid] = cur_index
        renamed_mask_name = '{}'.format(series_uid, cur_index)
        src_file = os.path.join(indir, '{}.mha'.format(mask_name))
        dst_file = os.path.join(outdir, '{}.mha'.format(renamed_mask_name))
        shutil.copyfile(src_file, dst_file)
        print('copy from {} to {}'.format(src_file, dst_file))


def copy_input_series(src_root, dst_root, mask_root):
    '''
    debug cmd: copy_input_series('../../data/experiment_0/0.ori', '../../data/seg_task/images', '../../data/seg_task/renamed_masks')
    invoke cmd: python download_fatty_seg_masks.py copy_input_series '../../data/experiment_0/0.ori' '../../data/seg_task/images' '../../data/seg_task/renamed_masks'
    '''
    os.makedirs(dst_root, exist_ok=True)
    uids = glob(os.path.join(mask_root, '*.mha'))
    uids = [os.path.basename(i).replace('.mha', '') for i in uids]
    for uid in uids:
        src_file = os.path.join(src_root, uid)
        if not os.path.isdir(src_file):
            print('{} not exist!'.format(src_file))
            continue
        if not os.path.isdir(os.path.join(src_file, 'echo_1')):
            print('{} echo 1 not exist!'.format(src_file))
            continue
        dst_file = os.path.join(dst_root, uid)
        shutil.copytree(src_file, dst_file)
        print('copy from {} to {}'.format(src_file, dst_file))


if __name__ == "__main__":
    fire.Fire()
    # download_masks('../data/pulmonaryEmbolism/data_batch_1/masks', '../data/pulmonaryEmbolism/data_batch_1/image_anno_TASK_2706.csv')
    # download_masks('../../data/seg_task/masks', '../../data/config_raw/image_anno_TASK_3491.csv')
    # copy_input_series('../../data/experiment_0/0.ori', '../../data/seg_task/images', '../../data/seg_task/renamed_masks')

