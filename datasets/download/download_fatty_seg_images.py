import os
import sys
import pandas as pd
import numpy as np

import fire

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))

from common.utils.download_utils import download_dcms_with_website

root_dir = '../data/pulmonaryEmbolism/data_batch_1'

def get_series_uids(infile, column_name, outfile):
    '''
    get_series_uids('../data/pulmonaryEmbolism/data_batch_1/image_anno_TASK_2706.csv', '序列编号', '../data/pulmonaryEmbolism/data_batch_1/image_anno_TASK_2706.txt')
    get_series_uids('../data/pulmonaryEmbolism/data_batch_1/image_anno_TASK_2899.csv', '序列编号', '../data/pulmonaryEmbolism/data_batch_1/image_anno_TASK_2899.txt')

    invoke cmd: python download_pulmonary_embolism_images.py get_series_uids '../data/pulmonaryEmbolism/data_batch_1/image_anno_TASK_2706.csv' '序列编号' '../data/pulmonaryEmbolism/data_batch_1/image_anno_TASK_2706.txt'
    invoke cmd: python download_pulmonary_embolism_images.py get_series_uids '../data/pulmonaryEmbolism/data_batch_1/image_anno_TASK_2899.csv' '序列编号' '../data/pulmonaryEmbolism/data_batch_1/image_anno_TASK_2899.txt'
    '''
    df = pd.read_csv(infile)
    series_uids = list(set(df[column_name].tolist()))
    with open(outfile, 'w') as f:
        f.write('\n'.join(series_uids))
    return series_uids


def download_images(out_path, config_file):
    '''
    invoke cmd: python download_pulmonary_embolism_images.py download_images '../data/pulmonaryEmbolism/data_batch_1/images' '../data/pulmonaryEmbolism/data_batch_1/文件内网地址信息-导出结果.xlsx'
    '''
    download_dcms_with_website(out_path, config_file)



if __name__ == '__main__':
    fire.Fire()
    # get_series_uids('../data/pulmonaryEmbolism/data_batch_1/image_anno_TASK_2706.csv', '序列编号', '../data/pulmonaryEmbolism/data_batch_1/image_anno_TASK_2706.txt')
    # get_series_uids('../data/pulmonaryEmbolism/data_batch_1/image_anno_TASK_2899.csv', '序列编号', '../data/pulmonaryEmbolism/data_batch_1/image_anno_TASK_2899.txt')