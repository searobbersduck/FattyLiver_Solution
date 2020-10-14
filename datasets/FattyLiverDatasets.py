import os
import sys

from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd

import SimpleITK as sitk

from torch.utils.data import Dataset, DataLoader
import nibabel
from scipy import ndimage
import time
import torch
import torch.nn as nn

import fire
import time

import pydicom

import shutil

def read_config_file(config_file):
    '''
    config_file: '../data/config/肝穿病人给放射科和合作者.xlsx'
    表头：['编号', '住院号', '姓名', 'series uid', 'Unnamed: 4', '性别1男2女', '年龄', 'MRS脂肪峰面积', '水峰面积', '脂肪含量', 'Fat', 'necrosisfoci', 'ballooning', 'NAS（total）', 'fibrosis', 'NAS大于4', '进展性纤维化', '脂肪肝病理评分']
    此处用到 'series uid', 'fibrosis'
    debug cmd: read_config_file('../data/config/肝穿病人给放射科和合作者.xlsx')
    '''
    df = pd.read_excel(config_file)
    series_fib_dict = {}
    for index, row in df.iterrows():
        series_fib_dict[row['series uid']] = int(row['fibrosis'])
    return series_fib_dict


def read_dcm_file(in_dcm_path):
    series_reader = sitk.ImageSeriesReader()
    dicomfilenames = series_reader.GetGDCMSeriesFileNames(in_dcm_path)
    series_reader.SetFileNames(dicomfilenames)

    series_reader.MetaDataDictionaryArrayUpdateOn()
    series_reader.LoadPrivateTagsOn()

    image = series_reader.Execute()
    return image


def split_data_to_two_phase_one_case(series_path, out_dir):
    '''
    debug cmd: split_data_to_two_phase_one_case('../data/images_mr_filtered/1.3.12.2.1107.5.2.30.25245.2015120320185731080640838.0.0.0', '')
    '''
    in_files1 = glob(os.path.join(series_path, '*.dcm'))
    in_files2 = glob(os.path.join(series_path, '*.DCM'))
    in_files = in_files1 + in_files2
    echo_1_files = []
    echo_2_files = []
    for infile in in_files:
        metadata = pydicom.dcmread(infile)
        if 1 == metadata.EchoNumbers:
            echo_1_files.append(infile)
        elif 2 == metadata.EchoNumbers:
            echo_2_files.append(infile)
    series_uid = os.path.basename(series_path)
    out_series_path = os.path.join(out_dir, series_uid)
    out_echo_1_path = os.path.join(out_series_path, 'echo_1')
    out_echo_2_path = os.path.join(out_series_path, 'echo_2')
    os.makedirs(out_series_path, exist_ok=True)
    os.makedirs(out_echo_1_path, exist_ok=True)
    os.makedirs(out_echo_2_path, exist_ok=True)

    assert len(echo_1_files) == len(echo_2_files)

    for src_file in echo_1_files:
        dst_file = os.path.join(out_echo_1_path, os.path.basename(src_file))
        shutil.copyfile(src_file, dst_file)
        print('====> copy from {} to {}'.format(src_file, dst_file))

    for src_file in echo_2_files:
        dst_file = os.path.join(out_echo_2_path, os.path.basename(src_file))
        shutil.copyfile(src_file, dst_file)
        print('====> copy from {} to {}'.format(src_file, dst_file))


def split_data_to_two_phase_singletask(in_dir, out_dir, config_file):
    '''
    indir: ../data/images_mr_filtered
    outdir: ../data/experiment_0/0.ori
    config_file: '../data/config/肝穿病人给放射科和合作者.xlsx'     根据配置文件确定需要进入后续操作的series，这里为防止文件夹中混入非序列的子文件夹

    debug cmd: split_data_to_two_phase_singletask('../data/images_mr_filtered', '../data/experiment_0/0.ori', '../data/config/肝穿病人给放射科和合作者.xlsx')
    invoke cmd: python FattyLiverDatasets.py split_data_to_two_phase_singletask '../data/images_mr_filtered' '../data/experiment_0/0.ori'  '../data/config/肝穿病人给放射科和合作者.xlsx'
    '''
    series_fib_dict = read_config_file(config_file)
    series_uids = os.listdir(in_dir)
    series_paths = []
    for series_uid in series_uids:
        if not series_uid in series_fib_dict:
            continue
        series_path = os.path.join(in_dir, series_uid)
        series_paths.append(series_path)
        split_data_to_two_phase_one_case(series_path, out_dir)
    

def resample_sitkImage_by_spacing(sitkImage, newSpacing, vol_default_value='min', interpolator=sitk.sitkNearestNeighbor):
    """
    :param sitkImage:
    :param newSpacing:
    :return:
    """
    if sitkImage == None:
        return None
    if newSpacing is None:
        return None


    dim = sitkImage.GetDimension()
    if len(newSpacing) != dim:
        return None

    # determine the default value
    vol_value = 0.0
    if vol_default_value == 'min':
        vol_value = float(np.ndarray.min(sitk.GetArrayFromImage(sitkImage)))
    elif vol_default_value == 'zero':
        vol_value = 0.0
    elif str(vol_default_value).isnumeric():
        vol_value = float(vol_default_value)

    # calculate new size
    np_oldSize = np.array(sitkImage.GetSize())
    np_oldSpacing = np.array(sitkImage.GetSpacing())

    np_newSpacing = np.array(newSpacing)
    np_newSize = np.divide(np.multiply(np_oldSize, np_oldSpacing), np_newSpacing)
    newSize = tuple(np_newSize.astype(np.uint).tolist())

    # resample sitkImage into new specs
    transform = sitk.Transform()

    return sitk.Resample(sitkImage, newSize, transform, interpolator, sitkImage.GetOrigin(),
                         newSpacing, sitkImage.GetDirection(), vol_value, sitkImage.GetPixelID())


def resample_data_one_case(series_path, out_dir, z_mul:int):
    '''
    series_path: ../data/experiment_0/0.ori/1.3.12.2.1107.5.2.30.25245.2015120320185731080640838.0.0.0/11

    resample_data_one_case('../data/experiment_0/0.ori/1.3.12.2.1107.5.2.30.25245.2015120320185731080640838.0.0.0/echo_1', '../data/experiment_0/0.ori/1.3.12.2.1107.5.2.30.25245.2015120320185731080640838.0.0.0', 1)
    '''
    beg = time.time()
    print('====> processing {}'.format(series_path))
    image = read_dcm_file(series_path)
    basename = os.path.basename(series_path)
    # 1. 保存原始分辨率数据的nii.gz
    out_raw_file = os.path.join(out_dir, '{}.nii.gz'.format(basename))
    
    # sitk.WriteImage(image, out_raw_file)
    # 2. resample, base x-spacing
    # spc = image.GetSpacing()
    # mults = [1,2,4,8]
    # for z_mul in mults:
    #     out_resampled_file = os.path.join(out_dir, '{}_z_mul{}.nii.gz'.format(basename, z_mul))
    #     new_spc = [spc[0]] + [spc[0]] + [spc[0]*z_mul] 
    #     resampled_img = resample_sitkImage_by_spacing(image, new_spc, interpolator=sitk.sitkLinear)
    #     sitk.WriteImage(resampled_img, out_resampled_file)
    end = time.time()
    print('=====> finish {}, time elapsed is {:.3f}s'.format(series_path, end-beg))
    return out_raw_file


def resample_data_singletask(series_paths):
    '''
    indir: ../data/experiment_0/0.ori
    debug cmd: resample_data_singletask('../data/experiment_0/0.ori')
    invoke cmd: python FattyLiverDatasets.py resample_data_singletask '../data/experiment_0/0.ori'
    '''
    print(series_paths)
    for series_path in tqdm(series_paths):
        if not os.path.isdir(series_path):
            continue
        echo_1_path = os.path.join(series_path, 'echo_1')
        echo_2_path = os.path.join(series_path, 'echo_2')
        out_dir = series_path
        if not os.path.isdir(echo_1_path):
            print('{} echo 1 data not exist!'.format(series_path))
            continue
        if not os.path.isdir(echo_2_path):
            print('{} echo 2 data not exist!'.format(series_path))
            continue
        out_echo_1_file = resample_data_one_case(echo_1_path, out_dir, 1)
        out_echo_2_file = resample_data_one_case(echo_2_path, out_dir, 1)

        echo_1_image = sitk.ReadImage(out_echo_1_file)
        echo_2_image = sitk.ReadImage(out_echo_2_file)

        echo_1_arr = sitk.GetArrayFromImage(echo_1_image)
        echo_2_arr = sitk.GetArrayFromImage(echo_2_image)

        echo_1_arr = np.array(echo_1_arr, dtype=np.int16)
        echo_2_arr = np.array(echo_2_arr, dtype=np.int16)

        diff_1_2_arr = echo_1_arr - echo_2_arr
        diff_1_2_image = sitk.GetImageFromArray(diff_1_2_arr)
        diff_1_2_image.CopyInformation(echo_1_image)

        out_diff_file = os.path.join(os.path.dirname(out_echo_1_file), 'diff_1_2.nii.gz')
        sitk.WriteImage(diff_1_2_image, out_diff_file)




def resample_data_multiprocessing(indir, process_num=12):
    '''
    indir: ../data/experiment_0/0.ori
    invoke cmd: python FattyLiverDatasets.py resample_data_multiprocessing '../data/experiment_0/0.ori' 12
    '''
    series_uids = os.listdir(indir)
    series_paths = [os.path.join(indir, i) for i in series_uids]

    import multiprocessing
    from multiprocessing import Process
    multiprocessing.freeze_support()

    pool = multiprocessing.Pool()
    results = []

    num_per_process = (len(series_paths) + process_num - 1)//process_num

    resample_data_singletask(series_paths)

    # for i in range(process_num):
    #     sub_infiles = series_paths[num_per_process*i:min(num_per_process*(i+1), len(series_paths))]
    #     print(sub_infiles)
    #     result = pool.apply_async(resample_data_singletask, args=(sub_infiles))
    #     results.append(result)

    # pool.close()
    # pool.join()



def split_data_to_train_val_test(data_root, config_file, outdir, train_ratio, val_ratio):
    '''
    debug cmd: split_data_to_train_val_test('../data/images_mr_filtered', '../data/config/肝穿病人给放射科和合作者.xlsx', '../data/config', 0.7, 0.1)
    invoke cmd: python FattyLiverDatasets.py split_data_to_train_val_test '../data/images_mr_filtered' '../data/config/肝穿病人给放射科和合作者.xlsx' '../data/config' 0.7 0.1
    
    debug cmd: split_data_to_train_val_test('../data/images_mr_filtered', '../data/config/肝穿病人给放射科和合作者宋筛重复序列终版.xlsx', '../data/config', 0.7, 0.1)
    invoke cmd: python FattyLiverDatasets.py split_data_to_train_val_test '../data/images_mr_filtered' '../data/config/肝穿病人给放射科和合作者宋筛重复序列终版.xlsx' '../data/config' 0.7 0.1
    '''
    series_fib_dict = read_config_file(config_file)
    series_uids = os.listdir(data_root)
    pairs = []
    for series_uid in series_uids:
        if not series_uid in series_fib_dict:
            continue
        pairs.append([series_uid, series_fib_dict[series_uid]])
    np.random.shuffle(pairs)
    train_pos = int(len(pairs)*train_ratio)
    val_pos = int(len(pairs)*(train_ratio+val_ratio))
    train_pairs = pairs[:train_pos]
    val_pairs = pairs[train_pos:val_pos]
    test_pairs = pairs[val_pos:]
    
    out_config_train_file = os.path.join(outdir, 'config_train.txt')
    out_config_val_file = os.path.join(outdir, 'config_val.txt')
    out_config_test_file = os.path.join(outdir, 'config_test.txt')

    with open(out_config_train_file, 'w') as f:
        for pair in train_pairs:
            f.write('{}\t{}\n'.format(pair[0], pair[1]))

    with open(out_config_val_file, 'w') as f:
        for pair in val_pairs:
            f.write('{}\t{}\n'.format(pair[0], pair[1]))

    with open(out_config_test_file, 'w') as f:
        for pair in test_pairs:
            f.write('{}\t{}\n'.format(pair[0], pair[1]))


def split_data_to_train_val_test_ratio(data_root, config_file, outdir, train_ratio, val_ratio):
    '''
    debug cmd: split_data_to_train_val_test('../data/images_mr_filtered', '../data/config/肝穿病人给放射科和合作者.xlsx', '../data/config_ratio', 0.7, 0.1)
    invoke cmd: python FattyLiverDatasets.py split_data_to_train_val_test '../data/images_mr_filtered' '../data/config/肝穿病人给放射科和合作者.xlsx' '../data/config_ratio' 0.7 0.1
    
    debug cmd: split_data_to_train_val_test('../data/images_mr_filtered', '../data/config/肝穿病人给放射科和合作者宋筛重复序列终版.xlsx', '../data/config_ratio', 0.7, 0.1)
    invoke cmd: python FattyLiverDatasets.py split_data_to_train_val_test '../data/images_mr_filtered' '../data/config/肝穿病人给放射科和合作者宋筛重复序列终版.xlsx' '../data/config_ratio' 0.7 0.1


    ====> train pairs label 0 count is 13
    ====> train pairs label 1 count is 39
    ====> train pairs label 2 count is 41
    ====> train pairs label 3 count is 19
    ====> train pairs label 4 count is 4
    ====> val pairs label 0 count is 2
    ====> val pairs label 1 count is 5
    ====> val pairs label 2 count is 6
    ====> val pairs label 3 count is 3
    ====> val pairs label 4 count is 1
    ====> test pairs label 0 count is 4
    ====> test pairs label 1 count is 12
    ====> test pairs label 2 count is 12
    ====> test pairs label 3 count is 6
    ====> test pairs label 4 count is 2

    '''
    series_fib_dict = read_config_file(config_file)
    series_uids = os.listdir(data_root)
    
    pairs_0 = []
    pairs_1 = []
    pairs_2 = []
    pairs_3 = []
    pairs_4 = []

    for series_uid in series_uids:
        if not series_uid in series_fib_dict:
            continue
        if series_fib_dict[series_uid] == 0:
            pairs_0.append([series_uid, 0])
        elif series_fib_dict[series_uid] == 1:
            pairs_1.append([series_uid, 1])
        elif series_fib_dict[series_uid] == 2:
            pairs_2.append([series_uid, 2])
        elif series_fib_dict[series_uid] == 3:
            pairs_3.append([series_uid, 3])
        elif series_fib_dict[series_uid] == 4:
            pairs_4.append([series_uid, 4])

    def inner_split(pairs, train_ratio, val_ratio):
        np.random.shuffle(pairs)
        train_pos = int(len(pairs)*train_ratio)
        val_pos = int(len(pairs)*(train_ratio+val_ratio))
        train_pairs = pairs[:train_pos]
        val_pairs = pairs[train_pos:val_pos]
        test_pairs = pairs[val_pos:]
        return train_pairs, val_pairs, test_pairs

    train_pairs_0, val_pairs_0, test_pairs_0 = inner_split(pairs_0, train_ratio, val_ratio)
    train_pairs_1, val_pairs_1, test_pairs_1 = inner_split(pairs_1, train_ratio, val_ratio)
    train_pairs_2, val_pairs_2, test_pairs_2 = inner_split(pairs_2, train_ratio, val_ratio)
    train_pairs_3, val_pairs_3, test_pairs_3 = inner_split(pairs_3, train_ratio, val_ratio)
    train_pairs_4, val_pairs_4, test_pairs_4 = inner_split(pairs_4, train_ratio, val_ratio)

    print('====> train pairs label 0 count is {}'.format(len(train_pairs_0)))
    print('====> train pairs label 1 count is {}'.format(len(train_pairs_1)))
    print('====> train pairs label 2 count is {}'.format(len(train_pairs_2)))
    print('====> train pairs label 3 count is {}'.format(len(train_pairs_3)))
    print('====> train pairs label 4 count is {}'.format(len(train_pairs_4)))
    print('====> val pairs label 0 count is {}'.format(len(val_pairs_0)))
    print('====> val pairs label 1 count is {}'.format(len(val_pairs_1)))
    print('====> val pairs label 2 count is {}'.format(len(val_pairs_2)))
    print('====> val pairs label 3 count is {}'.format(len(val_pairs_3)))
    print('====> val pairs label 4 count is {}'.format(len(val_pairs_4)))
    print('====> test pairs label 0 count is {}'.format(len(test_pairs_0)))
    print('====> test pairs label 1 count is {}'.format(len(test_pairs_1)))
    print('====> test pairs label 2 count is {}'.format(len(test_pairs_2)))
    print('====> test pairs label 3 count is {}'.format(len(test_pairs_3)))
    print('====> test pairs label 4 count is {}'.format(len(test_pairs_4)))

    train_pairs = train_pairs_0 + train_pairs_1 + train_pairs_2 + train_pairs_3 + train_pairs_4
    val_pairs = val_pairs_0 + val_pairs_1 + val_pairs_2 + val_pairs_3 + val_pairs_4
    test_pairs = test_pairs_0 + test_pairs_1 + test_pairs_2 + test_pairs_3 + test_pairs_4

    np.random.shuffle(train_pairs)
    np.random.shuffle(val_pairs)
    np.random.shuffle(test_pairs)

    out_config_train_file = os.path.join(outdir, 'config_train.txt')
    out_config_val_file = os.path.join(outdir, 'config_val.txt')
    out_config_test_file = os.path.join(outdir, 'config_test.txt')

    os.makedirs(outdir, exist_ok=True)
    with open(out_config_train_file, 'w') as f:
        for pair in train_pairs:
            f.write('{}\t{}\n'.format(pair[0], pair[1]))

    with open(out_config_val_file, 'w') as f:
        for pair in val_pairs:
            f.write('{}\t{}\n'.format(pair[0], pair[1]))

    with open(out_config_test_file, 'w') as f:
        for pair in test_pairs:
            f.write('{}\t{}\n'.format(pair[0], pair[1]))






class FattyLiverClsDatasets(Dataset):
    def __init__(self, data_root, config_file, crop_size, scale_size, phase='train'):
        self.image_files = []
        self.labels = []
        with open(config_file) as f:
            for line in f.readlines():
                line = line.strip()
                if line is None or len(line) == 0:
                    continue
                ss = line.split('\t')
                self.image_files.append(os.path.join(data_root, ss[0]))
                self.labels.append(ss[1])
            print('====> fatty liver count is:{}'.format(len(self.image_files)))
        
    def __getitem__(self, index):
        image_file = self.image_files[index]
        label = self.labels[index]
        image = read_dcm_file(image_file)
        print(image_file, '\t', image.GetSize())
        # arr = sitk.GetArrayFromImage(image)
        # # 截取5-14层用来训练
        # arr_slice = arr[5:15, :, :]

    def __len__(self):
        return len(self.image_files)

class FattyLiverClsDatasetsDiff3D(Dataset):
    '''
    输入数据的分辨率统一到(512, 384, 16), 对应(x,y,z)
    '''
    def __init__(self, data_root, config_file, crop_size=[32, 384, 512], phase='train'):
        self.crop_size = crop_size
        self.image_files = []
        self.labels = []
        with open(config_file) as f:
            for line in f.readlines():
                line = line.strip()
                if line is None or len(line) == 0:
                    continue
                ss = line.split('\t')
                image_file = os.path.join(data_root, ss[0])
                if not os.path.isdir(image_file):
                    continue
                self.image_files.append(image_file)
                self.labels.append(int(ss[1]))
            print('====> fatty liver count is:{}'.format(len(self.image_files)))

    def __getitem__(self, index):
        image_path = self.image_files[index]
        label = self.labels[index]
        echo_1_file = os.path.join(image_path, 'echo_1.nii.gz')
        echo_2_file = os.path.join(image_path, 'echo_2.nii.gz')

        image_1 = sitk.ReadImage(echo_1_file)
        image_2 = sitk.ReadImage(echo_2_file)

        arr_1 = sitk.GetArrayFromImage(image_1)
        arr_2 = sitk.GetArrayFromImage(image_2)

        arr_1 = np.array(arr_1, dtype=np.float32)
        arr_2 = np.array(arr_2, dtype=np.float32)

        diff_1_2 = arr_1 - arr_2
        diff_1_2 = diff_1_2 - diff_1_2.min()

        # diff_1_2 = arr_1

        # print(diff_1_2.min())
        # print((arr_2-arr_1).min())
        # print(diff_1_2.dtype)
        # print(diff_1_2.shape)


        diff_1_2_bg = np.zeros(self.crop_size, dtype=np.float32)

        boundary_z = min(diff_1_2.shape[0], diff_1_2_bg.shape[0])
        boundary_y = min(diff_1_2.shape[1], diff_1_2_bg.shape[1])
        
        if diff_1_2_bg.shape[1] > self.crop_size[1]:
            boundary_y_min_max = diff_1_2_bg.shape[1] - self.crop_size[1]
            boundary_y_min = np.random.randint(0, boundary_y_min_max+1)
            boundary_y_max = boundary_y_min + self.crop_size[1]
            diff_1_2_bg[:boundary_z,:,:] = diff_1_2[:boundary_z,boundary_y_min:boundary_y_max,:]
        else:
            diff_1_2_bg[:boundary_z,:boundary_y,:] = diff_1_2[:boundary_z,:boundary_y,:]

        diff_tensor = torch.from_numpy(diff_1_2_bg).float()
        diff_tensor = diff_tensor.unsqueeze(0)

        return diff_tensor, label, image_path

    def __len__(self):
        return len(self.image_files)

def test_FattyLiverClsDatasets():
    data_root = '../data/images_mr_filtered'
    config_file = '../data/config/config_train.txt'
    crop_size = 512
    ds = FattyLiverClsDatasets(data_root, config_file, crop_size, crop_size)
    for i in range(ds.__len__()):
        ds.__getitem__(i)
    print('====>test_FattyLiverClsDatasets finished!')


def test_FattyLiverClsDatasetsDiff3D():
    data_root = '../data/experiment_0/0.ori'
    config_file = '../data/config/config_train.txt'
    crop_size = [16, 384, 512]
    ds = FattyLiverClsDatasetsDiff3D(data_root, config_file, crop_size)
    dataloader = DataLoader(ds, batch_size=2, shuffle=True, pin_memory=True)
    for index, (images, labels, names) in enumerate(dataloader):
        # print(images.shape)
        pass


if  __name__ == '__main__':
    # fire.Fire()
    # print(read_config_file('../data/config/肝穿病人给放射科和合作者.xlsx'))

    # split_data_to_train_val_test('../data/images_mr_filtered', '../data/config/肝穿病人给放射科和合作者宋筛重复序列终版.xlsx', '../data/config', 0.7, 0.1)
    # test_FattyLiverClsDatasets()
    test_FattyLiverClsDatasetsDiff3D()
    
    # split_data_to_two_phase_one_case('../data/images_mr_filtered/1.3.12.2.1107.5.2.30.25245.2015120320185731080640838.0.0.0', '')
    # resample_data_one_case('../data/experiment_0/0.ori/1.3.12.2.1107.5.2.30.25245.2015120320185731080640838.0.0.0/echo_1', '')