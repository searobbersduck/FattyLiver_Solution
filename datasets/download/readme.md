## 分割标注数据下载

-----
### 原始数据下载
输入数据`../../data/config_raw/image_anno_TASK_3491.csv`

### 下载数据
- [x] 下载mask数据，调用如下命令`python download_fatty_seg_masks.py download_masks ../../data/seg_task/masks ../../data/config_raw/image_anno_TASK_3491.csv`
- [x] 重命名mask，调用如下命令`python download_fatty_seg_masks.py rename_mask_files '../../data/seg_task/masks' '../../data/seg_task/renamed_masks' '../../data/config_raw/image_anno_TASK_3491.csv'`, 删除问题数据`1.3.12.2.1107.5.2.30.25245.2012082417101060922979046.0.0.0`
- [x] 将相应的序列数据copy到指定文件夹下，源文件夹`../../data/experiment_0/0.ori`, 目标文件夹`../../data/seg_task/images`, 调用命令
- [ ]  