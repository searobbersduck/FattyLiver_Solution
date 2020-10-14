

## 数据预处理
- [x] 1.将数据按照两个期相分开，调用命令
  ```
  python FattyLiverDatasets.py split_data_to_two_phase_singletask '../data/images_mr_filtered' '../data/experiment_0/0.ori'  '../data/config/肝穿病人给放射科和合作者.xlsx'
  ```
  输出目录为'../data/experiment_0/0.ori'，结构如下
  ```
  
    .
    ├── 1.3.12.2.1107.5.2.18.52001.2014111920254623363975828.0.0.0
    │   ├── echo_1
    │   └── echo_2

  ```

- [x] 2. 将任务1中数据重采样到统一分辨率
  - [ ] 将数据保存成nii.gz的格式， 调用命令`python FattyLiverDatasets.py resample_data_multiprocessing '../data/experiment_0/0.ori'`


