
import sys
sys.path.append('/data1/weiyibo/NPC-MRI/Code/Pctch_model/')

import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from Utils.data_util import norm_ln_1, norm_ln_2, norm_log_1, norm_log_2, norm_mean_1, resize_arr, norm_global

class npy_3D_dataset(Dataset):
    def __init__(self, opt, mode):
            self.mode = mode  # 模式：train|val|test
            self.split_dir = opt.split_dir  # 划分数据的保存父路径
            self.norm_mode = opt.norm_mode  # 归一化模式
            self.img_size = opt.resize_image_shape  # 图像尺寸

            self.norm_method = norm_global # 默认归一化方法
            self.norm_methods = {
                 'global_ln': norm_global,
                 'ln_1': norm_ln_1,
                 'ln_2': norm_ln_2,
                 'log_1': norm_log_1,
                 'log_2': norm_log_2,
                 'mean_1': norm_mean_1
            }

            with open(os.path.join(self.split_dir, f'{self.mode}_with_info.txt'), 'r') as f:
                self.data_list = []
                for line in f.readlines():
                    self.data_list.append(line)


    def __getitem__(self, index):
        path = (self.data_list[index].strip()).split('&')[0]
        txt = (self.data_list[index].strip()).split('&')[1]

        path_dir = path.strip()
        patient_id = path_dir.split('/')[-1]

        # 加载数据
        T1_data = np.load(os.path.join(path_dir, 'T1.npy'))
        T1C_data = np.load(os.path.join(path_dir, 'T1C.npy'))
        T1C_mask_data = np.load(os.path.join(path_dir, 'T1C_mask.npy'))
        T2_data = np.load(os.path.join(path_dir, 'T2.npy'))
        T2_mask_data = np.load(os.path.join(path_dir, 'T2_mask.npy'))
        T1C_tumor_data = T1C_data * T1C_mask_data
        T2_tumor_data = T2_data * T2_mask_data

        # 计算单个病人的最大最小值
        max_value = max(np.max(T1_data), np.max(T1C_data), np.max(T2_data))
        min_value = min(np.min(T1_data), np.min(T1C_data), np.min(T2_data))

        # 归一化
        T1_data = self.norm_method(T1_data, max_value, min_value)
        T1C_data = self.norm_method(T1C_data, max_value, min_value)
        T2_data = self.norm_method(T2_data, max_value, min_value)
        T1C_tumor_data = self.norm_method(T1C_tumor_data, max_value, min_value)
        T2_tumor_data = self.norm_method(T2_tumor_data, max_value, min_value)

        # resize
        T1_data = resize_arr(T1_data, self.img_size)
        T1C_data = resize_arr(T1C_data, self.img_size)
        T1C_mask_data = resize_arr(T1C_mask_data, self.img_size)
        T2_data = resize_arr(T2_data, self.img_size)
        T2_mask_data = resize_arr(T2_mask_data, self.img_size)
        T1C_tumor_data = resize_arr(T1C_tumor_data, self.img_size)
        T2_tumor_data = resize_arr(T2_tumor_data, self.img_size)

        # 转换为 tensor
        T1_data = torch.from_numpy(T1_data).float().unsqueeze(0)
        T1C_data = torch.from_numpy(T1C_data).float().unsqueeze(0)
        T1C_mask_data = torch.from_numpy(T1C_mask_data).float().unsqueeze(0)
        T2_data = torch.from_numpy(T2_data).float().unsqueeze(0)
        T2_mask_data = torch.from_numpy(T2_mask_data).float().unsqueeze(0)
        T1C_tumor_data = torch.from_numpy(T1C_tumor_data).float().unsqueeze(0)
        T2_tumor_data = torch.from_numpy(T2_tumor_data).float().unsqueeze(0)

        return T1_data, T1C_data, T1C_mask_data, T2_data, T2_mask_data, T1C_tumor_data, T2_tumor_data, patient_id, txt

    def __len__(self):
        return len(self.data_list)
    
    def block_val_img(self, img):
        depth_start = random.randint(0, img.shape[2] - self.img_size[0])
        img = img[:, :, depth_start:depth_start+self.img_size[0],:,:]
        return img

# # region 以下为测试代码
# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--split_dir', type=str, default="./Split/zhongshan2", help='split data dir')
#     parser.add_argument('--norm_mode', type=str, default='global_ln', help='normalization mode')
#     parser.add_argument('--resize_image_shape', type=tuple, default=[36,256,256], help='image size')
#     opt = parser.parse_args()

#     dataset = npy_3D_dataset(opt, 'train')
#     print(len(dataset))
#     for i in range(len(dataset)):
#         data = dataset[i]
#         print(data[0].shape, data[1].shape, data[2].shape, data[3].shape, data[4].shape, data[5].shape, data[6].shape, data[7], data[8])
#         break