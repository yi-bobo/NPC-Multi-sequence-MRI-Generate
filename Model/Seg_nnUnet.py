
import os
import math
import torch
import random
import numpy as np
import nibabel as nib
from PIL import Image
import torch.nn as nn
from monai.losses import DiceLoss
from monai.networks.nets import DynUNet

from Utils.cal_util import cal_dice, cal_iou, cal_precision
from Utils.optim_util import create_optimizer, create_scheduler


class SegDynUNet(nn.Module):
    def __init__(self, opt):
        super(SegDynUNet, self).__init__()
        self.sour_name = opt.data.sour_img_name
        self.targ_name = opt.data.targ_img_name

        self.device = opt.train.device
        self.patch_size = opt.data.patch_image_shape
        self.val_overlap = opt.val.overlap

        self.lambda_bce = opt.loss.lambda_bce
        self.lambda_dice = opt.loss.lambda_dice

        self.net = DynUNet(
            spatial_dims=opt.net.spatial_dims,
            in_channels=opt.net.in_channels,
            out_channels=opt.net.out_channels,
            kernel_size=opt.net.kernel_size,
            strides=opt.net.strides,
            upsample_kernel_size=opt.net.upsample_kernel_size,
            filters=opt.net.filters,
            dropout=opt.net.dropout,
            norm_name=opt.net.norm_name,
            res_block=opt.net.res_block,
        )
        self.loss_bce = nn.BCEWithLogitsLoss()
        # self.loss_dice = DiceLoss(to_onehot_y=True, softmax=True, squared_pred=True, reduction="mean")
        self.loss_dice = DiceLoss(sigmoid=True, to_onehot_y=False, squared_pred=False, reduction="mean")  # ❗直接对单通道做 sigmoid，再计算 Dice，而不是做 softmax + one-hot

        self.optimizer = create_optimizer(opt.optim.optimizer.name, self.net.parameters(), **vars(opt.optim.optimizer.params))
        self.scheduler = create_scheduler(opt.optim.scheduler.name, self.optimizer, **vars(opt.optim.scheduler.params))

    def set_input(self, data):
        data_mapping = {
            'T1': data[0],
            'T1C': data[1],
            'T1C_mask': data[2],
            'T2': data[3],
            'T2_mask': data[4],
            'T1C_tumor': data[5],
            'T2_tumor': data[6],
        }
        input_data = data_mapping[self.sour_name].to(self.device)
        mask_data = data_mapping[self.targ_name].to(self.device)
        # 提取病人ID
        patient_id = data[-2][0]

        return input_data, mask_data, patient_id

    def forward(self, img, mask):
        self.net.train()
        self.optimizer.zero_grad()

        pred = self.net(img)
        loss_dice = self.loss_dice(pred, mask)
        loss_bce = self.loss_bce(pred, mask)
        loss = loss_dice * self.lambda_dice + loss_bce * self.lambda_bce
        loss.backward()
        self.optimizer.step()

        loss_dict = {
            'loss': loss.item(),
            'loss_dice': loss_dice.item(),
            'loss_bce': loss_bce.item()
        }

        return loss_dict
    
    def val(self, input, mask):
        self.net.eval()  # 设置模型为评估模式

        # 初始化预测结果张量和权重张量
        pred_targ = torch.zeros_like(input, device=input.device)  # 预测结果张量
        weight = torch.zeros_like(input, device=input.device)  # 用于记录每个像素的加权次数

        # 获取输入图像的维度
        b, c, depth, height, width = input.shape
        patch_d, patch_h, patch_w = self.patch_size  # 获取块大小
        stride_d = int(patch_d * (1 - self.val_overlap))  # 深度方向步长
        stride_h = int(patch_h * (1 - self.val_overlap))  # 高度方向步长
        stride_w = int(patch_w * (1 - self.val_overlap))  # 宽度方向步长

        # 初始化批次大小
        batch_size = 8

        # 生成裁剪块的起始索引
        d_start_list = list(range(0, depth - patch_d + 1, stride_d))
        h_start_list = list(range(0, height - patch_h + 1, stride_h))
        w_start_list = list(range(0, width - patch_w + 1, stride_w))

        # 如果最后一块没有覆盖到边缘，添加最后一块的起始位置
        if depth - patch_d % stride_d != 0:
            d_start_list.append(depth - patch_d)
        if height - patch_h % stride_h != 0:
            h_start_list.append(height - patch_h)
        if width - patch_w % stride_w != 0:
            w_start_list.append(width - patch_w)

        # 生成所有块的索引组合
        patch_indices = [(d, h, w) for d in d_start_list for h in h_start_list for w in w_start_list]

        # 按批次处理裁剪块
        total_patches = len(patch_indices)
        batch_num = math.ceil(total_patches / batch_size)  # 总批次数

        for i in range(batch_num):
            # 获取当前批次的索引
            batch_indices = patch_indices[i * batch_size: (i + 1) * batch_size]
            actual_batch_size = len(batch_indices)

            # 初始化当前批次张量
            input_batch = torch.zeros((actual_batch_size, c, patch_d, patch_h, patch_w), device=input.device)

            # 填充当前批次的输入
            for j, (d_start, h_start, w_start) in enumerate(batch_indices):
                input_batch[j] = input[:, :, d_start:d_start + patch_d, h_start:h_start + patch_h, w_start:w_start + patch_w]
                
           
            # 推理当前批次
            with torch.no_grad():
                pred_targ_batch = self.net(input_batch)

            # 将生成结果拼接回预测张量
            for j, (d_start, h_start, w_start) in enumerate(batch_indices):
                pred_targ[:, :, d_start:d_start + patch_d, h_start:h_start + patch_h, w_start:w_start + patch_w] += pred_targ_batch[j]
                weight[:, :, d_start:d_start + patch_d, h_start:h_start + patch_h, w_start:w_start + patch_w] += 1

        # 对重叠区域进行归一化
        pred_targ /= weight

        # 计算评估指标
        pred_targ = (pred_targ > 0.5).float()
        pred_mask = pred_targ.cpu().squeeze(0).squeeze(0).numpy()
        mask = mask.cpu().squeeze(0).squeeze(0).numpy()
        input = input.cpu().squeeze(0).squeeze(0).numpy()
        dice = cal_dice(pred_mask, mask)
        iou = cal_iou(pred_mask, mask)
        precison = cal_precision(pred_mask, mask)

        return input, mask,  pred_mask, dice, iou, precison

    def plot(self, img, mask, pred, path):
        # 保存为nii
        self.save_nii(mask, os.path.join(path, "mask.nii"))
        self.save_nii(pred, os.path.join(path, "pred_mask.nii"))
        self.save_nii(img, os.path.join(path, "img.nii"))

        # 保存为png
        nonzero_slices = []
        for slice in range(pred.shape[0]):
            if mask[slice, :, :].sum() > 0:
                nonzero_slices.append(slice)
        if len(nonzero_slices) != 0:
            slice = random.randint(0, len(nonzero_slices)-1)
            slice = nonzero_slices[slice]
            self.save_png(img[slice, :, :], path + f"img_{slice}.png")
            self.save_png(mask[slice, :, :], path + f"mask_{slice}.png")
            self.save_png(pred[slice, :, :], path + f"pred_mask_{slice}.png")

    def save_png(self, data, path):
        data = (data*255).astype(np.uint8)
        data = Image.fromarray(data)
        data.save(path)

    def save_nii(self, data, path):
        voxel_spacing = [6, 1, 1]  # z,y,x
        affine = np.diag(voxel_spacing + [1])
        nii_img = nib.Nifti1Image(data, affine)
        nib.save(nii_img, path)