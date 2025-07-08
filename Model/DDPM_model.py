# DM模型参考链接：https://shao.fun/blog/w/how-diffusion-models-work.html

import os
import math
import torch
import random
import itertools
import numpy as np
import nibabel as nib
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from .Net.DDPM_Net.DDPM_Net import DiffusionModelUNet
from Utils.optim_util import create_optimizer, GradualWarmupScheduler

def creat_net(device, **kwargs):
    net = DiffusionModelUNet(device=device, **kwargs)
    return net

class DDPMModel(nn.Module):
    "将输入的x_t和y进行concat后作为训练，x为源图像，y为目标图像； 计算预测噪声损失和图片重建损失来约束网络重建"
    def __init__(self, opt):
        super(DDPMModel, self).__init__()
        self.is_train = opt.train.is_train
        self.device = opt.train.device
        self.lambda_eps = opt.loss.lambda_eps
        self.lambda_l1 = opt.loss.lambda_l1

        self.sour_name = opt.data.sour_img_name
        self.targ_name = opt.data.targ_img_name
        self.patch_size = opt.data.patch_image_shape  # 图像分块大小
        self.val_overlap = opt.val.overlap  # 重叠度

        self.num_timesteps = opt.model.num_timesteps
        self.batch_size = opt.train.batch_size
        self.spitial_dims = opt.net.spatial_dims

        if self.spitial_dims == 2:
            self.expand_dims = [1, 1, 1]
        elif self.spitial_dims == 3:
            self.expand_dims = [1, 1, 1, 1]

        # Generator Network (UNet)
        self.net = creat_net(device=self.device, **(vars(opt.net)))

        if self.is_train:
            
            # Loss Functions
            self.criterionBCE = nn.BCELoss().to(self.device)  # 交叉熵损失函数
            self.criterionL1 = nn.L1Loss().to(self.device)  # L1损失函数

            # Optimizers and Learning Rate Schedulers
            scheduler_params = {
                "T_max": opt.train.max_epochs, "eta_min": 0, "last_epoch": -1
            }
            warmup_epochs = 100
            self.optimizer = create_optimizer(opt.optim.optimizer.name, self.net.parameters(), **vars(opt.optim.optimizer.params))
            self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=opt.optim.scheduler.params.multiplier, warm_epoch=warmup_epochs, 
                                                    after_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **scheduler_params)
            ) 
        
        # 参数
        # \beta_t
        self.beta = torch.linspace(1e-4, 0.02, self.num_timesteps).to(self.device)
        # \alpha_t = 1 - \beta_t
        self.alpha = 1 - self.beta
        # \alpha_t_bar = \prod_{s=1}^{t} \alpha_s
        self.alpha_bar = torch.cumprod(1 - self.beta, dim=0).to(self.device)

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

        # 获取源数据和目标数据，并从 data_mapping 中移除
        input_data = data_mapping.pop(self.sour_name).to(self.device)
        target_data = data_mapping.pop(self.targ_name).to(self.device)
        # 提取病人 ID
        patient_id = data[-2][0]
        
        data_mapping.pop('T1C_tumor')
        data_mapping.pop('T2_tumor')

        return input_data, target_data, patient_id, data_mapping
    
    def q_sample(self, x, eps, t):
        'x_0 -> x_t'
        # 计算参数
        sqrt_alpha_bar_t = self.alpha_bar[t].sqrt().view(-1, *self.expand_dims)
        sqrt_1_minus_alpha_bar_t = (1 - self.alpha_bar[t]).sqrt().view(-1, *self.expand_dims)
        # 计算采样结果
        x_t = sqrt_alpha_bar_t * x + sqrt_1_minus_alpha_bar_t * eps

        return x_t
    
    def p_sample(self, x_t, eps, t):
        'x_t -> x_{t-1}'
        # 计算均值参数
        sqrt_alpha_t = self.alpha[t].sqrt().view(-1, *self.expand_dims)
        beta_t = self.beta[t].view(-1, *self.expand_dims)
        sqrt_1_minus_alpha_bar_t = (1 - self.alpha_bar[t]).sqrt().view(-1, *self.expand_dims)

        # 计算均值
        mean = (1 / sqrt_alpha_t) * (x_t - beta_t * eps / sqrt_1_minus_alpha_bar_t)
        # 计算方差
        var = beta_t * ((1 - self.alpha_bar[t-1]) / (1 - self.alpha_bar[t])).view(-1, *([1] * (x_t.dim() - 1)))

        x_t_minus_1 = mean + var * torch.randn_like(x_t).to(self.device)

        return x_t_minus_1
    
    def p_sample_loop(self, x, y):
        '将y作为条件引导扩散模型从 x_T生成x_0 '
        # 生成初始噪声
        x_t = torch.randn_like(x).to(self.device)
        # 逐步采样
        for i, t_val in tqdm(enumerate(reversed(range(self.num_timesteps))), total=self.num_timesteps, desc="Time Step Progress", unit="step"):
        # for i in reversed(range(self.num_timesteps)):
            t = torch.full((x.shape[0],), t_val, device=self.device).long()
            # 拼接当前状态和条件 y 作为网络输入
            input = torch.cat([x_t, y], dim=1)
            pred_eps = self.net(input, t)

            # 如果不是最后一步（t > 0），继续采样
            if t_val > 0:
                x_t = self.p_sample(x_t, pred_eps, t)
            else:
                # t == 0，输出最终图像，无需再加噪声
                x_0 = (1 / self.alpha[t].sqrt().view(-1, *self.expand_dims)) * (x_t - (1 - self.alpha_bar[t]).sqrt().view(-1, *self.expand_dims) * pred_eps)
                return x_0  # 直接返回最终预测结果
    
    def forward(self, input_data, target_data):
        y = input_data
        x = target_data
        
        self.optimizer.zero_grad()

        # 生成随机时间步
        t = torch.randint(0, self.num_timesteps, (self.batch_size,)).to(self.device)
        eps = torch.randn_like(x).to(self.device)

        # 采样
        x_t = self.q_sample(x, eps, t)
        # 计算网络输出
        input = torch.cat([x_t, y], dim=1)
        pred_eps = self.net(input, t)
        # 计算噪声预测损失
        loss_eps = self.criterionL1(pred_eps, eps)
        # 计算图片重建损失
        pred_x_t_minus_1 = self.p_sample(x_t, pred_eps, t)
        real_x_t_minus_1 = self.p_sample(x_t, eps, t)
        loss_l1 = self.criterionL1(pred_x_t_minus_1, real_x_t_minus_1)
        # 计算总损失
        loss = self.lambda_eps * loss_eps + self.lambda_l1 * loss_l1

        # 反向传播
        loss.backward()
        self.optimizer.step()

        loss_dict = {
            'loss': loss.item(),
            'loss_eps': loss_eps.item(),
            'loss_l1': loss_l1.item()
        }

        return loss_dict
    
    def val(self, input_data, targ_data):
        self.net.eval()  # 设置模型为评估模式

        x = targ_data
        y = input_data

        # 初始化预测结果张量和权重张量
        pred_targ = torch.zeros_like(x, device=x.device)  # 预测结果张量
        weight = torch.zeros_like(x, device=x.device)  # 用于记录每个像素的加权次数

        # 获取输入图像的维度
        b, c, depth, height, width = x.shape
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
            x_batch = torch.zeros((actual_batch_size, c, patch_d, patch_h, patch_w), device=x.device)
            y_batch = torch.zeros((actual_batch_size, c, patch_d, patch_h, patch_w), device=x.device)

            # 填充当前批次的输入
            for j, (d_start, h_start, w_start) in enumerate(batch_indices):
                x_batch[j] = x[:, :, d_start:d_start + patch_d, h_start:h_start + patch_h, w_start:w_start + patch_w]
                y_batch[j] = y[:, :, d_start:d_start + patch_d, h_start:h_start + patch_h, w_start:w_start + patch_w]
                
           
            # 推理当前批次
            with torch.no_grad():
                pred_targ_batch = self.p_sample_loop(x_batch, y_batch)

            # 将生成结果拼接回预测张量
            for j, (d_start, h_start, w_start) in enumerate(batch_indices):
                pred_targ[:, :, d_start:d_start + patch_d, h_start:h_start + patch_h, w_start:w_start + patch_w] += pred_targ_batch[j]
                weight[:, :, d_start:d_start + patch_d, h_start:h_start + patch_h, w_start:w_start + patch_w] += 1

        # 对重叠区域进行归一化
        pred_targ /= weight

        sour = input_data.cpu().squeeze(0).squeeze(0).numpy()
        targ = targ_data.cpu().squeeze(0).squeeze(0).numpy()
        pred_targ_data = pred_targ.cpu().squeeze(0).squeeze(0).numpy()

        data_range = max(targ.max()-targ.min(), pred_targ_data.max()-pred_targ_data.min())
        mae = np.mean(np.abs(pred_targ_data - targ))
        SSIM = ssim(pred_targ_data, targ, data_range=data_range)
        PSNR = psnr(pred_targ_data, targ, data_range=data_range)

        return sour, targ, pred_targ_data, mae, SSIM, PSNR
    

    def plot(self, x, y, pred_x, path):
        # 保存为nii
        self.save_nii(x, os.path.join(path, f"{self.sour_name}.nii"))
        self.save_nii(pred_x, os.path.join(path, f"pred_{self.sour_name}.nii"))
        self.save_nii(y, os.path.join(path, f"{self.targ_name}.nii"))

        # 保存为png
        nonzero_slices = []
        for slice in range(pred_x.shape[0]):
            if y[slice, :, :].sum() > 0:
                nonzero_slices.append(slice)
        if len(nonzero_slices) != 0:
            slice = random.randint(0, len(nonzero_slices)-1)
            slice = nonzero_slices[slice]
            self.save_png(x[slice, :, :], os.path.join(path, f"{self.sour_name}_{slice}.png"))
            self.save_png(y[slice, :, :], os.path.join(path, f"{self.targ_name}_{slice}.png"))
            self.save_png(pred_x[slice, :, :], os.path.join(path, f"pred_{self.sour_name}_{slice}.png"))

    def save_png(self, data, path):
        data = (data*255).astype(np.uint8)
        data = Image.fromarray(data)
        data.save(path)

    def save_nii(self, data, path):
        voxel_spacing = [6, 1, 1]  # z,y,x
        affine = np.diag(voxel_spacing + [1])
        nii_x = nib.Nifti1Image(data, affine)
        nib.save(nii_x, path)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set the 'requires_grad' flag for all parameters of the networks."""
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
