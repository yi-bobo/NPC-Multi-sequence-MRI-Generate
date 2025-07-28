import os
import math
import torch
import random
import itertools
import numpy as np
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from Utils.init_net_util import init_net

from Model.Net.GAN_Net.UNet import UNet
from Utils.loss.gan_loss import GANLoss
from .Net.GAN_Net.Discriminator import NLayerDiscriminator

def set_requires_grad(nets, requires_grad=False):
    """requies_grad=Fasle: 这些网络不需要进行参数更新
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

class CycleGANModel(nn.Module):
    def __init__(self, opt):
        super(CycleGANModel, self).__init__()
        self.multi_gpu = opt.train.multi_gpu  # 是否使用多 GPU
        self.device = opt.train.device  # 设备配置（GPU或CPU）
        self.isTrain = opt.train.is_train  # 是否是训练模式
        self.lambda_L1 = opt.loss.lambda_L1  # L1损失的权重
        self.lambda_GAN = opt.loss.lambda_GAN  # GAN损失的权重
        self.lambda_cycle = opt.loss.lambda_cycle

        self.sour_name = opt.data.sour_img_name  # 源数据名称
        self.targ_name = opt.data.targ_img_name  # 目标数据名称
        self.patch_size = opt.data.patch_image_shape
        self.val_overlap = opt.val.overlap  # 验证时图像重叠的大小

        # 定义生成器（Generator）
        self.netG_A2B = UNet(**(vars(opt.net.G))).to(self.device)  # 使用UNet网络作为生成器
        self.netG_B2A = UNet(**(vars(opt.net.G))).to(self.device)  # 使用UNet网络作为生成器

        self.netG_A2B = init_net(self.netG_A2B, init_type=opt.net.init_type, init_gain=opt.net.init_gain)  # 初始化生成器
        self.netG_B2A = init_net(self.netG_B2A, init_type=opt.net.init_type, init_gain=opt.net.init_gain)  # 初始化生成器

        if self.multi_gpu:
            self.netG_A2B = nn.DataParallel(self.netG_A2B)
            self.netG_B2A = nn.DataParallel(self.netG_B2A)
            
        # 损失函数和优化器
        if self.isTrain:
            self.netD_A2B = NLayerDiscriminator(**(vars(opt.net.D))).to(self.device)  # 定义判别器（Discriminator）
            self.netD_B2A = NLayerDiscriminator(**(vars(opt.net.D))).to(self.device)  # 定义判别器（Discriminator）

            self.netD_A2B = init_net(self.netD_A2B, init_type=opt.net.init_type, init_gain=opt.net.init_gain)  # 初始化判别器
            self.netD_B2A = init_net(self.netD_B2A, init_type=opt.net.init_type, init_gain=opt.net.init_gain)  # 初始化判别器

            if self.multi_gpu:
                self.netD_A2B = nn.DataParallel(self.netD_A2B)
                self.netD_B2A = nn.DataParallel(self.netD_B2A)
           
            # GAN损失函数
            self.criterionGAN = GANLoss(opt.loss.gan_mode, target_real_label=0.9).to(self.device)
            self.criterionL1 = nn.L1Loss().to(self.device)  # L1损失

            # 优化器和学习率调度器
            self.optimizer_G_A2B = torch.optim.Adam(self.netG_A2B.parameters(), lr=5e-4, weight_decay=1e-4, betas = (0.5, 0.999))
            self.optimizer_G_B2A = torch.optim.Adam(self.netG_B2A.parameters(), lr=5e-4, weight_decay=1e-4, betas = (0.5, 0.999)) 
            self.optimizer_D_A2B = torch.optim.Adam(self.netD_A2B.parameters(), lr=1e-6, weight_decay=1e-4, betas = (0.9, 0.999))
            self.optimizer_D_B2A = torch.optim.Adam(self.netD_B2A.parameters(), lr=1e-6, weight_decay=1e-4, betas = (0.9, 0.999))
            self.scheduler_G_A2B = torch.optim.lr_scheduler.StepLR(self.optimizer_G_A2B, step_size=120, gamma=0.9)
            self.scheduler_G_B2A = torch.optim.lr_scheduler.StepLR(self.optimizer_G_B2A, step_size=120, gamma=0.9)
            self.scheduler_D_A2B = torch.optim.lr_scheduler.StepLR(self.optimizer_D_A2B, step_size=120, gamma=0.9)
            self.scheduler_D_B2A = torch.optim.lr_scheduler.StepLR(self.optimizer_D_B2A, step_size=120, gamma=0.9)


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
    
    def forward(self, input, target):
        self.real_A = input  # 源图像
        self.real_B = target  # 目标图像
        # 0️⃣生成器生成假图像
        with torch.amp.autocast('cuda'):
            self.fake_B = self.netG_A2B(self.real_A)
            self.fake_A = self.netG_B2A(self.real_B)
            self.rec_B = self.netG_A2B(self.fake_A.detach())
            self.rec_A = self.netG_B2A(self.fake_B.detach())

        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # 拼接源图像和生成的目标图像
        fake_BA = torch.cat((self.real_B, self.fake_A), 1)
        real_AB = torch.cat((self.real_A, self.real_B), 1)  # 拼接源图像和目标图像
        real_BA = torch.cat((self.real_B, self.real_A), 1)
        
        #1️⃣更新判别器
        self.optimizer_D_A2B.zero_grad()
        ## 判别器 A2B
        pred_fake_A2B = self.netD_A2B(fake_AB.detach(), isDetach=True)  # 判别器对生成的目标图像进行预测
        pred_real_A2B = self.netD_A2B(real_AB, isDetach=False)  # 判别器对真实的目标图像进行预测
        self.loss_D_A2B = (self.criterionGAN(pred_fake_A2B, False) + self.criterionGAN(pred_real_A2B, True)) * 0.5
        self.loss_D_A2B.backward()  # 判别器损失反向传播
        self.optimizer_D_A2B.step()  # 更新判别器 A2B 的

        self.optimizer_D_B2A.zero_grad()
        ## 判别器 B2A
        pred_fake_B2A = self.netD_B2A(fake_BA.detach(), isDetach=True)  # 判别器对生成的源图像进行预测
        pred_real_B2A = self.netD_B2A(real_BA, isDetach=False)  # 判别器对真实的源
        self.loss_D_B2A = (self.criterionGAN(pred_fake_B2A, False) + self.criterionGAN(pred_real_B2A, True)) * 0.5
        self.loss_D_B2A.backward()  # 判别器损失反向传播
        self.optimizer_D_B2A.step()  # 更新判别器 B2A 的
        

        # 2️⃣更新生成器
        self.optimizer_G_A2B.zero_grad()
        ## 生成器 A2B
        pred_fake_A2B = self.netD_A2B(fake_AB, isDetach=False)  # 判别器对生成的目标图像进行预测
        loss_G_GAN_A2B = self.criterionGAN(pred_fake_A2B, True)  # GAN损失
        loss_G_L1_A2B = self.criterionL1(self.fake_B, self.real_B)  # L1损失
        loss_G_cycle_A2B = self.criterionL1(self.rec_B, self.real_B)  # 循环一致性损失
        self.loss_G_A2B = loss_G_GAN_A2B * self.lambda_GAN + loss_G_L1_A2B * self.lambda_L1 + loss_G_cycle_A2B * self.lambda_cycle
        self.loss_G_A2B.backward()  # 生成器损失反向传播
        self.optimizer_G_A2B.step()  # 更新生成器 A2B 的
        
        self.optimizer_G_B2A.zero_grad()
        ## 生成器 B2A
        pred_fake_B2A = self.netD_B2A(fake_BA, isDetach=False)  # 判别器对生成的源图像进行预测
        loss_G_GAN_B2A = self.criterionGAN(pred_fake_B2A, True)  # GAN损失
        loss_G_L1_B2A = self.criterionL1(self.fake_A, self.real_A)  # L1损失
        loss_G_cycle_B2A = self.criterionL1(self.rec_A, self.real_A)  # 循环一致性损失
        self.loss_G_B2A = loss_G_GAN_B2A * self.lambda_GAN + loss_G_L1_B2A * self.lambda_L1 + loss_G_cycle_B2A * self.lambda_cycle
        self.loss_G_B2A.backward()  # 生成器损失反向传播
        self.optimizer_G_B2A.step()

        # 更新学习率
        lr_G_A2B = self.optimizer_G_A2B.param_groups[0]['lr']
        lr_G_B2A = self.optimizer_G_B2A.param_groups[0]['lr']
        lr_D_A2B = self.optimizer_D_A2B.param_groups[0]['lr']
        lr_D_B2A = self.optimizer_D_B2A.param_groups[0]['lr']
        if lr_G_A2B > 1e-6:
            self.scheduler_G_A2B.step()
        if lr_G_B2A > 1e-6:
            self.scheduler_G_B2A.step()
        if lr_D_A2B > 1e-8:
            self.scheduler_D_A2B.step()
        if lr_D_B2A > 1e-8:
            self.scheduler_D_B2A.step()

        loss_dict = {
            'loss_G_A2B': self.loss_G_A2B.item(),
            'loss_G_A2B_GAN': loss_G_GAN_A2B.item(),
            'loss_G_A2B_L1': loss_G_L1_A2B.item(),
            'loss_G_A2B_cycle': loss_G_cycle_A2B.item(),
            'loss_G_B2A': self.loss_G_B2A.item(),
            'loss_G_B2A_GAN': loss_G_GAN_B2A.item(),
            'loss_G_B2A_L1': loss_G_L1_B2A.item(),
            'loss_G_B2A_cycle': loss_G_cycle_B2A.item(),
            'loss_D_A2B': self.loss_D_A2B.item(),
            'loss_D_B2A': self.loss_D_B2A.item(),
        }


        return loss_dict

    def val(self, input, target):
        self.netG_A2B.eval()  # 设置模型为评估模式
        self.netG_B2A.eval()  # 设置模型为评估模式

        # 初始化预测结果张量和权重张量
        pred_targ = torch.zeros_like(input, device=input.device)  # 预测结果张量
        pred_input = torch.zeros_like(target, device=target.device)  # 预测结果张量
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
            target_batch = torch.zeros((actual_batch_size, c, patch_d, patch_h, patch_w), device=target.device)

            # 填充当前批次的输入
            for j, (d_start, h_start, w_start) in enumerate(batch_indices):
                input_batch[j] = input[:, :, d_start:d_start + patch_d, h_start:h_start + patch_h, w_start:w_start + patch_w]
                target_batch[j] = target[:, :, d_start:d_start + patch_d, h_start:h_start + patch_h, w_start:w_start + patch_w]
                
           
            # 推理当前批次
            with torch.no_grad():
                pred_targ_batch = self.netG_A2B(input_batch)
            with torch.no_grad():
                pred_input_batch = self.netG_B2A(target_batch)

            # 将生成结果拼接回预测张量
            for j, (d_start, h_start, w_start) in enumerate(batch_indices):
                pred_targ[:, :, d_start:d_start + patch_d, h_start:h_start + patch_h, w_start:w_start + patch_w] += pred_targ_batch[j]
                pred_input[:, :, d_start:d_start + patch_d, h_start:h_start + patch_h, w_start:w_start + patch_w] += pred_input_batch[j]
                weight[:, :, d_start:d_start + patch_d, h_start:h_start + patch_h, w_start:w_start + patch_w] += 1

        # 对重叠区域进行归一化
        pred_targ /= weight
        pred_input /= weight

        input_data = input.cpu().squeeze(0).squeeze(0).numpy()
        target_data = target.cpu().squeeze(0).squeeze(0).numpy()
        pred_targ_data = pred_targ.cpu().squeeze(0).squeeze(0).numpy()
        pred_input_data = pred_input.cpu().squeeze(0).squeeze(0).numpy()

        data_range_targ = max(input_data.max()-input_data.min(), pred_targ_data.max()-pred_targ_data.min())
        mae_targ = np.mean(np.abs(pred_targ_data - target_data))
        SSIM_targ = ssim(pred_targ_data, target_data, data_range=data_range_targ)
        PSNR_targ = psnr(pred_targ_data, target_data, data_range=data_range_targ)

        data_range_input = max(target_data.max()-target_data.min(), pred_input_data.max()-pred_input_data.min())
        mae_input = np.mean(np.abs(pred_input_data - input_data))
        SSIM_input = ssim(pred_input_data, input_data, data_range=data_range_input)
        PSNR_input = psnr(pred_input_data, input_data, data_range=data_range_input)

        return input_data, target_data, pred_targ_data, pred_input_data, mae_targ, SSIM_targ, PSNR_targ, mae_input, SSIM_input, PSNR_input
    
def plot(self, input, target, pred_target, pred_input, data_mapping, path):
        # 保存为nii
        for key, value in data_mapping.items():
            value = value.cpu().squeeze(0).squeeze(0).numpy()
            self.save_nii(value, os.path.join(path, f"{key}.nii.gz"))
        self.save_nii(input, os.path.join(path, f"{self.sour_name}.nii.gz"))
        self.save_nii(target, os.path.join(path, f"{self.targ_name}.nii.gz"))
        self.save_nii(pred_input, os.path.join(path, f"pred_{self.sour_name}.nii.gz"))
        self.save_nii(pred_target, os.path.join(path, f"pred_{self.targ_name}.nii.gz"))

        # 保存为png
        nonzero_slices = []
        for slice in range(pred_target.shape[0]):
            if target[slice, :, :].sum() > 0:
                nonzero_slices.append(slice)
        if len(nonzero_slices) != 0:
            slice = random.randint(0, len(nonzero_slices)-1)
            slice = nonzero_slices[slice]
            self.save_png(self.norm_png(input[slice, :, :]), os.path.join(path, f"{self.sour_name}_{slice}.png"))
            self.save_png(self.norm_png(target[slice, :, :]), os.path.join(path, f"{self.targ_name}_{slice}.png"))
            self.save_png(self.norm_png(pred_input[slice, :, :]), os.path.join(path, f"pred_{self.sour_name}_{slice}.png"))
            self.save_png(self.norm_png(pred_target[slice, :, :]), os.path.join(path, f"pred_{self.targ_name}_{slice}.png"))

