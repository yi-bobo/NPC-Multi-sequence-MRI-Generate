import os
import math
import torch
import random
import numpy as np
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from Model.Net.RegGAN_Net.Reg_net import Reg
from Model.Net.GAN_Net.UNet import UNet
from Utils.loss.gan_loss import GANLoss
from Utils.loss.grad_loss import Grad
from Utils.loss.perceptual_loss import VGGLoss_3D
from Utils.optim_util import GradualWarmupScheduler
from .Net.GAN_Net.Discriminator import NLayerDiscriminator
from Model.Net.RegGAN_Net.Blocks.transformer_block import SpatialTransformer


class RegGANModel(nn.Module):
    def __init__(self, opt):
        super(RegGANModel, self).__init__()
        self.multi_gpu = opt.train.multi_gpu  # 是否使用多 GPU
        self.device = opt.train.device  # 设备配置（GPU或CPU）
        self.isTrain = opt.train.is_train  # 是否是训练模式
        self.lambda_L1 = opt.loss.lambda_L1  # L1损失的权重
        self.lambda_GAN = opt.loss.lambda_GAN  # GAN损失的权重
        self.lambda_perc = opt.loss.lambda_perc  # 风格损失的权重
        self.lambda_SR = opt.loss.lambda_SR  # 超分辨损失的权重
        self.lambda_SM = opt.loss.lambda_SM  # 平滑损失的权重
        self.warm_epoch = opt.loss.warm_epoch  # 预热轮数

        self.sour_name = opt.data.sour_img_name  # 源数据名称
        self.targ_name = opt.data.targ_img_name  # 目标数据名称
        self.patch_size = opt.data.patch_image_shape
        self.val_overlap = opt.val.overlap  # 验证时图像重叠的大小

        # 定义生成器（Generator）
        self.netG = UNet(**(vars(opt.net.G)))  # 使用UNet网络作为生成器
            
        # 损失函数和优化器
        if self.isTrain:
            self.netR = Reg(opt.net.R, opt.data.patch_image_shape, int_downsize=2).to(self.device)
            self.netD = NLayerDiscriminator(**(vars(opt.net.D)))  # 定义判别器（Discriminator）
            # configure transformer
            self.SpatialTransformer = SpatialTransformer(opt.data.patch_image_shape).to(self.device)
           
            # GAN损失函数
            self.criterionGAN = GANLoss(opt.loss.gan_mode).to(self.device)
            self.criterionL1 = nn.L1Loss().to(self.device)  # L1损失
            self.smoothing_loss = Grad('l2', loss_mult=2).loss
            self.perceptual_loss = VGGLoss_3D(device=self.device)
            
            # 优化器和学习率调度器
            self.optimizer_G = torch.optim.AdamW(self.netG.parameters(), lr=opt.optim_G.lr, weight_decay=1e-4)
            self.cosineScheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_G, T_max=opt.train.max_epochs, eta_min=0, last_epoch=-1)
            self.warmUpScheduler_G = GradualWarmupScheduler(self.optimizer_G, multiplier=2, warm_epoch=opt.train.max_epochs // 10 , 
                                         after_scheduler=self.cosineScheduler_G)
            self.optimizer_D = torch.optim.AdamW(self.netD.parameters(), lr=opt.optim_D.lr, weight_decay=1e-4)
            self.cosineScheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_D, T_max=opt.train.max_epochs, eta_min=0, last_epoch=-1)
            self.warmUpScheduler_D = GradualWarmupScheduler(self.optimizer_D, multiplier=2, warm_epoch=opt.train.max_epochs // 10, 
                                         after_scheduler=self.cosineScheduler_D)
            self.optimizer_R = torch.optim.AdamW(self.netR.parameters(), lr=opt.optim_G.lr, weight_decay=1e-4)
            self.cosineScheduler_R = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_R, T_max=opt.train.max_epochs, eta_min=0, last_epoch=-1)
            self.warmUpScheduler_R = GradualWarmupScheduler(self.optimizer_R, multiplier=2, warm_epoch=opt.train.max_epochs // 10, 
                                         after_scheduler=self.cosineScheduler_R)

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
    
    def forward(self, input, target, epoch):
        self.real_A = input.to(self.device)  # 输入的真实图像A
        self.real_B = target.to(self.device)  # 目标的真实图像B

        # 通过生成器生成假图像B
        with torch.amp.autocast('cuda'):#使用半精度训练f16
            self.fake_B = self.netG(self.real_A)
            Trans, full_Trans = self.netR(self.fake_B, self.real_B)
            self.SysRegist_A2B = self.SpatialTransformer(self.fake_B, full_Trans)
        
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        
        ##1️⃣ 更新判别器（D）
        if epoch > self.warm_epoch:
            self.optimizer_D.zero_grad()
            # 计算假图像的判别结果
            pred_fake = self.netD(fake_AB.detach(), isDetach=True)  # 使用detach避免计算梯度
            self.loss_D_fake = self.criterionGAN(pred_fake, False)
            # 计算真实图像的判别结果
            real_AB = torch.cat((self.real_A, self.real_B), 1)
            pred_real = self.netD(real_AB, isDetach=False)
            self.loss_D_real = self.criterionGAN(pred_real, True)
            # 判别器的总损失
            self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
            self.loss_D.backward()
            self.optimizer_D.step()
        else:
            self.loss_D = torch.tensor(0.0, device=self.device)
            self.loss_D_fake = torch.tensor(0.0, device=self.device)
            self.loss_D_real = torch.tensor(0.0, device=self.device)

        ##2️⃣ 更新生成器（G）更新配准器（R）
        self.optimizer_G.zero_grad()
        self.optimizer_R.zero_grad()
        # 生成器的损失（针对假图像）
        pred_fake = self.netD(fake_AB, isDetach=True)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # 计算L1损失
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B)
        # 计算感知损失
        self.loss_perc = self.perceptual_loss(self.fake_B.float(), self.real_B)
        # 计算对齐损失
        self.loss_SR = self.criterionL1(self.SysRegist_A2B, self.real_B)  
        self.loss_SM = self.smoothing_loss(Trans)
        # 计算总生成器损失
        self.loss_G = self.lambda_GAN * self.loss_G_GAN + self.lambda_L1 * self.loss_G_L1 + self.lambda_perc * self.loss_perc + self.lambda_SR * self.loss_SR + self.lambda_SM * self.loss_SM
        self.loss_G.backward()
        self.optimizer_G.step()
        self.optimizer_R.step()

        loss_dict = {
            'loss_G': self.loss_G.item(),
            'loss_G_GAN': self.loss_G_GAN.item(),
            'loss_perc': self.loss_perc.item(),
            'loss_SR': self.loss_SR.item(),
            'loss_SM': self.loss_SM.item(),
            'loss_D': self.loss_D.item(),
            'loss_D_fake': self.loss_D_fake.item(),
            'loss_D_real': self.loss_D_real.item(),
        }
        return loss_dict


    def val(self, input, target):
        self.netG.eval()  # 设置模型为评估模式

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
                pred_targ_batch = self.netG(input_batch)

            # 将生成结果拼接回预测张量
            for j, (d_start, h_start, w_start) in enumerate(batch_indices):
                pred_targ[:, :, d_start:d_start + patch_d, h_start:h_start + patch_h, w_start:w_start + patch_w] += pred_targ_batch[j]
                weight[:, :, d_start:d_start + patch_d, h_start:h_start + patch_h, w_start:w_start + patch_w] += 1

        # 对重叠区域进行归一化
        pred_targ /= weight

        input_data = input.cpu().squeeze(0).squeeze(0).numpy()
        target_data = target.cpu().squeeze(0).squeeze(0).numpy()
        pred_targ_data = pred_targ.cpu().squeeze(0).squeeze(0).numpy()

        data_range = max(input_data.max()-input_data.min(), pred_targ_data.max()-pred_targ_data.min())
        mae = np.mean(np.abs(pred_targ_data - target_data))
        SSIM = ssim(pred_targ_data, target_data, data_range=data_range)
        PSNR = psnr(pred_targ_data, target_data, data_range=data_range)

        return input_data, target_data, pred_targ_data, mae, SSIM, PSNR
    
def plot(self, input, target, pred_target, data_mapping, path):
        # 保存为nii
        for key, value in data_mapping.items():
            value = value.cpu().squeeze(0).squeeze(0).numpy()
            self.save_nii(value, os.path.join(path, f"{key}.nii.gz"))
        self.save_nii(input, os.path.join(path, f"{self.sour_name}.nii.gz"))
        self.save_nii(target, os.path.join(path, f"{self.targ_name}.nii.gz"))
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
            self.save_png(self.norm_png(pred_target[slice, :, :]), os.path.join(path, f"pred_{self.targ_name}_{slice}.png"))

