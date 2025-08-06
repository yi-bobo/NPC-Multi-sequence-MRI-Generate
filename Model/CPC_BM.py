
import sys
sys.path.append('/data1/weiyibo/NPC-MRI/Code/Pctch_model/')

import os
import time
import math
import torch
import random
import warnings
import numpy as np
from PIL import Image
import nibabel as nib
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from torch.nn import DataParallel
from torch.cuda.amp import autocast, GradScaler

from inspect import isfunction
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from Utils.init_net_util import init_net
from Utils.loss.perceptual_loss import VGGLoss_3D
from Model.Net.CPC_BM_Net.UNet import DiffusionModelUNet
from Utils.optim_util import create_optimizer,  GradualWarmupScheduler


# region 辅助函数
def creat_net(device, **kwargs):
    net = DiffusionModelUNet(device=device, **kwargs)
    return net

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def grad_norm(model):
    """
    计算给定 model 的所有参数梯度的全局 2-范数。
    需要在 loss.backward() 之后、optimizer.step() 之前调用。
    """
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            # p.grad.data.norm(2) 返回该张量梯度的 2-范数
            param_norm = p.grad.data.norm(2)
            total_norm_sq += param_norm.item() ** 2
    total_norm = total_norm_sq ** 0.5
    return total_norm
# endregion

# region CBM_model

class CPC_BM(nn.Module):
    def __init__(self, opt):
        super(CPC_BM, self).__init__()
        self.is_train = opt.train.is_train
        self.device = opt.train.device
        self.multi_gpu = opt.train.multi_gpu  # 是否使用多 GPU
        self.batch_size = opt.train.batch_size
        self.accumulate_grad_batches = opt.train.accumulate_grad_batches  # 梯度累积步数

        self.img_con_num = 3 # opt.net.image_channels  # 条件图像数量
        self.patch_size = opt.data.patch_image_shape  # 图像分块大小
        self.val_overlap = opt.val.overlap  # 重叠度
        self.sour_name = opt.data.sour_img_name  # 源域数据名称
        self.targ_name = opt.data.targ_img_name  # 目标域数据名称

        self.is_text = opt.train.is_text  # 是否采用文本特征
        self.is_img = opt.train.is_image# 是否采用图像特征

        self.loss_type = opt.loss.loss_type
        self.lambda_con = opt.loss.lambda_con
        self.lambda_rec = opt.loss.lambda_rec
        self.lambda_cyc = opt.loss.lambda_cyc
        self.lambda_per = opt.loss.lambda_per
        self.b_scale = opt.loss.b_scale
        self.use_perceptual = opt.loss.is_perceptual

        self.condition_key = opt.ddbm.condition_key  # 是否采用条件引导
        self.num_timesteps = opt.ddbm.num_timesteps  # 时序步数
        self.mt_type = opt.ddbm.mt_type  # 定义 mt 类型
        self.max_var = opt.ddbm.max_var  # 最大方差
        self.skip_sample = opt.ddbm.skip_sample  # 是否跳过采样
        self.sample_type = opt.ddbm.sample_type  # 采样类型
        self.sample_step = opt.ddbm.sample_step  # 采样步数
        self.eta = opt.ddbm.eta

        self.register_schedule()

        # Generator Network (UNet)
        self.net_f = creat_net(device=self.device, **(vars(opt.net)))
        self.net_b = creat_net(device=self.device, **(vars(opt.net)))

        # Initialising the network
        self.net_f = init_net(self.net_f, init_type=opt.init.init_type, init_gain=opt.init.init_gain)  # 初始化生成网络
        self.net_b = init_net(self.net_b, init_type=opt.init.init_type, init_gain=opt.init.init_gain)

        # 损失函数
        # //? 损失 = 一致性损失 + 重建损失 + 循环一致性损失 + 感知损失
        if self.loss_type == 'l1':
            self.loss = nn.L1Loss()
        elif self.loss_type == 'l2':
            self.loss = nn.MSELoss()
        else:
            raise NotImplementedError
        if self.use_perceptual:
            self.perceptual_loss = VGGLoss_3D(device=self.device, multi_gpu=self.multi_gpu)

        # Optimizer + Scheduler
        self.optimizer_f = torch.optim.Adam(self.net_f.parameters(), lr=opt.optim.optimizer.params.lr_f, weight_decay=1e-4, betas = (0.1, 0.999))
        self.optimizer_b = torch.optim.Adam(self.net_b.parameters(), lr=opt.optim.optimizer.params.lr_b, weight_decay=1e-4, betas = (0.1, 0.999))
        self.schduler_f = torch.optim.lr_scheduler.StepLR(self.optimizer_f, step_size=400, gamma=0.9)
        self.schduler_b = torch.optim.lr_scheduler.StepLR(self.optimizer_b, step_size=400, gamma=0.9)
        
        self.scaler = GradScaler()  # mixed precision
        
    def set_input(self, data):
        data_mapping = {
            'T1_data': data[0],
            'T1_mask': data[1],
            'T1_tumor': data[2],
            'T1C_data': data[3],
            'T1C_mask': data[4],
            'T1C_tumor': data[5],
            'T2_data': data[6],
            'T2_mask': data[7],
            'T2_tumor': data[8],
        }

        # 获取源数据和目标数据，并从 data_mapping 中移除
        x_T = data_mapping.pop(self.sour_name).to(self.device)
        x_0 = data_mapping.pop(self.targ_name).to(self.device)
        # 提取病人 ID
        patient_id = data[-2][0]
        batch_size = x_T.shape[0]  # 获取 batch_size

        # 获取文本数据
        txt = (data[-1]).to(self.device) if self.is_text else None
        txt_con_mask = None
        
        # 根据 self.img_con_num 动态获取条件图像
        if self.sour_name == 'T1_data' and self.targ_name == 'T1C_data':
            img_con1 = torch.cat([data_mapping['T1_mask'], data_mapping['T1_tumor'], data_mapping['T2_data']], dim=1)
            img_con2 = torch.cat([data_mapping['T1C_mask'], data_mapping['T1C_tumor'], data_mapping['T2_data']], dim=1)
        elif self.sour_name == 'T1_data' and self.targ_name == 'T2_data':
            img_con1 = torch.cat([data_mapping['T1_mask'], data_mapping['T1_tumor'], data_mapping['T1C_data']], dim=1)
            img_con2 = torch.cat([data_mapping['T2_mask'], data_mapping['T2_tumor'], data_mapping['T1C_data']], dim=1)
        elif self.sour_name == 'T1C_data' and self.targ_name == 'T2_data':
            img_con1 = torch.cat([data_mapping['T1C_mask'], data_mapping['T1C_tumor'], data_mapping['T1_data']], dim=1)
            img_con2 = torch.cat([data_mapping['T2_mask'], data_mapping['T2_tumor'], data_mapping['T1_data']], dim=1)
        else:
            raise NotImplementedError

        # 初始化 img_con_mask: [batch_size, img_con_num]
        img_con_mask = torch.ones(batch_size, self.img_con_num, device=self.device)  # 全 1 表示没有缺失
        # 随机设置每个条件图像是否缺失
        for i in range(batch_size):
            for j in range(self.img_con_num):
                if random.random() < 0.2:  # 20% 概率缺失
                    img_con_mask[i, j] = 0  # 第 j 个条件图像缺失
        
        # 合并条件图像
        if not self.is_text:
            txt = None
        if not self.is_img:
            img_con1 = None
            img_con2 = None
        # 返回原始数据方便验证
        data_true_save = {
            'T1_data': data[0],
            'T1C_data': data[3],
            'T1C_mask': data[4],
            'T2_data': data[6],
            'T2_mask': data[7],
        }
        return x_0, x_T, txt, txt_con_mask, img_con1, img_con2, img_con_mask, patient_id, data_true_save
    
    def register_schedule(self):
        T = self.num_timesteps

        if self.mt_type == "linear":
            m_min, m_max = 0.001, 0.999
            m_t = np.linspace(m_min, m_max, T)
        elif self.mt_type == "sin":
            m_t = 1.0075 ** np.linspace(0, T, T)
            m_t = m_t / m_t[-1]
            m_t[-1] = 0.999
        else:
            raise NotImplementedError
        m_tminus = np.append(0, m_t[:-1])

        variance_t = 2. * (m_t - m_t ** 2) * self.max_var
        variance_tminus = np.append(0., variance_t[:-1])
        variance_t_tminus = variance_t - variance_tminus * ((1. - m_t) / (1. - m_tminus)) ** 2
        posterior_variance_t = variance_t_tminus * variance_tminus / variance_t

        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('m_t', to_torch(m_t))
        self.register_buffer('m_tminus', to_torch(m_tminus))
        self.register_buffer('variance_t', to_torch(variance_t))
        self.register_buffer('variance_tminus', to_torch(variance_tminus))
        self.register_buffer('variance_t_tminus', to_torch(variance_t_tminus))
        self.register_buffer('posterior_variance_t', to_torch(posterior_variance_t))

        if self.skip_sample:
            if self.sample_type == 'linear':
                midsteps = torch.arange(self.num_timesteps - 1, 1,
                                        step=-((self.num_timesteps - 1) / (self.sample_step - 2))).long()
                self.steps = torch.cat((midsteps, torch.Tensor([1, 0]).long()), dim=0)
            elif self.sample_type == 'cosine':
                steps = np.linspace(start=0, stop=self.num_timesteps, num=self.sample_step + 1)
                steps = (np.cos(steps / self.num_timesteps * np.pi) + 1.) / 2. * self.num_timesteps
                self.steps = torch.from_numpy(steps)
        else:
            self.steps = torch.arange(self.num_timesteps-1, -1, -1)
            
    def predict_x_from_objective(self, x_t, sour_data, t, objective_recon):
        "根据目标函数实现：x_t -> x_0"
        if self.objective == 'grad':
            x0_recon = x_t - objective_recon
        elif self.objective == 'noise':
            m_t = extract(self.m_t, t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            sigma_t = torch.sqrt(var_t)  # 计算标准差
            x0_recon = (x_t - m_t * sour_data - sigma_t * objective_recon) / (1. - m_t)  # 根据前向公式实现从x_t到x_0的逆过程
        elif self.objective == 'ysubx':
            x0_recon = sour_data - objective_recon
        else:
            raise NotImplementedError
        return x0_recon

    def q_sample(self, sour_data, targ_data, t, noise=None):
        "前向过程， 从 x_0和y 到 x_t和训练目标 (objective)"
        # Step 1: 如果 noise 为空，生成与 x 形状相同的随机噪声
        noise = default(noise, lambda: torch.randn_like(sour_data))
        # Step 2: 从 self.m_t 和 self.variance_t 中提取与时间步 t 对应的值，并调整形状
        m_t = extract(self.m_t, t, sour_data.shape)
        var_t = extract(self.variance_t, t, sour_data.shape)
        sigma_t = torch.sqrt(var_t)  # 计算标准差
        # Step 3: 返回结果 : x_t = (1-m_t)*x_0 + m_t*y_t + sigma_t*noise, 目标 (objective)
        x_t = (1. - m_t) * targ_data + m_t * sour_data + sigma_t * noise
        return x_t
    
    def forward(self, x_0, x_T, text_con=None, img_con1=None, img_con2=None, iters=0):
        """
        前向传播函数，计算 net_f 和 net_b 的输出。
        :param x_0: 原始图像
        :param x_T: 目标图像
        :param text_con: 文本特征
        :param img_con1: 条件图像特征1
        :param img_con2: 条件图像特征2
        :return: net_f 和 net_b 的输出
        """
        t = torch.randint(0, self.num_timesteps - 1, (x_0.shape[0],), device=self.device)
        T = torch.full_like(t, self.num_timesteps - 1)

        x_t = self.q_sample(x_0, x_T, t)
        # F direction  x_T -> x_0
        self.net_f.train()
        self.optimizer_f.zero_grad()
        if iters > 50:
            with torch.no_grad():
                x_T_bar = self.net_b(x_0, timesteps=T, text_feat=text_con, image_feat=img_con1)
                # x_T_bar = self.net_b(x_0, T)
            f3 = self.net_f(x_T_bar, timesteps=T, text_feat=text_con, image_feat=img_con2)
            # f3 = self.net_f(x_T_bar, T)
            loss_cyc = self.loss(f3, x_0)
        else:
            # 其它迭代，force cycle loss to zero
            loss_cyc = torch.tensor(0.0, device=self.device)
        f1 = self.net_f(x_t,  timesteps=t, text_feat=text_con, image_feat=img_con2)
        f2 = self.net_f(x_T,   timesteps=T, text_feat=text_con, image_feat=img_con2)
        # f1 = self.net_f(x_t,  t)
        # f2 = self.net_f(x_T, T)

        loss_rec = (self.loss(f1, x_0) + self.loss(f2, x_0)) * 0.5
        loss_con = self.loss(f1, f2)
        loss_f_dict = {
            'f_rec': loss_rec.item(),
            'f_con': loss_con.item(),
            'f_cyc': loss_cyc.item()
        }
        loss_f = loss_rec * self.lambda_rec + loss_con * self.lambda_con + loss_cyc * self.lambda_cyc
        if self.use_perceptual:
            loss_per = (self.perceptual_loss(f1, x_0) + self.perceptual_loss(f2, x_0)) * 0.5
            loss_f_dict['f_per'] = loss_per.item()
            loss_f += loss_per * self.lambda_per
        loss_f_dict['loss_f'] = loss_f.item()
        loss_f.backward()
        self.optimizer_f.step()
        self.schduler_f.step()

        x_t = self.q_sample(x_T, x_0, t)
        # B direction  x_0 -> x_T
        self.net_b.train()
        self.optimizer_b.zero_grad()
        if iters > 50:
            with torch.no_grad():
                x_0_bar = self.net_f(x_T, timesteps=T, text_feat=text_con, image_feat=img_con2)
                # x_0_bar = self.net_f(x_T, T)
            b3 = self.net_b(x_0_bar, timesteps=T, text_feat=text_con, image_feat=img_con1)
            # b3 = self.net_b(x_0_bar, T)
            loss_cyc = self.loss(b3, x_T)
        else:
            # 其它迭代，force cycle loss to zero
            loss_cyc = torch.tensor(0.0, device=self.device)
        b1 = self.net_b(x_t, timesteps=t, text_feat=text_con, image_feat=img_con1)
        b2 = self.net_b(x_0, timesteps=T, text_feat=text_con, image_feat=img_con1)
        # b1 = self.net_b(x_t, t)
        # b2 = self.net_b(x_0, T)

        loss_rec = (self.loss(b1, x_T) + self.loss(b2, x_T)) * 0.5
        loss_con = self.loss(b1, b2)
        
        loss_b_dict = {
            'b_rec': loss_rec.item(),
            'b_con': loss_con.item(),
            'b_cyc': loss_cyc.item()
        }
        loss_b = (loss_rec * self.lambda_rec + loss_con * self.lambda_con + loss_cyc * self.lambda_cyc)
        if self.use_perceptual:
            loss_per = (self.perceptual_loss(b1, x_T) + self.perceptual_loss(b2, x_T)) * 0.5
            loss_b_dict['b_per'] = loss_per.item()
            loss_b += loss_per * self.lambda_per
        loss_b_dict['loss_b'] = loss_b.item()
        loss_b.backward()
        self.optimizer_b.step()
        self.schduler_b.step()

        # g_f = grad_norm(self.net_f); g_b = grad_norm(self.net_b)
        # print(f"||∇f||={g_f:.3f}, ||∇b||={g_b:.3f}")
        loss = loss_f + loss_b
        loss_dict = {'loss': loss.item(), **loss_f_dict, **loss_b_dict}
        
        return loss, loss_dict

    def p_sample_loop(self, sour_data, txt_con, img_con):

        T = torch.full((sour_data.shape[0],), self.steps[0], device=self.device).long()
        pred_targ = self.net(sour_data, T, txt_con, img_con)

        return pred_targ
    
    def val_batch(self, sour_data, mode,
                txt_con: torch.Tensor = None, 
                img_con: torch.Tensor = None):
        """
        对原始图像按 self.patch_size 和 self.overlap 裁剪为目标大小的块，
        并送入网络得到目标图像块。

        Args:
            sour_data (torch.Tensor): 输入源数据，形状为 [B, C, D, H, W]。
            mode (str): 模型模式 ('f' 或 'b')。
            txt_con (torch.Tensor): 文本条件 (可选)。
            img_con (torch.Tensor): 图像条件 (可选)。

        Returns:
            pred_targ (torch.Tensor): 验证集预测结果，与输入图像形状一致。
        """
        # 设置模型模式
        if mode == 'f':
            self.net = self.net_f
        elif mode == 'b':
            self.net = self.net_b
        else:
            raise ValueError('mode should be "f" or "b"')

        self.net.eval()  # 设置模型为评估模式

        # 初始化预测结果张量和权重张量
        pred_targ = torch.zeros_like(sour_data, device=sour_data.device)  # 预测结果张量
        weight = torch.zeros_like(sour_data, device=sour_data.device)  # 用于记录每个像素的加权次数

        # 获取输入图像的维度
        _, c, depth, height, width = sour_data.shape
        patch_d, patch_h, patch_w = self.patch_size  # 获取块大小
        stride_d = int(patch_d * (1 - self.val_overlap))  # 深度方向步长
        stride_h = int(patch_h * (1 - self.val_overlap))  # 高度方向步长
        stride_w = int(patch_w * (1 - self.val_overlap))  # 宽度方向步长

        # 初始化批次大小
        if self.multi_gpu:
            batch_size = 8
        else:
            batch_size = self.batch_size

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
            xT_batch = torch.zeros((actual_batch_size, c, patch_d, patch_h, patch_w), device=sour_data.device)
            img_con_batch = None
            if self.is_img:
                img_con_batch = torch.zeros((actual_batch_size, self.img_con_num, patch_d, patch_h, patch_w), device=sour_data.device)

            # 填充当前批次的输入
            for j, (d_start, h_start, w_start) in enumerate(batch_indices):
                xT_batch[j] = sour_data[:, :, d_start:d_start + patch_d, h_start:h_start + patch_h, w_start:w_start + patch_w]
                if self.is_img:
                    img_con_batch[j] = img_con[:, :, d_start:d_start + patch_d, h_start:h_start + patch_h, w_start:w_start + patch_w]


            # 推理当前批次
            with torch.no_grad():
                pred_targ_batch = self.p_sample_loop(xT_batch, txt_con, img_con_batch)

            # 将生成结果拼接回预测张量
            for j, (d_start, h_start, w_start) in enumerate(batch_indices):
                pred_targ[:, :, d_start:d_start + patch_d, h_start:h_start + patch_h, w_start:w_start + patch_w] += pred_targ_batch[j]
                weight[:, :, d_start:d_start + patch_d, h_start:h_start + patch_h, w_start:w_start + patch_w] += 1

        # 对重叠区域进行归一化
        pred_targ /= weight

        return pred_targ
    
    def val(self, x_0, x_T,
            txt_con: torch.Tensor = None, 
            img_con1: torch.Tensor = None,
            img_con2: torch.Tensor = None):
        "x->y"
        self.net_f.eval()
        self.net_b.eval()
        with torch.no_grad():
            pred_x0 = self.val_batch(x_T, 'f', txt_con, img_con2)
            pred_xT = self.val_batch(x_0, 'b', txt_con, img_con1)

            pred_x0  = pred_x0.cpu().squeeze(0).squeeze(0).numpy()
            pred_xT  = pred_xT.cpu().squeeze(0).squeeze(0).numpy()
            x0  = x_0.cpu().squeeze(0).squeeze(0).numpy()
            xT  = x_T.cpu().squeeze(0).squeeze(0).numpy()

            data_range_x0 = max(pred_x0.max()-pred_x0.min(), x0.max()-x0.min())
            mae_x0 = np.mean(np.abs(pred_x0 - x0))
            ssim_x0 = ssim(pred_x0, x0, data_range=data_range_x0)
            psnr_x0 = psnr(pred_x0, x0, data_range=data_range_x0)

            data_range_xT = max(pred_xT.max()-pred_xT.min(), xT.max()-xT.min())
            mae_xT = np.mean(np.abs(pred_xT - xT))
            ssim_xT = ssim(pred_xT, xT, data_range=data_range_xT)
            psnr_xT = psnr(pred_xT, xT, data_range=data_range_xT)

        return x0, xT, pred_x0, pred_xT, mae_x0, ssim_x0, psnr_x0, mae_xT, ssim_xT, psnr_xT
        
    def plot(self, x0, xT, pred_x0, pred_xT, data_mapping, path):
        # 保存为nii
        for key, value in data_mapping.items():
            value = value.cpu().squeeze(0).squeeze(0).numpy()
            self.save_nii(value, os.path.join(path, f"{key}.nii.gz"))
        self.save_nii(pred_xT, os.path.join(path, f"pred_{self.sour_name}.nii.gz"))
        self.save_nii(pred_x0, os.path.join(path, f"pred_{self.targ_name}.nii.gz"))

        # 保存为png
        nonzero_slices = []
        for slice in range(pred_x0.shape[0]):
            if x0[slice, :, :].sum() > 0:
                nonzero_slices.append(slice)
        if len(nonzero_slices) != 0:
            slice = random.randint(0, len(nonzero_slices)-1)
            slice = nonzero_slices[slice]
            self.save_png(self.norm_png(x0[slice, :, :]), os.path.join(path, f"{self.sour_name}_{slice}.png"))
            self.save_png(self.norm_png(xT[slice, :, :]), os.path.join(path, f"{self.targ_name}_{slice}.png"))
            self.save_png(self.norm_png(pred_xT[slice, :, :]), os.path.join(path, f"pred_{self.sour_name}_{slice}.png"))
            self.save_png(self.norm_png(pred_x0[slice, :, :]), os.path.join(path, f"pred_{self.targ_name}_{slice}.png"))

    def plot_mid_sample(self, sour_data, targ_data, pred_x_mid_sample, path):
        # 保存为png
        path = os.path.join(path, f"{self.sour_name}_to_{self.targ_name}.png")
        nonzero_slices = []
        for slice in range(sour_data.shape[0]):
            if targ_data[slice, :, :].sum() > 0:
                nonzero_slices.append(slice)
        if len(nonzero_slices) != 0:
            slice = random.randint(0, len(nonzero_slices)-1)
            slice = nonzero_slices[slice]
            img1 = self.norm_png(sour_data[slice, :, :])
            img2 = self.norm_png(targ_data[slice, :, :])
            img3 = self.norm_png(pred_x_mid_sample[0][slice, :, :])
            img4 = self.norm_png(pred_x_mid_sample[1][slice, :, :])
            img5 = self.norm_png(pred_x_mid_sample[2][slice, :, :])
            img6 = self.norm_png(pred_x_mid_sample[3][slice, :, :])
            self.save_combined_image([img1, img2, img3, img4, img5, img6], path, grid_size=(1, 6))

    def save_combined_image(self, img_list, output_path, grid_size=(1, 6)):
        # 检查输入
        assert len(img_list) == grid_size[0] * grid_size[1], "图片数量与网格大小不匹配！"
        # 获取单张图片的高度和宽度
        height, width = img_list[0].shape
        # 创建空白画布，用于拼接
        grid_height = height * grid_size[0]  # 总高度
        grid_width = width * grid_size[1]   # 总宽度
        combined_image = np.zeros((grid_height, grid_width), dtype=np.uint8)  # 假设图片是灰度图
        # 将每张图片放到画布上的对应位置
        for idx, img in enumerate(img_list):
            row, col = divmod(idx, grid_size[1])  # 计算图片的行、列位置
            y_start, y_end = row * height, (row + 1) * height
            x_start, x_end = col * width, (col + 1) * width
            combined_image[y_start:y_end, x_start:x_end] = img
        # 将拼接后的 NumPy 数组转换为图片并保存
        final_image = Image.fromarray(combined_image)
        final_image.save(output_path)

    def norm_png(self, data):
        data = (data - data.min()) / (data.max() - data.min())
        return data

    def save_png(self, data, path):
        data = (data*255).astype(np.uint8)
        data = Image.fromarray(data)
        data.save(path)

    def save_nii(self, data, path):
        data = np.transpose(data, (2, 1, 0))
        data = np.rot90(data, k=-2, axes=(1, 2))  # 顺时针旋转 180 度
        voxel_spacing = [1, 1, 6]  # z, x, y
        affine = np.diag(voxel_spacing + [1])
        nii_x = nib.Nifti1Image(data, affine)
        nib.save(nii_x, path)


# class CPC_BM(nn.Module):
#     def __init__(self, opt):
#         super(CPC_BM, self).__init__()
#         self.is_train = opt.train.is_train
#         self.device = opt.train.device
#         self.multi_gpu = opt.train.multi_gpu  # 是否使用多 GPU
#         self.batch_size = opt.train.batch_size
#         self.accumulate_grad_batches = opt.train.accumulate_grad_batches  # 梯度累积步数

#         self.img_con_num = opt.net.cond_channels  # 条件图像数量
#         self.patch_size = opt.data.patch_image_shape  # 图像分块大小
#         self.val_overlap = opt.val.overlap  # 重叠度
#         self.sour_name = opt.data.sour_img_name  # 源域数据名称
#         self.targ_name = opt.data.targ_img_name  # 目标域数据名称

#         self.is_text = opt.train.is_text  # 是否采用文本特征
#         self.is_img = opt.train.is_image# 是否采用图像特征

#         self.loss_type = opt.loss.loss_type
#         self.lambda_con = opt.loss.lambda_con
#         self.lambda_rec = opt.loss.lambda_rec
#         self.lambda_cyc = opt.loss.lambda_cyc
#         self.lambda_per = opt.loss.lambda_per
#         self.b_scale = opt.loss.b_scale
#         self.use_perceptual = opt.loss.is_perceptual

#         self.condition_key = opt.ddbm.condition_key  # 是否采用条件引导
#         self.num_timesteps = opt.ddbm.num_timesteps  # 时序步数
#         self.mt_type = opt.ddbm.mt_type  # 定义 mt 类型
#         self.max_var = opt.ddbm.max_var  # 最大方差
#         self.skip_sample = opt.ddbm.skip_sample  # 是否跳过采样
#         self.sample_type = opt.ddbm.sample_type  # 采样类型
#         self.sample_step = opt.ddbm.sample_step  # 采样步数
#         self.eta = opt.ddbm.eta

#         self.register_schedule()

#         # Generator Network (UNet)
#         self.net_f = creat_net(device=self.device, **(vars(opt.net)))
#         self.net_b = creat_net(device=self.device, **(vars(opt.net)))

#         # Initialising the network
#         self.net_f = init_net(self.net_f, init_type=opt.init.init_type, init_gain=opt.init.init_gain)  # 初始化生成网络
#         self.net_b = init_net(self.net_b, init_type=opt.init.init_type, init_gain=opt.init.init_gain)

#         # 损失函数
#         # //? 损失 = 一致性损失 + 重建损失 + 循环一致性损失 + 感知损失
#         if self.loss_type == 'l1':
#             self.loss = nn.L1Loss()
#         elif self.loss_type == 'l2':
#             self.loss = nn.MSELoss()
#         else:
#             raise NotImplementedError
#         if self.use_perceptual:
#             self.perceptual_loss = VGGLoss_3D(device=self.device, multi_gpu=self.multi_gpu)

#         # Optimizer + Scheduler
#         self.optimizer_f = torch.optim.Adam(self.net_f.parameters(), lr=opt.optim.optimizer.params.lr_f, weight_decay=1e-4, betas = (0.5, 0.999))
#         self.optimizer_b = torch.optim.Adam(self.net_b.parameters(), lr=opt.optim.optimizer.params.lr_b, weight_decay=1e-4, betas = (0.5, 0.999))
        
#         self.scaler = GradScaler()  # mixed precision
        
#     def set_input(self, data):
#         data_mapping = {
#             'T1_data': data[0],
#             'T1_mask': data[1],
#             'T1_tumor': data[2],
#             'T1C_data': data[3],
#             'T1C_mask': data[4],
#             'T1C_tumor': data[5],
#             'T2_data': data[6],
#             'T2_mask': data[7],
#             'T2_tumor': data[8],
#         }

#         # 获取源数据和目标数据，并从 data_mapping 中移除
#         x_T = data_mapping.pop(self.sour_name).to(self.device)
#         x_0 = data_mapping.pop(self.targ_name).to(self.device)
#         # 提取病人 ID
#         patient_id = data[-2][0]
#         batch_size = x_T.shape[0]  # 获取 batch_size

#         # 获取文本数据
#         txt = (data[-1]).to(self.device) if self.is_text else None
#         txt_con_mask = None
        
#         # 根据 self.img_con_num 动态获取条件图像
#         if self.sour_name == 'T1_data' and self.targ_name == 'T1C_data':
#             img_con1 = torch.cat([data_mapping['T1_mask'], data_mapping['T1_tumor'], data_mapping['T2_data']], dim=1)
#             img_con2 = torch.cat([data_mapping['T1C_mask'], data_mapping['T1C_tumor'], data_mapping['T2_data']], dim=1)
#         elif self.sour_name == 'T1_data' and self.targ_name == 'T2_data':
#             img_con1 = torch.cat([data_mapping['T1_mask'], data_mapping['T1_tumor'], data_mapping['T1C_data']], dim=1)
#             img_con2 = torch.cat([data_mapping['T2_mask'], data_mapping['T2_tumor'], data_mapping['T1C_data']], dim=1)
#         elif self.sour_name == 'T1C_data' and self.targ_name == 'T2_data':
#             img_con1 = torch.cat([data_mapping['T1C_mask'], data_mapping['T1C_tumor'], data_mapping['T1_data']], dim=1)
#             img_con2 = torch.cat([data_mapping['T2_mask'], data_mapping['T2_tumor'], data_mapping['T1_data']], dim=1)
#         else:
#             raise NotImplementedError

#         # 初始化 img_con_mask: [batch_size, img_con_num]
#         img_con_mask = torch.ones(batch_size, self.img_con_num, device=self.device)  # 全 1 表示没有缺失
#         # 随机设置每个条件图像是否缺失
#         for i in range(batch_size):
#             for j in range(self.img_con_num):
#                 if random.random() < 0.2:  # 20% 概率缺失
#                     img_con_mask[i, j] = 0  # 第 j 个条件图像缺失
        
#         # 合并条件图像
#         if not self.is_text:
#             txt = None
#         if not self.is_img:
#             img_con1 = None
#             img_con2 = None
#         # 返回原始数据方便验证
#         data_true_save = {
#             'T1_data': data[0],
#             'T1C_data': data[3],
#             'T1C_mask': data[4],
#             'T2_data': data[6],
#             'T2_mask': data[7],
#         }
#         return x_0, x_T, txt, txt_con_mask, img_con1, img_con2, img_con_mask, patient_id, data_true_save
    
#     def register_schedule(self):
#         T = self.num_timesteps

#         if self.mt_type == "linear":
#             m_min, m_max = 0.001, 0.999
#             m_t = np.linspace(m_min, m_max, T)
#         elif self.mt_type == "sin":
#             m_t = 1.0075 ** np.linspace(0, T, T)
#             m_t = m_t / m_t[-1]
#             m_t[-1] = 0.999
#         else:
#             raise NotImplementedError
#         m_tminus = np.append(0, m_t[:-1])

#         variance_t = 2. * (m_t - m_t ** 2) * self.max_var
#         variance_tminus = np.append(0., variance_t[:-1])
#         variance_t_tminus = variance_t - variance_tminus * ((1. - m_t) / (1. - m_tminus)) ** 2
#         posterior_variance_t = variance_t_tminus * variance_tminus / variance_t

#         to_torch = partial(torch.tensor, dtype=torch.float32)
#         self.register_buffer('m_t', to_torch(m_t))
#         self.register_buffer('m_tminus', to_torch(m_tminus))
#         self.register_buffer('variance_t', to_torch(variance_t))
#         self.register_buffer('variance_tminus', to_torch(variance_tminus))
#         self.register_buffer('variance_t_tminus', to_torch(variance_t_tminus))
#         self.register_buffer('posterior_variance_t', to_torch(posterior_variance_t))

#         if self.skip_sample:
#             if self.sample_type == 'linear':
#                 midsteps = torch.arange(self.num_timesteps - 1, 1,
#                                         step=-((self.num_timesteps - 1) / (self.sample_step - 2))).long()
#                 self.steps = torch.cat((midsteps, torch.Tensor([1, 0]).long()), dim=0)
#             elif self.sample_type == 'cosine':
#                 steps = np.linspace(start=0, stop=self.num_timesteps, num=self.sample_step + 1)
#                 steps = (np.cos(steps / self.num_timesteps * np.pi) + 1.) / 2. * self.num_timesteps
#                 self.steps = torch.from_numpy(steps)
#         else:
#             self.steps = torch.arange(self.num_timesteps-1, -1, -1)
            
#     def predict_x_from_objective(self, x_t, sour_data, t, objective_recon):
#         "根据目标函数实现：x_t -> x_0"
#         if self.objective == 'grad':
#             x0_recon = x_t - objective_recon
#         elif self.objective == 'noise':
#             m_t = extract(self.m_t, t, x_t.shape)
#             var_t = extract(self.variance_t, t, x_t.shape)
#             sigma_t = torch.sqrt(var_t)  # 计算标准差
#             x0_recon = (x_t - m_t * sour_data - sigma_t * objective_recon) / (1. - m_t)  # 根据前向公式实现从x_t到x_0的逆过程
#         elif self.objective == 'ysubx':
#             x0_recon = sour_data - objective_recon
#         else:
#             raise NotImplementedError
#         return x0_recon

#     def q_sample(self, sour_data, targ_data, t, noise=None):
#         "前向过程， 从 x_0和y 到 x_t和训练目标 (objective)"
#         # Step 1: 如果 noise 为空，生成与 x 形状相同的随机噪声
#         noise = default(noise, lambda: torch.randn_like(sour_data))
#         # Step 2: 从 self.m_t 和 self.variance_t 中提取与时间步 t 对应的值，并调整形状
#         m_t = extract(self.m_t, t, sour_data.shape)
#         var_t = extract(self.variance_t, t, sour_data.shape)
#         sigma_t = torch.sqrt(var_t)  # 计算标准差
#         # Step 3: 返回结果 : x_t = (1-m_t)*x_0 + m_t*y_t + sigma_t*noise, 目标 (objective)
#         x_t = (1. - m_t) * targ_data + m_t * sour_data + sigma_t * noise
#         return x_t
    
#     def forward(self, x_0, x_T, text_con=None, img_con1=None, img_con2=None, global_step=0):
#         """
#         前向传播函数，计算 net_f 和 net_b 的输出。
#         :param x_0: 原始图像
#         :param x_T: 目标图像
#         :param text_con: 文本特征
#         :param img_con1: 条件图像特征1
#         :param img_con2: 条件图像特征2
#         :return: net_f 和 net_b 的输出
#         """
#         t = torch.randint(0, self.num_timesteps - 1, (x_0.shape[0],), device=self.device)
#         T = torch.full_like(t, self.num_timesteps - 1)

#         x_t = self.q_sample(x_0, x_T, t)
#         # F direction  x_T -> x_0
#         self.optimizer_f.zero_grad()
#         with torch.no_grad():
#             x_T_bar = self.net_b(x_0, timesteps=T, text_feat=text_con, cond_image=img_con1)
#         self.net_f.train()
#         f1 = self.net_f(x_t,  timesteps=t, text_feat=text_con, cond_image=img_con2)
#         f2 = self.net_f(x_T,   timesteps=T, text_feat=text_con, cond_image=img_con2)
#         f3 = self.net_f(x_T_bar, timesteps=T, text_feat=text_con, cond_image=img_con2)

#         loss_rec = (self.loss(f1, x_0) + self.loss(f2, x_0)) * 0.5
#         loss_con = self.loss(f1, f2)
#         loss_cyc = self.loss(f3, x_0)
#         loss_f_dict = {
#             'f_rec': loss_rec.item(),
#             'f_con': loss_con.item(),
#             'f_cyc': loss_cyc.item()
#         }
#         loss_f = loss_rec * self.lambda_rec + loss_con * self.lambda_con + loss_cyc * self.lambda_cyc
#         if self.use_perceptual:
#             loss_per = (self.perceptual_loss(f1, x_0) + self.perceptual_loss(f2, x_0) + self.perceptual_loss(f3, x_0)) * 0.3
#             loss_f_dict['f_per'] = loss_per.item()
#             loss_f += loss_per * self.lambda_per
#         loss_f_dict['loss_f'] = loss_f.item()
#         loss_f.backward()
#         self.optimizer_f.step()

#         x_t = self.q_sample(x_T, x_0, t)
#         # B direction  x_0 -> x_T
#         self.optimizer_b.zero_grad()
#         with torch.no_grad():
#             x_0_bar = self.net_f(x_T, timesteps=T, text_feat=text_con, cond_image=img_con2)
#         self.net_b.train()
#         b1 = self.net_b(x_t, timesteps=t, text_feat=text_con, cond_image=img_con1)
#         b2 = self.net_b(x_0, timesteps=T, text_feat=text_con, cond_image=img_con1)
#         b3 = self.net_b(x_0_bar, timesteps=T, text_feat=text_con, cond_image=img_con1)

#         loss_rec = (self.loss(b1, x_T) + self.loss(b2, x_T)) * 0.5
#         loss_con = self.loss(b1, b2)
#         loss_cyc = self.loss(b3, x_T)
#         loss_b_dict = {
#             'b_rec': loss_rec.item(),
#             'b_con': loss_con.item(),
#             'b_cyc': loss_cyc.item()
#         }
#         loss_b = (loss_rec * self.lambda_rec + loss_con * self.lambda_con + loss_cyc * self.lambda_cyc)
#         if self.use_perceptual:
#             loss_per = (self.perceptual_loss(b1, x_T) + self.perceptual_loss(b2, x_T) + self.perceptual_loss(b3, x_T)) * 0.3
#             loss_b_dict['b_per'] = loss_per.item()
#             loss_b += loss_per * self.lambda_per
#         loss_b_dict['loss_b'] = loss_b.item()
#         loss_b.backward()
#         self.optimizer_b.step()

#         # g_f = grad_norm(self.net_f); g_b = grad_norm(self.net_b)
#         # print(f"||∇f||={g_f:.3f}, ||∇b||={g_b:.3f}")
#         loss = loss_f + loss_b
#         loss_dict = {'loss': loss.item(), **loss_f_dict, **loss_b_dict}
        
#         return loss, loss_dict

#     def p_sample_loop(self, sour_data, txt_con, img_con):

#         T = torch.full((sour_data.shape[0],), self.steps[0], device=self.device).long()
#         pred_targ = self.net(sour_data, T, txt_con, img_con)

#         return pred_targ
    
#     def val_batch(self, sour_data, mode,
#                 txt_con: torch.Tensor = None, 
#                 img_con: torch.Tensor = None):
#         """
#         对原始图像按 self.patch_size 和 self.overlap 裁剪为目标大小的块，
#         并送入网络得到目标图像块。

#         Args:
#             sour_data (torch.Tensor): 输入源数据，形状为 [B, C, D, H, W]。
#             mode (str): 模型模式 ('f' 或 'b')。
#             txt_con (torch.Tensor): 文本条件 (可选)。
#             img_con (torch.Tensor): 图像条件 (可选)。

#         Returns:
#             pred_targ (torch.Tensor): 验证集预测结果，与输入图像形状一致。
#         """
#         # 设置模型模式
#         if mode == 'f':
#             self.net = self.net_f
#         elif mode == 'b':
#             self.net = self.net_b
#         else:
#             raise ValueError('mode should be "f" or "b"')

#         self.net.eval()  # 设置模型为评估模式

#         # 初始化预测结果张量和权重张量
#         pred_targ = torch.zeros_like(sour_data, device=sour_data.device)  # 预测结果张量
#         weight = torch.zeros_like(sour_data, device=sour_data.device)  # 用于记录每个像素的加权次数

#         # 获取输入图像的维度
#         _, c, depth, height, width = sour_data.shape
#         patch_d, patch_h, patch_w = self.patch_size  # 获取块大小
#         stride_d = int(patch_d * (1 - self.val_overlap))  # 深度方向步长
#         stride_h = int(patch_h * (1 - self.val_overlap))  # 高度方向步长
#         stride_w = int(patch_w * (1 - self.val_overlap))  # 宽度方向步长

#         # 初始化批次大小
#         if self.multi_gpu:
#             batch_size = 8
#         else:
#             batch_size = self.batch_size

#         # 生成裁剪块的起始索引
#         d_start_list = list(range(0, depth - patch_d + 1, stride_d))
#         h_start_list = list(range(0, height - patch_h + 1, stride_h))
#         w_start_list = list(range(0, width - patch_w + 1, stride_w))

#         # 如果最后一块没有覆盖到边缘，添加最后一块的起始位置
#         if depth - patch_d % stride_d != 0:
#             d_start_list.append(depth - patch_d)
#         if height - patch_h % stride_h != 0:
#             h_start_list.append(height - patch_h)
#         if width - patch_w % stride_w != 0:
#             w_start_list.append(width - patch_w)

#         # 生成所有块的索引组合
#         patch_indices = [(d, h, w) for d in d_start_list for h in h_start_list for w in w_start_list]

#         # 按批次处理裁剪块
#         total_patches = len(patch_indices)
#         batch_num = math.ceil(total_patches / batch_size)  # 总批次数

#         for i in range(batch_num):
#             # 获取当前批次的索引
#             batch_indices = patch_indices[i * batch_size: (i + 1) * batch_size]
#             actual_batch_size = len(batch_indices)

#             # 初始化当前批次张量
#             xT_batch = torch.zeros((actual_batch_size, c, patch_d, patch_h, patch_w), device=sour_data.device)
#             img_con_batch = None
#             if self.is_img:
#                 img_con_batch = torch.zeros((actual_batch_size, self.img_con_num, patch_d, patch_h, patch_w), device=sour_data.device)

#             # 填充当前批次的输入
#             for j, (d_start, h_start, w_start) in enumerate(batch_indices):
#                 xT_batch[j] = sour_data[:, :, d_start:d_start + patch_d, h_start:h_start + patch_h, w_start:w_start + patch_w]
#                 if self.is_img:
#                     img_con_batch[j] = img_con[:, :, d_start:d_start + patch_d, h_start:h_start + patch_h, w_start:w_start + patch_w]

#             # 提取文本特征
#             text_features = None
#             txt_con_batch = txt_con * actual_batch_size
#             if self.is_text:
#                 text_features = self.text_feature_extraction(txt_con_batch)

#             # 推理当前批次
#             with torch.no_grad():
#                 pred_targ_batch = self.p_sample_loop(xT_batch, text_features, img_con_batch)

#             # 将生成结果拼接回预测张量
#             for j, (d_start, h_start, w_start) in enumerate(batch_indices):
#                 pred_targ[:, :, d_start:d_start + patch_d, h_start:h_start + patch_h, w_start:w_start + patch_w] += pred_targ_batch[j]
#                 weight[:, :, d_start:d_start + patch_d, h_start:h_start + patch_h, w_start:w_start + patch_w] += 1

#         # 对重叠区域进行归一化
#         pred_targ /= weight

#         return pred_targ
    
#     def val(self, x_0, x_T,
#             txt_con: torch.Tensor = None, 
#             img_con1: torch.Tensor = None,
#             img_con2: torch.Tensor = None):
#         "x->y"
#         self.net_f.eval()
#         self.net_b.eval()
#         with torch.no_grad():
#             pred_x0 = self.val_batch(x_T, 'f', txt_con, img_con2)
#             pred_xT = self.val_batch(x_0, 'b', txt_con, img_con1)

#             pred_x0  = pred_x0.cpu().squeeze(0).squeeze(0).numpy()
#             pred_xT  = pred_xT.cpu().squeeze(0).squeeze(0).numpy()
#             x0  = x_0.cpu().squeeze(0).squeeze(0).numpy()
#             xT  = x_T.cpu().squeeze(0).squeeze(0).numpy()

#             data_range_x0 = max(pred_x0.max()-pred_x0.min(), x0.max()-x0.min())
#             mae_x0 = np.mean(np.abs(pred_x0 - x0))
#             ssim_x0 = ssim(pred_x0, x0, data_range=data_range_x0)
#             psnr_x0 = psnr(pred_x0, x0, data_range=data_range_x0)

#             data_range_xT = max(pred_xT.max()-pred_xT.min(), xT.max()-xT.min())
#             mae_xT = np.mean(np.abs(pred_xT - xT))
#             ssim_xT = ssim(pred_xT, xT, data_range=data_range_xT)
#             psnr_xT = psnr(pred_xT, xT, data_range=data_range_xT)

#         return x0, xT, pred_x0, pred_xT, mae_x0, ssim_x0, psnr_x0, mae_xT, ssim_xT, psnr_xT
        
#     def plot(self, x0, xT, pred_x0, pred_xT, data_mapping, path):
#         # 保存为nii
#         for key, value in data_mapping.items():
#             value = value.cpu().squeeze(0).squeeze(0).numpy()
#             self.save_nii(value, os.path.join(path, f"{key}.nii.gz"))
#         self.save_nii(pred_xT, os.path.join(path, f"pred_{self.sour_name}.nii.gz"))
#         self.save_nii(pred_x0, os.path.join(path, f"pred_{self.targ_name}.nii.gz"))

#         # 保存为png
#         nonzero_slices = []
#         for slice in range(pred_x0.shape[0]):
#             if x0[slice, :, :].sum() > 0:
#                 nonzero_slices.append(slice)
#         if len(nonzero_slices) != 0:
#             slice = random.randint(0, len(nonzero_slices)-1)
#             slice = nonzero_slices[slice]
#             self.save_png(self.norm_png(x0[slice, :, :]), os.path.join(path, f"{self.sour_name}_{slice}.png"))
#             self.save_png(self.norm_png(xT[slice, :, :]), os.path.join(path, f"{self.targ_name}_{slice}.png"))
#             self.save_png(self.norm_png(pred_xT[slice, :, :]), os.path.join(path, f"pred_{self.sour_name}_{slice}.png"))
#             self.save_png(self.norm_png(pred_x0[slice, :, :]), os.path.join(path, f"pred_{self.targ_name}_{slice}.png"))

#     def plot_mid_sample(self, sour_data, targ_data, pred_x_mid_sample, path):
#         # 保存为png
#         path = os.path.join(path, f"{self.sour_name}_to_{self.targ_name}.png")
#         nonzero_slices = []
#         for slice in range(sour_data.shape[0]):
#             if targ_data[slice, :, :].sum() > 0:
#                 nonzero_slices.append(slice)
#         if len(nonzero_slices) != 0:
#             slice = random.randint(0, len(nonzero_slices)-1)
#             slice = nonzero_slices[slice]
#             img1 = self.norm_png(sour_data[slice, :, :])
#             img2 = self.norm_png(targ_data[slice, :, :])
#             img3 = self.norm_png(pred_x_mid_sample[0][slice, :, :])
#             img4 = self.norm_png(pred_x_mid_sample[1][slice, :, :])
#             img5 = self.norm_png(pred_x_mid_sample[2][slice, :, :])
#             img6 = self.norm_png(pred_x_mid_sample[3][slice, :, :])
#             self.save_combined_image([img1, img2, img3, img4, img5, img6], path, grid_size=(1, 6))

#     def save_combined_image(self, img_list, output_path, grid_size=(1, 6)):
#         # 检查输入
#         assert len(img_list) == grid_size[0] * grid_size[1], "图片数量与网格大小不匹配！"
#         # 获取单张图片的高度和宽度
#         height, width = img_list[0].shape
#         # 创建空白画布，用于拼接
#         grid_height = height * grid_size[0]  # 总高度
#         grid_width = width * grid_size[1]   # 总宽度
#         combined_image = np.zeros((grid_height, grid_width), dtype=np.uint8)  # 假设图片是灰度图
#         # 将每张图片放到画布上的对应位置
#         for idx, img in enumerate(img_list):
#             row, col = divmod(idx, grid_size[1])  # 计算图片的行、列位置
#             y_start, y_end = row * height, (row + 1) * height
#             x_start, x_end = col * width, (col + 1) * width
#             combined_image[y_start:y_end, x_start:x_end] = img
#         # 将拼接后的 NumPy 数组转换为图片并保存
#         final_image = Image.fromarray(combined_image)
#         final_image.save(output_path)

#     def norm_png(self, data):
#         data = (data - data.min()) / (data.max() - data.min())
#         return data

#     def save_png(self, data, path):
#         data = (data*255).astype(np.uint8)
#         data = Image.fromarray(data)
#         data.save(path)

#     def save_nii(self, data, path):
#         data = np.transpose(data, (2, 1, 0))
#         data = np.rot90(data, k=-2, axes=(1, 2))  # 顺时针旋转 180 度
#         voxel_spacing = [1, 1, 6]  # z, x, y
#         affine = np.diag(voxel_spacing + [1])
#         nii_x = nib.Nifti1Image(data, affine)
#         nib.save(nii_x, path)
