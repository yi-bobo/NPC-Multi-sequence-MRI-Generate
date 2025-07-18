
import sys
sys.path.append('/data1/weiyibo/NPC-MRI/Code/Pctch_model/')

import os
import math
import torch
import random
import itertools
import numpy as np
from PIL import Image
import nibabel as nib
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from torch.nn import DataParallel
from monai.inferers import SlidingWindowInferer
from transformers import CLIPTokenizer, CLIPModel

from inspect import isfunction
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from Utils.loss.perceptual_loss import VGGLoss_3D
from .Net.Cycle_CPBM_Net.Cycle_CPBM_Net import DiffusionModelUNet
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
# endregion

# region 文本特征提取
class TextFeature(nn.Module):
    def __init__(self,
                 device,
                 is_text: bool = False,
                 latent_dim: int = 256,
                 text_dim: int = 512):
        """
        文本特征提取模型
        Args:
            device: 所使用的设备 (如 'cuda' 或 'cpu')。
            is_text: 是否启用文本特征提取。
            latent_dim: 投影后文本特征的维度。
            text_dim: CLIP文本编码器输出的特征维度。
        """
        super().__init__()
        self.device = device
        self.is_text = is_text

        # CLIP文本编码器
        self.clip_tokenizer = CLIPTokenizer.from_pretrained("/data1/weiyibo/NPC-MRI/Models/Pre_model/CLIP/",local_files_only=True)
        self.clip_model = CLIPModel.from_pretrained("/data1/weiyibo/NPC-MRI/Models/Pre_model/CLIP/",local_files_only=True)
        self.clip_projection = nn.Linear(text_dim, latent_dim)
        self.clip_model.to(device)
        self.text_dim = text_dim
        self.latent_dim = latent_dim

    def encode_text(self, text):
        """
        对文本信息进行编码，分别处理每种文本条件。
        动态生成起始标识符和结束标识符。
        Args:
            text: 输入的文本信息（字符串）。
        Returns:
            编码后的文本特征，形状为 [total_tokens_per_text, latent_dim]。
        """
        # 分割文本条件
        stages = text.split('.')
        stages = [s.strip() for s in stages if s.strip()]  # 去除空白项

        stage_tokens = []
        for i, stage in enumerate(stages):
            # 动态生成起始和结束标识符
            start_token = torch.randn(1, self.latent_dim, device=self.device)  # 随机初始化 [1, latent_dim]
            end_token = torch.randn(1, self.latent_dim, device=self.device)  # 随机初始化 [1, latent_dim]

            # 文本编码
            inputs = self.clip_tokenizer(stage, return_tensors="pt", truncation=True, padding=True)
            inputs = inputs.to(self.device)
            outputs = self.clip_model.get_text_features(**inputs)  # [1, text_dim]
            outputs = self.clip_projection(outputs)  # [1, latent_dim]

            # 拼接起始和结束标识符
            outputs = torch.cat([start_token, outputs, end_token], dim=0)
            stage_tokens.append(outputs)

        # 拼接所有阶段的编码结果
        output = torch.cat(stage_tokens, dim=0)  # [total_tokens_per_text, latent_dim]
        output = F.normalize(output, p=2, dim=-1)  # 特征归一化
        return output

    def forward(self, text_con=None):
        """
        前向传播，处理文本条件。
        Args:
            text_con: 文本条件列表，每个元素是一个文本字符串。
        Returns:
            文本特征张量，形状为 [B, total_tokens_per_text, latent_dim]，或 None（如果未启用文本特征）。
        """
        if self.is_text:
            text_features = []
            for i in range(len(text_con)):
                encoded_text = self.encode_text(text_con[i])  # [total_tokens_per_text, latent_dim]
                text_features.append(encoded_text)
            text_features = torch.stack(text_features, dim=0)  # [B, total_tokens_per_text, latent_dim]
        else:
            text_features = None
        return text_features
# endregion

# region CBM_model

class Cycle_CPBM_model(nn.Module):
    def __init__(self, opt):
        super(Cycle_CPBM_model, self).__init__()
        self.is_train = opt.train.is_train
        self.device = opt.train.device
        self.multi_gpu = opt.train.multi_gpu  # 是否使用多 GPU
        self.img_con_num = opt.net.con_img_channels  # 条件图像数量
        self.batch_size = opt.train.batch_size
        self.patch_size = opt.data.patch_image_shape  # 图像分块大小
        self.val_overlap = opt.val.overlap  # 重叠度
        self.sour_name = opt.data.sour_img_name  # 源域数据名称
        self.targ_name = opt.data.targ_img_name  # 目标域数据名称
        self.img_con_name = opt.data.img_con_name  # 图像条件名称
        self.is_text = opt.net.is_text  # 是否采用文本条件
        self.is_img = opt.net.is_img  # 是否采用图像条件

        self.is_perceptual = opt.loss.is_perceptual  # 是否采用感知损失
        self.loss_type = opt.loss.loss_type  # 损失类型
        self.objective = opt.loss.objective  # 
        self.lambda_con = opt.loss.lambda_con  # 条件损失权重
        self.lambda_rec = opt.loss.lambda_rec  # 重建损失权重
        self.lambda_cycle = opt.loss.lambda_cycle  # 循环损失权重
        self.lambda_perceptual = opt.loss.lambda_perceptual  # 图像损失权重

        self.condition_key = opt.ddbm.condition_key  # 是否采用条件引导
        self.num_timesteps = opt.ddbm.num_timesteps  # 时序步数
        self.mt_type = opt.ddbm.mt_type  # 定义 mt 类型
        self.max_var = opt.ddbm.max_var  # 最大方差
        self.skip_sample = opt.ddbm.skip_sample  # 是否跳过采样
        self.sample_type = opt.ddbm.sample_type  # 采样类型
        self.sample_step = opt.ddbm.sample_step  # 采样步数
        self.eta = opt.ddbm.eta

        self.register_schedule()

        self.infer = SlidingWindowInferer(roi_size=opt.data.patch_image_shape, sw_batch_size=opt.train.batch_size, overlap=opt.data.overlap)

        # 定义损失函数
        if self.loss_type == 'l1':
            self.loss = nn.L1Loss()
        elif self.loss_type == 'l2':
            self.loss = nn.MSELoss()
        else:
            raise NotImplementedError
        if self.is_perceptual:
            self.perceptual_loss = VGGLoss_3D(device=self.device, multi_gpu=self.multi_gpu)

        # Generator Network (UNet)
        self.net_f_single = creat_net(device=self.device, **(vars(opt.net)))
        self.net_b_single = creat_net(device=self.device, **(vars(opt.net)))

        # 文本特征提取器
        self.text_feature_extraction = TextFeature(device=self.device, is_text=self.is_text)

        # 保存原始模型的引用（用于验证）
        self.net_f = self.net_f_single
        self.net_b = self.net_b_single

        # 多 GPU 包装（仅用于训练）
        if self.multi_gpu:
            print(f"Using {torch.cuda.device_count()} GPUs for training.")
            self.net_f = DataParallel(self.net_f_single)
            self.net_b = DataParallel(self.net_b_single)

        # Optimizers and Learning Rate Schedulers
        scheduler_params = {
            "T_max": opt.train.max_epochs, "eta_min": 0, "last_epoch": -1
        }
        warmup_epochs = 100
        self.optimizer = create_optimizer(opt.optim.optimizer.name, itertools.chain(self.net_f.parameters(), self.net_b.parameters()), **vars(opt.optim.optimizer.params))
        self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=opt.optim.scheduler.params.multiplier, warm_epoch=warmup_epochs, 
                                                after_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **scheduler_params)
        ) 

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
        x_T = data_mapping.pop(self.sour_name).to(self.device)
        x_0 = data_mapping.pop(self.targ_name).to(self.device)
        # 提取病人 ID
        patient_id = data[-2][0]
        batch_size = x_T.shape[0]  # 获取 batch_size

        # 获取文本数据
        txt = data[-1]
        
        split_txt = txt[0].split('.')
        text_con_num = len(split_txt)
        txt_con_mask = torch.ones(batch_size, text_con_num, device=self.device)  # 全 1 表示没有缺失
        for i in range(batch_size):
            for j in range(text_con_num):
                if random.random() < 0.2:  # 20% 概率缺失
                    txt_con_mask[i, j] = 0  # 第 j 个条件文本缺失
    
        # 根据 self.img_con_num 动态获取条件图像
        img_con_data_list = [data_mapping[name].to(self.device) for name in self.img_con_name]
        # 初始化 img_con_mask: [batch_size, img_con_num]
        img_con_mask = torch.ones(batch_size, self.img_con_num, device=self.device)  # 全 1 表示没有缺失
        # 随机设置每个条件图像是否缺失
        for i in range(batch_size):
            for j in range(self.img_con_num):
                if random.random() < 0.2:  # 20% 概率缺失
                    img_con_mask[i, j] = 0  # 第 j 个条件图像缺失
        # 应用 mask 到每个条件图像
        img_con_data_list = [
            img_con_data * img_con_mask[:, j].view(-1, 1, 1, 1, 1)
            for j, img_con_data in enumerate(img_con_data_list)
        ]
        # 合并条件图像
        img_con = torch.cat(img_con_data_list, dim=1)
        if not self.is_text:
            txt = None
        if not self.is_img:
            img_con = None
        data_mapping.pop('T1C_tumor')
        data_mapping.pop('T2_tumor')
        return x_0, x_T, txt, txt_con_mask, img_con, img_con_mask, patient_id, data_mapping
    
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
    
    def backward_f(self, x_0, x_t, x_T, t, T, text_features=None, img_features=None):
        "x_T->x_0 "
        pred_x0_from_xt = self.net_f(x_t, timesteps=t, text_features=text_features, img_features=img_features)
        pred_x0_fron_xT = self.net_f(x_T, timesteps=T, text_features=text_features, img_features=img_features)
        with torch.no_grad():
            pred_xT_from_x0 = self.net_b(x_0.detach(), timesteps=T, text_features=text_features, img_features=img_features)
        pred_x0_from_x0 = self.net_f(pred_xT_from_x0.detach(), timesteps=T, text_features=text_features, img_features=img_features)
        
        loss_f_con = self.loss(pred_x0_from_xt, pred_x0_fron_xT)
        loss_f_rec_t = self.loss(pred_x0_from_xt, x_0)
        loss_f_rec_T = self.loss(pred_x0_fron_xT, x_0)
        loss_f_rec = (loss_f_rec_t + loss_f_rec_T) * 0.5
        loss_f_cycle = self.loss(pred_x0_from_x0, x_0)
        loss_f = self.lambda_con * loss_f_con + self.lambda_rec * loss_f_rec + self.lambda_cycle * loss_f_cycle
        loss_f_dict = {
            'loss_f': loss_f.item(),
            'loss_f_con': loss_f_con.item(),
            'loss_f_rec': loss_f_rec.item(),
            'loss_f_cycle': loss_f_cycle.item(),
        }
        if self.is_perceptual:
            loss_f_perceptual_t = self.perceptual_loss(pred_x0_from_xt, x_0)
            loss_f_perceptual_T = self.perceptual_loss(pred_x0_fron_xT, x_0)
            loss_f_perceptual_0 = self.perceptual_loss(pred_x0_from_x0, x_0)
            loss_f_perceptual = ((loss_f_perceptual_t + loss_f_perceptual_T + loss_f_perceptual_0) / 3.).mean()
            loss_f += self.lambda_perceptual * loss_f_perceptual
            loss_f_dict['loss_f_perceptual'] = loss_f_perceptual.item()
            loss_f_dict['loss_f'] = loss_f.item()

        # 梯度回传
        loss_f.backward()  # 反向传播

        return loss_f, loss_f_dict
        
    def backward_b(self, x_0, x_t, x_T, t, T, text_features=None, img_features=None):
        " x_0->x_T "
        pred_xT_from_xt = self.net_b(x_t, timesteps=(T-t), text_features=text_features, img_features=img_features)
        pred_xT_from_x0 = self.net_b(x_0, timesteps=T, text_features=text_features, img_features=img_features)
        with torch.no_grad():
            pred_x0_from_xT = self.net_f(x_T.detach(), timesteps=T, text_features=text_features, img_features=img_features)
        pred_xT_from_xT = self.net_b(pred_x0_from_xT.detach(), timesteps=T, text_features=text_features, img_features=img_features)

        loss_b_con = self.loss(pred_xT_from_xt, pred_xT_from_x0)
        loss_b_rec_t = self.loss(pred_xT_from_xt, x_T)
        loss_b_rec_T = self.loss(pred_xT_from_x0, x_T)
        loss_b_rec = (loss_b_rec_t + loss_b_rec_T) * 0.5
        loss_b_cycle = self.loss(pred_xT_from_xT, x_T)
        loss_b = self.lambda_con * loss_b_con + self.lambda_rec * loss_b_rec + self.lambda_cycle * loss_b_cycle
        loss_b_dict = {
            'loss_b': loss_b.item(),
            'loss_b_con': loss_b_con.item(),
            'loss_b_rec': loss_b_rec.item(),
            'loss_b_cycle': loss_b_cycle.item(),
        }
        if self.is_perceptual:
            loss_b_perceptual_t = self.perceptual_loss(pred_xT_from_xt, x_T)
            loss_b_perceptual_T = self.perceptual_loss(pred_xT_from_x0, x_T)
            loss_b_perceptual_0 = self.perceptual_loss(pred_xT_from_xT, x_T)
            loss_b_perceptual = ((loss_b_perceptual_t + loss_b_perceptual_T + loss_b_perceptual_0) / 3.).mean()
            loss_b += self.lambda_perceptual * loss_b_perceptual
            loss_b_dict['loss_b_perceptual'] = loss_b_perceptual.item()
            loss_b_dict['loss_b'] = loss_b.item()

        # 梯度回传
        loss_b.backward()  # 反向传播
        
        return loss_b, loss_b_dict

    def get_loss_names(self):
        loss_names = ['loss', 'loss_f', 'loss_f_con', 'loss_f_rec', 'loss_f_cycle', 'loss_b', 'loss_b_con', 'loss_b_rec', 'loss_b_cycle']
        if self.is_perceptual:
            loss_names.append('loss_f_perceptual')
            loss_names.append('loss_b_perceptual')
        return loss_names
    
    def p_losses(self, x_0, x_T, t, T, text_con=None, img_con=None, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_0))
        x_t = self.q_sample(x_0, x_T, t, noise)

        # 计算条件特征
        if self.is_text or self.is_img:
            with torch.no_grad():
                text_features = self.text_feature_extraction(text_con)
        else:
            text_features = None

        # 1️⃣ 计算 net_f 的损失
        # x_T -> x_0
        loss_f, loss_f_dict = self.backward_f(x_0, x_t, x_T, t, T, text_features, img_con)
        # torch.cuda.empty_cache()  # 清理 CUDA 缓存，释放内存

        # 2️⃣ 计算 net_b 的损失
        # x_0 -> x_T
        loss_b, loss_b_dict = self.backward_b(x_0, x_t, x_T, t, T, text_features, img_con)
        # torch.cuda.empty_cache()  # 再次清理 CUDA 缓存

        # 总损失
        loss = loss_f + loss_b
        loss_dict = {'loss': loss.item()}
        loss_dict.update(loss_f_dict)
        loss_dict.update(loss_b_dict)

        return loss, loss_dict
    
    def forward(self, x_0, x_T, 
                text_con:torch.Tensor = None, 
                img_con:torch.Tensor = None,): 
        t = torch.randint(0, self.num_timesteps-1, (x_0.shape[0],), device=self.device).long()
        T = torch.full((x_0.shape[0],), self.num_timesteps - 1, device=self.device).long()
        self.optimizer.zero_grad()
        loss, loss_dict = self.p_losses(x_0, x_T, t=t, T=T, text_con=text_con, img_con=img_con)
        self.optimizer.step()
        self.scheduler.step() 
        return loss.item(), loss_dict
        
    def p_sample_loop(self, sour_data, txt_con, img_con):

        T = torch.full((sour_data.shape[0],), self.steps[0], device=self.device).long()
        pred_targ = self.net(sour_data, T, txt_con, img_con)

        return pred_targ
    
    def test_batch(self, sour_data, mode,
                  txt_con: torch.Tensor = None, 
                  img_con: torch.Tensor = None, 
                  ):
        # 使用单GPU的原始模型进行验证
        if mode == 'f':
            self.net = self.net_f_single
        elif mode == 'b':
            self.net = self.net_b_single
        else:
            raise ValueError('mode should be "f" or "b"')
        
        self.net.eval()
        
        # 初始化预测结果张量和权重张量
        pred_targ = torch.zeros_like(sour_data)
        weight = torch.zeros_like(sour_data)  # 用于记录每个像素的加权次数
        
        # 验证时始终使用batch_size=4，不管是否多GPU
        b = 4
        
        depth, height, width = sour_data.shape[2:]  # 提取深度、高度和宽度
        d = self.img_d  # 每个块的深度大小
        overlap = int(d * self.overlap)  # 将浮点数转换为整数
        txt_con_batch = []
        if self.is_text:
            txt_con_batch = txt_con * b

        # 生成 d_start_list：所有分块的起始深度索引
        d_start_list = []
        for d_start in range(0, depth, d - overlap):
            if d_start + d > depth:
                d_start = depth - d  # 调整最后一块的起始位置
            d_start_list.append(d_start)

        # 去重，避免重复索引
        d_start_list = list(dict.fromkeys(d_start_list))

        # 计算批次数（向上取整）
        batch_num = math.ceil(len(d_start_list) / b)

        # 按批次处理分块
        for i in range(batch_num):
            # 获取当前批次的 d_start 列表
            if (i+1)*b > (len(d_start_list)+1):
                d_start_list_batch = d_start_list[len(d_start_list)-b:len(d_start_list)]
            else:
                d_start_list_batch = d_start_list[i * b : (i+1)*b]
            while len(d_start_list_batch) < b:
                d_start_list_batch.append(random.choice(d_start_list))
            actual_batch_size = len(d_start_list_batch)  # 当前批次的实际大小

            # 初始化批次张量
            xT_batch = torch.zeros((actual_batch_size, 1, d, height, width), device=sour_data.device)
            if self.is_img:
                img_con_batch = torch.zeros((actual_batch_size, self.img_con_num, d, height, width), device=sour_data.device)

            # 填充当前批次的输入
            for j in range(actual_batch_size):
                d_start = d_start_list_batch[j]
                xT_batch[j, :, :, :, :] = sour_data[:, :, d_start : d_start + d, :, :]
                if self.is_img:
                    img_con_batch[j, :, :, :, :] = img_con[:, :, d_start : d_start + d, :, :]

            text_features = self.text_feature_extraction(txt_con_batch)
            # 推理当前批次
            with torch.no_grad():
                pred_targ_batch = self.p_sample_loop(xT_batch, text_features, img_con_batch, clip_denoised=True)
                torch.cuda.empty_cache()

            # 将生成结果拼接回预测张量
            for j in range(actual_batch_size):
                d_start = d_start_list_batch[j]
                pred_targ[:, :, d_start : d_start + d, :, :] += pred_targ_batch[j, :, :, :, :]
                weight[:, :, d_start : d_start + d, :, :] += 1

        # 对重叠区域进行归一化
        pred_targ /= weight

        return pred_targ
    
    def test(self, x_0, x_T,
            txt_con: torch.Tensor = None, 
            img_con: torch.Tensor = None):
        "x->y"
        # 使用原始模型进行验证
        self.net_f_single.eval()
        self.net_b_single.eval()
        
        with torch.no_grad():
            pred_x0 = self.test_batch(x_T, 'f', txt_con, img_con)
            pred_xT = self.test_batch(x_0, 'b', txt_con, img_con)

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
        b, c, depth, height, width = sour_data.shape
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

            # 提取文本特征
            text_features = None
            txt_con_batch = txt_con * actual_batch_size
            if self.is_text:
                text_features = self.text_feature_extraction(txt_con_batch)

            # 推理当前批次
            with torch.no_grad():
                pred_targ_batch = self.p_sample_loop(xT_batch, text_features, img_con_batch)

            # 将生成结果拼接回预测张量
            for j, (d_start, h_start, w_start) in enumerate(batch_indices):
                pred_targ[:, :, d_start:d_start + patch_d, h_start:h_start + patch_h, w_start:w_start + patch_w] += pred_targ_batch[j]
                weight[:, :, d_start:d_start + patch_d, h_start:h_start + patch_h, w_start:w_start + patch_w] += 1

        # 对重叠区域进行归一化
        pred_targ /= weight

        return pred_targ
    
    def val(self, x_0, x_T,
            txt_con: torch.Tensor = None, 
            img_con: torch.Tensor = None):
        "x->y"
        self.net_f.eval()
        self.net_b.eval()
        with torch.no_grad():
            pred_x0 = self.val_batch(x_T, 'f', txt_con, img_con)
            pred_xT = self.val_batch(x_0, 'b', txt_con, img_con)

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
        self.save_nii(x0, os.path.join(path, f"{self.sour_name}.nii.gz"))
        self.save_nii(xT, os.path.join(path, f"{self.targ_name}.nii.gz"))
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
