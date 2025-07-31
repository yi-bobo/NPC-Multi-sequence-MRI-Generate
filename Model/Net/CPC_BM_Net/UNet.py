
from __future__ import annotations
import sys
sys.path.append("/data1/weiyibo/NPC-MRI/Code/NPC-Multi-sequence-MRI-Generate/")
import torch
import torch.nn as nn
from monai.networks.blocks import Convolution

# region  U-Net 网络
from Model.Net.CPC_BM_Net.zero_module import zero_module
from Model.Net.CPC_BM_Net.time_embedding import get_timestep_embedding
from Model.Net.CPC_BM_Net.attention_block import MultiHeadAttention
from Model.Net.CPC_BM_Net.up_block import UpSample, DiffusionUpSample
from Model.Net.CPC_BM_Net.down_block import DownSample, DiffusionDownSample
from Model.Net.CPC_BM_Net.resnet_block import ResnetBlock, DiffusionResnetBlock, ConditionalResnetBlock, ConditionalDiffusionResnetBlock

class DiffusionUNet(nn.Module):
    """
    基于 U-Net 的扩散模型网络结构
    """
    def __init__(
        self,
        spatial_dims: int,         # Spatial dimension of the data (2D or 3D)
        in_channels: int,
        out_channels: int,
        channels: list[int],
        num_res_blocks: list[int],
        num_heads: int=4,
        norm_num_groups: int=32,
        norm_eps: float=1e-6,
        with_attn: bool=True,
    ):
        super().__init__()
        
        # 1.timestep embedding
        time_dim = channels[0] * 4
        self.time_embed = nn.Sequential(
            nn.Linear(channels[0], time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # 2.Input Convolution
        self.conv_in = Convolution(
            spatial_dims=spatial_dims, 
            in_channels=in_channels, 
            out_channels=channels[0],
            padding=1,
            conv_only=True,             # //?保持线性特征映射，不破坏噪声分布 
            )
        
        # 3.Downsampling Blocks
        self.down_blocks = nn.ModuleList()
        for i in range(int(len(channels)-1)):
            is_last = (i == len(channels) - 2)   # 是否为最后一个下采样块,在最后一个下采样块只在 H和 W 维度下采样
            in_ch = channels[i]
            ch = channels[i+1]

            if is_last:
                stride=(1,2,2)
                kernel_size=(1,3,3)
                padding=(0,1,1)
            else:
                stride=(2,2,2)
                kernel_size=(3,3,3)
                padding=(1,1,1)
            for _ in range(num_res_blocks[i]):
                res_block = DiffusionResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=in_ch,
                    out_channels=in_ch,
                    temb_channels=time_dim,
                )
                self.down_blocks.append(res_block)
            down_block = DiffusionDownSample(
                spatial_dims=spatial_dims,
                in_channels=in_ch,
                out_channels=ch,
                stride=stride,
                kernel_size=kernel_size,
                padding=padding
            )
            self.down_blocks.append(down_block)
        
        # 4.MiddleSampling Blocks
        mid_ch = channels[-1]
        self.mid_blocks = nn.ModuleList()
        if with_attn:
            attn_block = MultiHeadAttention(
                spatial_dims=spatial_dims,
                channels=mid_ch,
                num_heads=num_heads,
            )
            self.mid_blocks.append(attn_block)
            res_block = DiffusionResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=mid_ch,
                    out_channels=mid_ch,
                    temb_channels=time_dim,
                )
            self.mid_blocks.append(res_block)
            attn_block = MultiHeadAttention(
                spatial_dims=spatial_dims,
                channels=mid_ch,
                num_heads=num_heads,
            )
            self.mid_blocks.append(attn_block)
        else:
            res_block = DiffusionResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=mid_ch,
                    out_channels=mid_ch,
                    temb_channels=time_dim,
                )
            self.mid_blocks.append(res_block)
        
        # 5.UpSampling Block
        self.up_blocks = nn.ModuleList()
        up_channels = list(reversed(channels))
        up_num_res_blocks = list(reversed(num_res_blocks))
        for i in range(len(up_channels)-1):
            ch = up_channels[i]
            out_ch = up_channels[i+1]
            if i == 0:
                stride=(1,2,2)
                kernel_size=(1,3,3)
                padding=(0,1,1)
            else:
                stride=(2,2,2)
                kernel_size=(3,3,3)
                padding=(1,1,1)
            up_block = DiffusionUpSample(
                spatial_dims=spatial_dims,
                in_channels=ch,
                out_channels=out_ch,
                stride=stride,
                kernel_size=kernel_size,
                padding=padding
            )
            self.up_blocks.append(up_block)
            for j in range(up_num_res_blocks[i]):
                res_block = DiffusionResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=out_ch*2,
                    out_channels=out_ch,
                    temb_channels=time_dim,
                )
                self.up_blocks.append(res_block)

        # 6.out conv
        self.out = nn.Sequential(
            nn.GroupNorm(norm_num_groups, channels[0], eps=norm_eps, affine=True),
            nn.SiLU(),
            zero_module(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=channels[0],
                    out_channels=out_channels,
                    strides=1,
                    kernel_size=3,
                    padding=1,
                    conv_only=True,
                )
            ),
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        x: [B, in_channels, D, H, W] 输入图像/特征
        timesteps: [B] 扩散步骤 t (整数索引)，会嵌入成时间特征
        """
        # 1. timestep embedding (sinusoidal or given)
        # 先对 timestep 做 positional embedding（假设外部已处理好）
        t_emb = get_timestep_embedding(timesteps, self.conv_in.out_channels)
        t_emb = self.time_embed(t_emb)   # [B, time_dim]

        # 2. Input conv
        h = self.conv_in(x)   # [B, C0, D, H, W]

        # 初始化 skip samples 用于解码器
        skip_sample: list[torch.Tensor] = [h]

        # 3. Downsampling
        for down_block in self.down_blocks:
            if isinstance(down_block, DiffusionDownSample):
                h = down_block(h, t_emb)  # [B, C, D, H, W]
            else:
                h = down_block(h, t_emb)       # [B, C, D/2, H/2, W/2]
                skip_sample.append(h)

        # 4. Middle blocks
        for mid_block in self.mid_blocks:
            h = mid_block(h, t_emb)

        # 5. UpSampling
        # 注意：up_blocks 数量与 skip_sample 配对（跳跃连接）
        # 因为 skip_sample 是正向顺序存的，这里需要反向取
        for i, up_block in enumerate(self.up_blocks):
            if isinstance(up_block, DiffusionUpSample):
                h = up_block(h, t_emb)
            else:
                skip_h = skip_sample.pop()
                h = torch.cat([h, skip_h], dim=1)
                h = up_block(h, t_emb)

        # 6. Output conv
        out = self.out(h)  # [B, out_channels, D, H, W]

        return out

from Model.Net.CPC_BM_Net.condition import TextEncoder, CondImageEncoder, T2I_OTF

class ConditionalDiffusionUNet(nn.Module):
    """
    基于 U-Net 的扩散模型网络结构
    """
    def __init__(
        self,
        spatial_dims: int,         # Spatial dimension of the data (2D or 3D)
        in_channels: int,
        out_channels: int,
        channels: list[int],
        text_channels: int,
        cond_channels: int,
        num_res_blocks: list[int],
        num_heads: int=4,
        norm_num_groups: int=32,
        norm_eps: float=1e-6,
        epsilon: float=0.1,
        niter: int=50,
        with_attn: bool=True,
    ):
        super().__init__()
        
        # 1.timestep embedding
        time_dim = channels[0] * 4
        self.time_embed = nn.Sequential(
            nn.Linear(channels[0], time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # //?文本编码
        self.text_encode = TextEncoder(text_channels=text_channels, channels=channels)
        # //?条件图像编码
        self.cond_image_encode = CondImageEncoder(spatial_dims=spatial_dims, cond_channels=cond_channels, channels=channels)
        # //?文本-图像 最优化特征对齐
        self.text_cond_ot_fusion = T2I_OTF(epsilon=epsilon, niter=niter, spatial_dims=spatial_dims, channels=channels)

        # 2.Input Convolution
        self.conv_in = Convolution(
            spatial_dims=spatial_dims, 
            in_channels=in_channels, 
            out_channels=channels[0],
            padding=1,
            conv_only=True,             # //?保持线性特征映射，不破坏噪声分布 
            )
        
        # 3.Downsampling Blocks
        self.down_blocks = nn.ModuleList()
        self.text_down_blocks = nn.ModuleList()
        self.cond_down_blocks = nn.ModuleList()
        for i in range(int(len(channels)-1)):
            in_ch = channels[i]
            ch = channels[i+1]
            if i == (len(channels)-2):
                only_down_HW = True
            else:
                only_down_HW = False
            
            for _ in range(num_res_blocks[i]):
                res_block = DiffusionResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=in_ch,
                    out_channels=in_ch,
                    temb_channels=time_dim,
                )
                self.down_blocks.append(res_block)
            down_block = DiffusionDownSample(
                spatial_dims=spatial_dims,
                in_channels=in_ch,
                out_channels=ch,
                only_down_HW=only_down_HW
            )
            self.down_blocks.append(down_block)
        
        # 4.MiddleSampling Blocks
        mid_ch = channels[-1]
        self.mid_blocks = nn.ModuleList()
        if with_attn:
            attn_block = MultiHeadAttention(
                spatial_dims=spatial_dims,
                channels=mid_ch,
                num_heads=num_heads,
            )
            self.mid_blocks.append(attn_block)
            res_block = DiffusionResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=mid_ch,
                    out_channels=mid_ch,
                    temb_channels=time_dim,
                )
            self.mid_blocks.append(res_block)
            attn_block = MultiHeadAttention(
                spatial_dims=spatial_dims,
                channels=mid_ch,
                num_heads=num_heads,
            )
            self.mid_blocks.append(attn_block)
        else:
            res_block = DiffusionResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=mid_ch,
                    out_channels=mid_ch,
                    temb_channels=time_dim,
                )
            self.mid_blocks.append(res_block)
        
        # 5.UpSampling Block
        self.up_blocks = nn.ModuleList()
        up_channels = list(reversed(channels))
        up_num_res_blocks = list(reversed(num_res_blocks))
        for i in range(len(up_channels)-1):
            ch = up_channels[i]
            out_ch = up_channels[i+1]
            if i == 0:
                stride=(1,2,2)
                kernel_size=(1,3,3)
                padding=(0,1,1)
            else:
                stride=(2,2,2)
                kernel_size=(3,3,3)
                padding=(1,1,1)
            up_block = DiffusionUpSample(
                spatial_dims=spatial_dims,
                in_channels=ch,
                out_channels=out_ch,
                stride=stride,
                kernel_size=kernel_size,
                padding=padding
            )
            self.up_blocks.append(up_block)
            for j in range(up_num_res_blocks[i]):
                res_block = DiffusionResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=out_ch*2,
                    out_channels=out_ch,
                    temb_channels=time_dim,
                )
                self.up_blocks.append(res_block)

        # 6.out conv
        self.out = nn.Sequential(
            nn.GroupNorm(norm_num_groups, channels[0], eps=norm_eps, affine=True),
            nn.SiLU(),
            zero_module(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=channels[0],
                    out_channels=out_channels,
                    strides=1,
                    kernel_size=3,
                    padding=1,
                    conv_only=True,
                )
            ),
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor,
                text_feat: torch.Tensor, cond_image: torch.Tensor) -> torch.Tensor:
        """
        x: [B, in_channels, D, H, W] 输入图像/特征
        timesteps: [B] 扩散步骤 t (整数索引)，会嵌入成时间特征
        """
        # 1. timestep embedding (sinusoidal or given)
        # 先对 timestep 做 positional embedding（假设外部已处理好）
        t_emb = get_timestep_embedding(timesteps, self.conv_in.out_channels)
        t_emb = self.time_embed(t_emb)   # [B, time_dim]

        text_feat_list = self.text_encode(text_feat)
        text_feat_list.reverse()
        cond_image_feat_list = self.cond_image_encode(cond_image)
        cond_image_feat_list.reverse()

        # 2. Input conv
        h = self.conv_in(x)   # [B, C0, D, H, W]
        t = text_feat_list.pop()
        c = cond_image_feat_list.pop()

        # 初始化 skip samples 用于解码器
        skip_sample: list[torch.Tensor] = [h]

        # 3. Downsampling
        for down_block in self.down_blocks:
            if isinstance(down_block, DiffusionDownSample):
                h = down_block(h, t_emb)  # [B, C, D, H, W]
                t = text_feat_list.pop()
                c = cond_image_feat_list.pop()
            else:
                h = down_block(h, t_emb)       # [B, C, D/2, H/2, W/2]
                skip_sample.append(h)

        # 4. Middle blocks
        for mid_block in self.mid_blocks:
            h = mid_block(h, t_emb)

        # 5. UpSampling
        # 注意：up_blocks 数量与 skip_sample 配对（跳跃连接）
        # 因为 skip_sample 是正向顺序存的，这里需要反向取
        for i, up_block in enumerate(self.up_blocks):
            if isinstance(up_block, DiffusionUpSample):
                h = up_block(h, t_emb)
            else:
                skip_h = skip_sample.pop()
                h = torch.cat([h, skip_h], dim=1)
                h = up_block(h, t_emb)

        # 6. Output conv
        out = self.out(h)  # [B, out_channels, D, H, W]

        return out
    
# ===== 假设 DiffusionUNet 已经定义并包含 forward =====
# 这里直接用上一次给出的 DiffusionUNet + forward 函数
# 注意：DiffusionResnetBlock / DiffusionDownSample / DiffusionUpSample / MultiHeadAttention / Convolution
# 都需要在同一个文件中定义或导入

if __name__ == "__main__":
    # 测试参数
    spatial_dims = 3        # 支持 3D 数据
    in_channels = 1         # 输入通道数
    out_channels = 1        # 输出通道数
    text_channels = 256     # 文本通道数
    cond_channels = 3       # 条件图像通道数
    channels = [32, 64, 128, 256, 512]
    num_res_blocks = [2, 2, 2, 2, 2]   # 每层的ResnetBlock个数
    gpu = 7
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    # 构建模型
    model = ConditionalDiffusionUNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        text_channels=text_channels,
        cond_channels=cond_channels,
        num_res_blocks=num_res_blocks,
        num_heads=4,
        with_attn=True,
    ).to(device)

    # 随机生成输入数据 [B, C, D, H, W]
    B, C, D, H, W = 4, in_channels, 8, 256, 256
    x = torch.randn(B, C, D, H, W).to(device)
    t = torch.randn(B, 256, 21).to(device)
    c = torch.randn(B, 3, D, H, W).to(device)

    # 构造时间步（随机整数）并映射到向量
    timesteps = torch.randint(low=0, high=1000, size=(B,)).to(device)
    
    # 前向传播
    with torch.no_grad():
        out = model(x, timesteps, t, c)

    print(f"Input shape : {x.shape}")
    print(f"Output shape: {out.shape}")