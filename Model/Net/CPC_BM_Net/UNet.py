from __future__ import annotations
from collections.abc import Sequence
import sys
sys.path.append("/data1/weiyibo/NPC-MRI/Code/NPC-Multi-sequence-MRI-Generate/")
import torch
import torch.nn as nn
from monai.networks.blocks import Convolution
from Model.Net.CPC_BM_Net.condition import TextEncoder, CondImageEncoder, T2I_OTF, DynamicConvFiLM


# region  U-Net 网络
from Model.Net.CPC_BM_Net.module import zero_module
from Model.Net.CPC_BM_Net.time_embedding import get_timestep_embedding
from Model.Net.CPC_BM_Net.unet_block import get_down_block, get_mid_block, get_up_block

class DiffusionModelUNet(nn.Module):
    def __init__(
        self,
        device,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        text_channels: int,
        image_channels: int,
        num_res_blocks: Sequence[int] | int = (2, 2, 2, 2),
        channels: Sequence[int] = (32, 64, 64, 64),
        attention_levels: Sequence[bool] = (False, False, True, True),
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        resblock_updown: bool = False,
        num_head_channels: int | Sequence[int] = 8,
        with_conditioning: bool = False,
        transformer_num_layers: int = 1,
        cross_attention_dim: int | None = None,
        upcast_attention: bool = False,
        dropout_cattn: float = 0.0,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
        shallow_layer_threshold: int = 2,  # 前几层被认为是浅层
        epsilon: float=0.1,
        niter: int=50,
        is_text: bool=True,
        is_image: bool=True
    ) -> None:
        super().__init__()
        
        self.in_channels = in_channels
        self.block_out_channels = channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_levels = attention_levels
        self.num_head_channels = num_head_channels
        self.with_conditioning = with_conditioning
        self.shallow_layer_threshold = shallow_layer_threshold

        # //*条件编码
        if is_text:
            # 文本编码
            self.text_encode = TextEncoder(text_channels=text_channels, channels=channels)
        if is_image:
            # 条件图像编码
            self.cond_image_encode = CondImageEncoder(spatial_dims=spatial_dims, cond_channels=image_channels, channels=channels)
        if is_text and is_image:
            # 文本-图像 最优化特征对齐
            self.text_cond_ot_fusion = T2I_OTF(epsilon=epsilon, niter=niter, spatial_dims=spatial_dims, channels=channels)
        
        # input
        self.conv_in = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=channels[0],
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )

        # time
        time_embed_dim = channels[0] * 4
        self.time_embed = nn.Sequential(
            nn.Linear(channels[0], time_embed_dim), 
            nn.SiLU(), 
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        # down
        self.down_blocks = nn.ModuleList([])
        output_channel = channels[0]
        for i in range(len(channels)):
            input_channel = output_channel
            output_channel = channels[i]
            is_final_block = i == len(channels) - 1

            down_block = get_down_block(
                spatial_dims=spatial_dims,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                num_res_blocks=num_res_blocks[i],
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                add_downsample=not is_final_block,
                resblock_updown=resblock_updown,
                with_attn=(attention_levels[i] and not with_conditioning),
                with_cross_attn=(attention_levels[i] and with_conditioning),
                num_head_channels=num_head_channels[i] if isinstance(num_head_channels, list) else num_head_channels,
                transformer_num_layers=transformer_num_layers,
                cross_attention_dim=cross_attention_dim,
                upcast_attention=upcast_attention,
                dropout_cattn=dropout_cattn,
                include_fc=include_fc,
                use_combined_linear=use_combined_linear,
                use_flash_attention=use_flash_attention,
            )

            self.down_blocks.append(down_block)

        # mid
        self.middle_block = get_mid_block(
            spatial_dims=spatial_dims,
            in_channels=channels[-1],
            temb_channels=time_embed_dim,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            with_conditioning=with_conditioning,
            num_head_channels=num_head_channels[-1] if isinstance(num_head_channels, list) else num_head_channels,
            transformer_num_layers=transformer_num_layers,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            dropout_cattn=dropout_cattn,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
        )

        # up
        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(channels))
        reversed_num_res_blocks = list(reversed(num_res_blocks))
        reversed_attention_levels = list(reversed(attention_levels))
        reversed_num_head_channels = list(reversed(num_head_channels)) if isinstance(num_head_channels, list) else [num_head_channels] * len(channels)
        output_channel = reversed_block_out_channels[0]
        for i in range(len(reversed_block_out_channels)):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(channels) - 1)]

            is_final_block = i == len(channels) - 1

            up_block = get_up_block(
                spatial_dims=spatial_dims,
                in_channels=input_channel,
                prev_output_channel=prev_output_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                num_res_blocks=reversed_num_res_blocks[i] + 1,
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                add_upsample=not is_final_block,
                resblock_updown=resblock_updown,
                with_attn=(reversed_attention_levels[i] and not with_conditioning),
                with_cross_attn=(reversed_attention_levels[i] and with_conditioning),
                num_head_channels=reversed_num_head_channels[i],
                transformer_num_layers=transformer_num_layers,
                cross_attention_dim=cross_attention_dim,
                upcast_attention=upcast_attention,
                dropout_cattn=dropout_cattn,
                include_fc=include_fc,
                use_combined_linear=use_combined_linear,
                use_flash_attention=use_flash_attention,
            )

            self.up_blocks.append(up_block)

        # out
        self.out = nn.Sequential(
            nn.GroupNorm(num_groups=norm_num_groups, num_channels=channels[0], eps=norm_eps, affine=True),
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

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        text_feat: torch.Tensor=None,
        image_feat: torch.Tensor=None
    ) -> torch.Tensor:
        """
        Args:
            x: input tensor (B, C, SpatialDims).
            timesteps: timestep tensor (B,).
            text_feat: tensor (B, C, N).
            image_feat: tensor (B, C, SpatialDims).
        """
        # 1. time
        t_emb = get_timestep_embedding(timesteps, self.block_out_channels[0])
        t_emb = t_emb.to(dtype=x.dtype)
        emb = self.time_embed(t_emb)

        # 2. initial convolution
        h = self.conv_in(x)
        

        # 3. down blocks with conditional fusion
        down_block_res_samples: list[torch.Tensor] = [h]
        
        for i, downsample_block in enumerate(self.down_blocks):
            # 下采样块处理
            h, res_samples = downsample_block(hidden_states=h, temb=emb)
            for residual in res_samples:
                down_block_res_samples.append(residual)

        # 4. middle block with conditional fusion
        h = self.middle_block(hidden_states=h, temb=emb)
        h = self.middle_block(hidden_states=h, temb=emb)

        # 5. up blocks
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]
            h = upsample_block(hidden_states=h, res_hidden_states_list=res_samples, temb=emb)

        # 6. output block
        output: torch.Tensor = self.out(h)

        return output

from Model.Net.CPC_BM_Net.attention_block import MultiHeadAttention
from Model.Net.CPC_BM_Net.up_sample import DiffusionUpSample
from Model.Net.CPC_BM_Net.down_sample import DiffusionDownSample
from Model.Net.CPC_BM_Net.resnet_block import DiffusionResnetBlock

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


class ConditionalDiffusionUNet(nn.Module):
    """
    基于 U-Net 的扩散模型网络结构
    """
    def __init__(
        self,
        device,
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
        self.mode_embed = nn.Sequential(
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
        self.conv_in_blocks = nn.ModuleList()
        self.conv_in = Convolution(
            spatial_dims=spatial_dims, 
            in_channels=in_channels, 
            out_channels=channels[0],
            padding=1,
            conv_only=True,             # //?保持线性特征映射，不破坏噪声分布 
            )
        for _ in range(num_res_blocks[0]):
            res_block = DiffusionResnetBlock(
                spatial_dims=spatial_dims,
                in_channels=channels[0],
                out_channels=channels[0],
                temb_channels=time_dim,
                )
            self.conv_in_blocks.append(res_block)
        
        # 3.Downsampling Blocks
        self.down_blocks = nn.ModuleList()
        self.text_down_blocks = nn.ModuleList()
        self.cond_down_blocks = nn.ModuleList()
        self.cond_fusion_blocks = nn.ModuleList()
        for i in range(int(len(channels)-1)):
            in_ch = channels[i]
            ch = channels[i+1]
            # //?网络特征与条件特征融合，浅层动态卷积FiLM调制融合，深层cross-attn融合。
            is_Shallow = True if in_ch <= channels[-2] else False

            if is_Shallow:
                fusion_block = DynamicConvFiLM(channels=in_ch, spatial_dims=spatial_dims)
                self.down_blocks.append(fusion_block)
            else:
                fusion_block = MultiHeadAttention(spatial_dims=spatial_dims,channels=in_ch,num_heads=num_heads,cross_attention=True)
                self.down_blocks.append(fusion_block)

            if i == (len(channels)-2):
                only_down_HW = True
            else:
                only_down_HW = False

            down_block = DiffusionDownSample(
                spatial_dims=spatial_dims,
                in_channels=in_ch,
                out_channels=ch,
                only_down_HW=only_down_HW
            )
            self.down_blocks.append(down_block)
            for _ in range(num_res_blocks[i]):
                res_block = DiffusionResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=ch,
                    out_channels=ch,
                    temb_channels=time_dim,
                )
                self.down_blocks.append(res_block)
            
        
        # 4.MiddleSampling Blocks
        mid_ch = channels[-1]
        self.mid_blocks = nn.ModuleList()
        self.mid_fusion_blocks = MultiHeadAttention(spatial_dims=spatial_dims,channels=mid_ch,num_heads=num_heads,cross_attention=True)
        if with_attn:
            attn_block = MultiHeadAttention(
                spatial_dims=spatial_dims,
                channels=mid_ch,
                num_heads=num_heads,
            )
            self.mid_blocks.append(attn_block)
            for _ in range(num_res_blocks[-1]):
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
            for _ in range(num_res_blocks[-1]):
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
                only_up_HW = True
            else:
                only_up_HW = False
            up_block = DiffusionUpSample(
                spatial_dims=spatial_dims,
                in_channels=ch,
                out_channels=out_ch,
                only_up_HW=only_up_HW
            )
            self.up_blocks.append(up_block)
            for j in range(up_num_res_blocks[i]):
                if j==0:
                    res_block = DiffusionResnetBlock(
                        spatial_dims=spatial_dims,
                        in_channels=out_ch*2,
                        out_channels=out_ch,
                        temb_channels=time_dim,
                    )
                    self.up_blocks.append(res_block)
                else:
                    res_block = DiffusionResnetBlock(
                        spatial_dims=spatial_dims,
                        in_channels=out_ch,
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
                text_feat: torch.Tensor, cond_image: torch.Tensor, mode: torch.Tensor=None) -> torch.Tensor:
        """
        x: [B, in_channels, D, H, W] 输入图像/特征
        timesteps: [B] 扩散步骤 t (整数索引)，会嵌入成时间特征
        """
        # 1. timestep embedding
        t_emb = get_timestep_embedding(timesteps, self.conv_in.out_channels)
        emb = self.time_embed(t_emb)   # [B, time_dim]
        if mode is not None:
            mode_emb = self.mode_embed(mode)
            emb = emb + mode_emb
        # 文本特征与图像特征提取、对齐、融合
        text_feat_list = self.text_encode(text_feat)
        cond_image_feat_list = self.cond_image_encode(cond_image)
        cond_feat_list = self.text_cond_ot_fusion(text_feat_list, cond_image_feat_list)
        cond_feat_list.reverse()

        # 2. Input conv
        h = self.conv_in(x)
        for conv_in in self.conv_in_blocks:
            h = conv_in(h, emb)

        # 初始化 skip samples 用于解码器
        skip_sample: list[torch.Tensor] = []

        # 3. Downsampling
        for down_block in self.down_blocks:
            if isinstance(down_block, DynamicConvFiLM):
                skip_sample.append(h)
                h_cond = cond_feat_list.pop()
                h = down_block(h, h_cond)
            elif isinstance(down_block, MultiHeadAttention):
                skip_sample.append(h)
                h_cond = cond_feat_list.pop()
                h = down_block(h, h_cond)
            elif isinstance(down_block, DiffusionDownSample):
                h = down_block(h, emb)  # [B, C, D, H, W]
            else:# 残差块
                h = down_block(h, emb)       # [B, C, D/2, H/2, W/2]

        # 4. Middle blocks
        h_cond = cond_feat_list.pop()
        h = self.mid_fusion_blocks(h, h_cond)
        for mid_block in self.mid_blocks:
            h = mid_block(h, emb)
        h = self.mid_fusion_blocks(h, h_cond)

        # 5. UpSampling
        # 注意：up_blocks 数量与 skip_sample 配对（跳跃连接）
        # 因为 skip_sample 是正向顺序存的，这里需要反向取
        for i, up_block in enumerate(self.up_blocks):
            if isinstance(up_block, DiffusionUpSample):
                h = up_block(h, emb)
                skip_h = skip_sample.pop()
                h = torch.cat([h, skip_h], dim=1)
            else:
                h = up_block(h, emb)

        # 6. Output conv
        out = self.out(h)  # [B, out_channels, D, H, W]

        return out
    
# ===== 假设 DiffusionUNet 已经定义并包含 forward =====

# if __name__ == "__main__":
#     # 测试参数
#     spatial_dims = 3        # 支持 3D 数据
#     in_channels = 1         # 输入通道数
#     out_channels = 1        # 输出通道数
#     text_channels = 256     # 文本通道数
#     cond_channels = 3       # 条件图像通道数
#     channels = [32, 64, 128, 256, 512]
#     num_res_blocks = [2, 2, 2, 2, 2]   # 每层的ResnetBlock个数
#     gpu = 7
#     device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

#     # 构建模型
#     model = ConditionalDiffusionUNet(
#         spatial_dims=spatial_dims,
#         in_channels=in_channels,
#         out_channels=out_channels,
#         channels=channels,
#         text_channels=text_channels,
#         cond_channels=cond_channels,
#         num_res_blocks=num_res_blocks,
#         num_heads=4,
#         with_attn=True,
#     ).to(device)

#     # 随机生成输入数据 [B, C, D, H, W]
#     B, C, D, H, W = 4, in_channels, 8, 256, 256
#     x = torch.randn(B, C, D, H, W).to(device)
#     t = torch.randn(B, 256, 21).to(device)
#     c = torch.randn(B, 3, D, H, W).to(device)

#     # 构造时间步（随机整数）并映射到向量
#     timesteps = torch.randint(low=0, high=1000, size=(B,)).to(device)
    
#     # 前向传播
#     with torch.no_grad():
#         out = model(x, timesteps, t, c)

#     print(f"Input shape : {x.shape}")
#     print(f"Output shape: {out.shape}")