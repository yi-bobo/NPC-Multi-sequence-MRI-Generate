from __future__ import annotations
from collections.abc import Sequence
import torch
from torch import nn

from .Blocks.module import zero_module
from .Blocks.time_embedding import get_timestep_embedding
from .Blocks.unet_block import get_down_block, get_mid_block, get_up_block, get_image_down_block

from monai.networks.blocks import Convolution

class AdaptiveGatingModule(nn.Module):
    """自适应门控模块，用于计算融合权重"""
    def __init__(self, in_channels, spatial_dims=3):
        super().__init__()
        self.spatial_dims = spatial_dims
        
        # 门控网络
        self.gate_conv = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels * 2,  # 原始特征 + 融合特征
                out_channels=in_channels,
                kernel_size=3,
                padding=1,
                conv_only=True
            ),
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=1,
                kernel_size=1,
                conv_only=True
            ),
            nn.Sigmoid()
        )
    
    def forward(self, x, fused_features):
        """计算自适应权重并融合特征"""
        gate_input = torch.cat([x, fused_features], dim=1)
        gate_weight = self.gate_conv(gate_input)
        return x + gate_weight * fused_features


class ConditionalFusionModule(nn.Module):
    """条件融合模块，处理文本和图像条件"""
    def __init__(
        self,
        spatial_dims,
        in_channels,
        text_dim=768,
        is_shallow_layer=True,
        num_heads=8,
        dropout=0.0
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.is_shallow_layer = is_shallow_layer
        
        # 文本特征映射
        self.text_proj = nn.Linear(text_dim, in_channels)
        
        # 浅层使用卷积融合，深层使用cross-attention
        if is_shallow_layer:
            # 卷积融合模块
            self.conv_fusion = nn.Sequential(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels * 2,  # 网络特征 + 条件特征
                    out_channels=in_channels,
                    kernel_size=3,
                    padding=1,
                    conv_only=True
                ),
                nn.GroupNorm(8, in_channels),
                nn.SiLU(),
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=3,
                    padding=1,
                    conv_only=True
                )
            )
        else:
            # Cross-attention融合模块
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=in_channels,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.norm1 = nn.LayerNorm(in_channels)
            self.norm2 = nn.LayerNorm(in_channels)
            
        # 自适应门控
        self.adaptive_gate = AdaptiveGatingModule(in_channels, spatial_dims)
    
    def forward(self, x, text_features=None, img_features=None):
        """
        x: 网络特征 (B, C, *spatial)
        text_features: 文本特征 (B, L, C) 
        img_features: 图像特征 (B, C, *spatial)
        """
        B, C = x.shape[:2]
        spatial_shape = x.shape[2:]
        
        # 情况1：没有条件信息
        if text_features is None and img_features is None:
            return x
        
        fused_features = None
        
        # 情况2：只有文本特征
        if text_features is not None and img_features is None:
            # 映射文本特征到通道维度
            text_feat = self.text_proj(text_features)  # (B, L, C)
            
            if self.is_shallow_layer:
                # 浅层：将文本特征广播到空间维度并使用卷积融合
                text_feat_spatial = text_feat.mean(dim=1, keepdim=True)  # (B, 1, C)
                text_feat_spatial = text_feat_spatial.permute(0, 2, 1)  # (B, C, 1)
                # 广播到空间维度
                for _ in range(self.spatial_dims-1):
                    text_feat_spatial = text_feat_spatial.unsqueeze(-1)
                text_feat_spatial = text_feat_spatial.expand(-1, -1, *spatial_shape)
                
                # 卷积融合
                concat_feat = torch.cat([x, text_feat_spatial], dim=1)
                fused_features = self.conv_fusion(concat_feat)
            else:
                # 深层：使用cross-attention
                x_flat = x.flatten(2).permute(0, 2, 1)  # (B, N, C)
                attn_out, _ = self.cross_attn(x_flat, text_feat, text_feat)
                fused_features = attn_out + x_flat
                fused_features = fused_features.permute(0, 2, 1).reshape(B, C, *spatial_shape)
        
        # 情况3：只有图像特征
        elif text_features is None and img_features is not None:
            if self.is_shallow_layer:
                # 浅层：卷积融合
                concat_feat = torch.cat([x, img_features], dim=1)
                fused_features = self.conv_fusion(concat_feat)
            else:
                # 深层：cross-attention融合
                x_flat = x.flatten(2).permute(0, 2, 1)  # (B, N, C)
                img_flat = img_features.flatten(2).permute(0, 2, 1)  # (B, N, C)
                attn_out, _ = self.cross_attn(x_flat, img_flat, img_flat)
                fused_features = attn_out + x_flat
                fused_features = fused_features.permute(0, 2, 1).reshape(B, C, *spatial_shape)
        
        # 情况4：同时有文本和图像特征
        else:
            # 映射文本特征
            text_feat = self.text_proj(text_features)  # (B, L, C)
            
            if self.is_shallow_layer:
                # 浅层：先融合文本和图像，再与网络特征融合
                text_feat_spatial = text_feat.mean(dim=1, keepdim=True).permute(0, 2, 1)
                for _ in range(self.spatial_dims-1):
                    text_feat_spatial = text_feat_spatial.unsqueeze(-1)
                text_feat_spatial = text_feat_spatial.expand(-1, -1, *spatial_shape)
                
                # 第一步：融合文本和图像
                text_img_concat = torch.cat([text_feat_spatial, img_features], dim=1)
                text_img_fused = self.conv_fusion(text_img_concat)
                
                # 第二步：与网络特征融合
                concat_feat = torch.cat([x, text_img_fused], dim=1)
                fused_features = self.conv_fusion(concat_feat)
            else:
                # 深层：使用cross-attention同时处理文本和图像
                x_flat = x.flatten(2).permute(0, 2, 1)  # (B, N, C)
                img_flat = img_features.flatten(2).permute(0, 2, 1)  # (B, N, C)
                
                # 拼接文本和图像作为key和value
                cond_features = torch.cat([text_feat, img_flat], dim=1)  # (B, L+N, C)

                attn_out, _ = self.cross_attn(x_flat, cond_features, cond_features)
                fused_features = attn_out + x_flat
                fused_features = fused_features.permute(0, 2, 1).reshape(B, C, *spatial_shape)
        
        # 使用自适应门控融合
        output = self.adaptive_gate(x, fused_features)
        return output


class ImageDownsampler(nn.Module):
    """图像下采样模块，将条件图像下采样到对应层的分辨率"""
    def __init__(self, spatial_dims, in_channels, out_channels, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        
        # 使用卷积进行下采样和通道调整
        layers = []
        current_channels = in_channels
        
        # 逐步下采样
        num_downsamples = int(torch.log2(torch.tensor(scale_factor)))
        for i in range(num_downsamples):
            next_channels = out_channels if i == num_downsamples - 1 else current_channels * 2
            layers.append(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=current_channels,
                    out_channels=next_channels,
                    strides=2,
                    kernel_size=3,
                    padding=1,
                    conv_only=True
                )
            )
            layers.append(nn.GroupNorm(8, next_channels))
            layers.append(nn.SiLU())
            current_channels = next_channels
        
        self.downsample = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.downsample(x)


class DiffusionModelUNet(nn.Module):
    def __init__(
        self,
        device,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_res_blocks: Sequence[int] | int = (2, 2, 2, 2),
        channels: Sequence[int] = (32, 64, 64, 64),
        attention_levels: Sequence[bool] = (False, False, True, True),
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        resblock_updown: bool = False,
        num_head_channels: int | Sequence[int] = 8,
        with_conditioning: bool = False,
        is_text: bool = False,
        is_img: bool = False,
        text_dim: int = 256,
        con_img_channels: int = 3,
        transformer_num_layers: int = 1,
        cross_attention_dim: int | None = None,
        upcast_attention: bool = False,
        dropout_cattn: float = 0.0,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
        shallow_layer_threshold: int = 2,  # 前几层被认为是浅层
    ) -> None:
        super().__init__()
        
        self.in_channels = in_channels
        self.block_out_channels = channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_levels = attention_levels
        self.num_head_channels = num_head_channels
        self.with_conditioning = with_conditioning
        self.is_text = is_text
        self.is_img = is_img
        self.shallow_layer_threshold = shallow_layer_threshold

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

        # 为每个下采样层创建条件融合模块和图像下采样器
        if is_text or is_img:
            self.cond_fusion_down = nn.ModuleList()
            self.img_downsamplers = nn.ModuleList() if is_img else None
            
            for i in range(len(channels)):
                # 判断是否为浅层
                is_shallow = i < shallow_layer_threshold
                
                # 条件融合模块
                fusion_module = ConditionalFusionModule(
                    spatial_dims=spatial_dims,
                    in_channels=channels[i],
                    text_dim=text_dim,
                    is_shallow_layer=is_shallow,
                    num_heads=num_head_channels[i] if isinstance(num_head_channels, list) else num_head_channels,
                    dropout=dropout_cattn
                )
                self.cond_fusion_down.append(fusion_module)
                
            # 图像下采样器
            if is_img:
                self.con_image_conv_in = Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=con_img_channels,
                    out_channels=channels[0],
                    strides=1,
                    kernel_size=3,
                    padding=1,
                    conv_only=True,
                )
                self.con_image_down_blocks = nn.ModuleList([])
                output_channel = channels[0]
                for i in range(len(channels)):
                    input_channel = output_channel
                    output_channel = channels[i]
                    is_final_block = i == len(channels) - 1
                    img_down_block = get_image_down_block(
                        spatial_dims=spatial_dims,
                        in_channels=input_channel,
                        out_channels=output_channel,
                        num_res_blocks=num_res_blocks[i],
                        norm_num_groups=norm_num_groups,
                        norm_eps=norm_eps,
                        add_downsample=not is_final_block,
                        resblock_updown=resblock_updown)
                    self.con_image_down_blocks.append(img_down_block)
            
            # 中间层的条件融合
            self.cond_fusion_mid = ConditionalFusionModule(
                spatial_dims=spatial_dims,
                in_channels=channels[-1],
                text_dim=text_dim,
                is_shallow_layer=False,  # 中间层总是使用cross-attention
                num_heads=num_head_channels[-1] if isinstance(num_head_channels, list) else num_head_channels,
                dropout=dropout_cattn
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
        text_features=None,
        img_features=None,
        down_block_additional_residuals: tuple[torch.Tensor] | None = None,
        mid_block_additional_residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: input tensor (N, C, SpatialDims).
            timesteps: timestep tensor (N,).
            text_features: text features (N, L, D).
            text_con_mask: text condition mask (N, L).
            img_features: image features (N, C_img, SpatialDims).
            img_con_mask: image condition mask.
            down_block_additional_residuals: additional residual tensors for down blocks.
            mid_block_additional_residual: additional residual tensor for mid block.
        """
        # 1. time
        t_emb = get_timestep_embedding(timesteps, self.block_out_channels[0])
        t_emb = t_emb.to(dtype=x.dtype)
        emb = self.time_embed(t_emb)

        # 2. initial convolution
        h = self.conv_in(x)
        if not self.is_img:
            current_img_features = None
        if self.is_img and img_features is not None:
            current_img_features = self.con_image_conv_in(img_features)
            con_img_down_block_res_samples: list[torch.Tensor] = [current_img_features]

        # 3. down blocks with conditional fusion
        down_block_res_samples: list[torch.Tensor] = [h]
        
        for i, downsample_block in enumerate(self.down_blocks):
            # 下采样块处理
            h, res_samples = downsample_block(hidden_states=h, temb=emb)
            for residual in res_samples:
                down_block_res_samples.append(residual)
            # 在每个下采样块之前进行条件融合
            if self.is_text or self.is_img:
                # 准备当前层的图像特征
                if self.is_img and img_features is not None:
                    current_img_features, con_img_res_samples = self.con_image_down_blocks[i](current_img_features)
                    for con_img_residual in con_img_res_samples:
                        con_img_down_block_res_samples.append(con_img_residual)
                
                # 应用条件融合
                h = self.cond_fusion_down[i](h, text_features, current_img_features)
            

        # Additional residual connections for ControlNets
        if down_block_additional_residuals is not None:
            new_down_block_res_samples: list[torch.Tensor] = []
            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples += [down_block_res_sample]

            down_block_res_samples = new_down_block_res_samples

        # 4. middle block with conditional fusion
        h = self.middle_block(hidden_states=h, temb=emb)
        if self.is_text or self.is_img:
            # 应用条件融合
            h = self.cond_fusion_mid(h, text_features, current_img_features)
        h = self.middle_block(hidden_states=h, temb=emb)
        

        # Additional residual connections for ControlNets
        if mid_block_additional_residual is not None:
            h = h + mid_block_additional_residual

        # 5. up blocks
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]
            h = upsample_block(hidden_states=h, res_hidden_states_list=res_samples, temb=emb)

        # 6. output block
        output: torch.Tensor = self.out(h)

        return output