from __future__ import annotations
from collections.abc import Sequence
import torch
from torch import nn

from .Blocks.module import zero_module
from .Blocks.time_embedding import get_timestep_embedding
from .Blocks.unet_block import get_down_block, get_mid_block, get_up_block

from monai.networks.blocks import Convolution



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
        down_block_additional_residuals: tuple[torch.Tensor] | None = None,
        mid_block_additional_residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: input tensor (N, C, SpatialDims).
            timesteps: timestep tensor (N,).
            down_block_additional_residuals: additional residual tensors for down blocks.
            mid_block_additional_residual: additional residual tensor for mid block.
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