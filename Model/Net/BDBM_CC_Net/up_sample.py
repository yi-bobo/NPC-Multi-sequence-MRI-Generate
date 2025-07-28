import torch
import torch.nn as nn
from monai.networks.blocks import Convolution, SpatialAttentionBlock, Upsample

from Model.Net.BDBM_CC_Net.resnet import DiffusionUNetResnetBlock, ConditionalUNetResnetBlock
from Model.Net.BDBM_CC_Net.transformer_block import SpatialTransformer

class WrappedUpsample(Upsample):
    """
    上采样算子包装类，主要目的是：
      1. 统一接口：支持接收多余的 emb 参数（如时间/条件嵌入），与其它模块保持一致。
      2. 委托给父类 Upsample 完成实际的插值放大操作。
    """

    def forward(self,
                x: torch.Tensor,
                emb: torch.Tensor | None = None
               ) -> torch.Tensor:
        # 1️⃣ 接收但忽略 emb 参数，以兼容调用接口
        del emb

        # 2️⃣ 调用父类的 forward 方法，执行真正的上采样
        #    Upsample 会根据初始化时的 scale_factor、mode、align_corners 等配置完成插值
        upsampled: torch.Tensor = super().forward(x)

        # 3️⃣ 返回上采样后的特征图
        return upsampled
    
class ConditionalWrappedUpsample(Upsample):
    """
    上采样算子包装类，主要目的是：
      1. 统一接口：支持接收多余的 emb 参数（如时间/条件嵌入），与其它模块保持一致。
      2. 委托给父类 Upsample 完成实际的插值放大操作。
    """

    def forward(self,
                x: torch.Tensor,
               ) -> torch.Tensor:
        # 1️⃣ 调用父类的 forward 方法，执行真正的上采样
        #    Upsample 会根据初始化时的 scale_factor、mode、align_corners 等配置完成插值
        upsampled: torch.Tensor = super().forward(x)

        # 2️⃣ 返回上采样后的特征图
        return upsampled

class UpBlock(nn.Module):
    """
    U-Net 上采样块：在 Decoder 阶段将低分辨率特征与高分辨率跳跃连接特征融合，
    并可选地做上采样（nearest + optional post-conv 或 ResNet-based upsample）。

    结构 (num_res_blocks 次)：
      1. 从 res_hidden_states_list 弹出跳跃连接特征
      2. 与当前 hidden_states 在通道维拼接
      3. 通过 DiffusionUNetResnetBlock 做融合
    末尾可选：
      4. 上采样（WrappedUpsample 或 ResNetBlock(up=True)）
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_upsample: bool = True,
        resblock_updown: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims:         空间维度 (2 或 3)
            in_channels:          跳跃连接特征的通道数
            prev_output_channel:  上一层输出特征的通道数
            out_channels:         本层输出特征的通道数
            temb_channels:        时间/条件嵌入的维度
            num_res_blocks:       融合块的数量 (拼接 + ResNet 的重复次数)
            norm_num_groups:      GroupNorm 组数
            norm_eps:             GroupNorm eps
            add_upsample:         是否在末尾添加上采样
            resblock_updown:      若 True，使用 ResNetBlock(up=True) 做上采样；否则用 WrappedUpsample
        """
        super().__init__()
        self.resblock_updown = resblock_updown

        # 构建多个 ResNet 融合块 (拼接后通道 = prev_output_channel + in_channels 或 out_channels + in_channels)
        resnets = []
        for i in range(num_res_blocks):
            # 最后一层使用原始跳跃通道数 in_channels；中间层都以 out_channels 作为跳跃通道
            res_skip_channels = in_channels if (i == num_res_blocks - 1) else out_channels
            # 输入给第一个 ResNet 的通道是 prev_output_channel；后续都是 out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            # 融合通道为 resnet_in_channels + res_skip_channels
            resnets.append(
                DiffusionUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )
        # 注册所有 ResNet Block
        self.resnets = nn.ModuleList(resnets)

        # 上采样模块，可选
        if add_upsample:
            if resblock_updown:
                # 用 ResNetBlock(up=True) 进行可学习上采样
                self.upsampler = DiffusionUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    up=True,  # 指定为上采样模式
                )
            else:
                # 用 WrappedUpsample (nearest) + post_conv (3×3) 的方式上采样
                post_conv = Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    strides=1,
                    kernel_size=3,
                    padding=1,
                    conv_only=True,
                )
                self.upsampler = WrappedUpsample(
                    spatial_dims=spatial_dims,
                    mode="nontrainable",       # 非训练型插值
                    in_channels=out_channels,
                    out_channels=out_channels,
                    interp_mode="nearest",     # 最近邻插值
                    scale_factor=2.0,          # 放大两倍
                    post_conv=post_conv,       # 插值后跟一个 3×3 卷积
                    align_corners=None,
                )
        else:
            self.upsampler = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_list: list[torch.Tensor],
        temb: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states:          当前低分辨率特征 [B, prev_output_channel, ...]
            res_hidden_states_list: 跳跃连接特征列表，最后一个是最靠近当前的高分辨率特征
            temb:                   时间/条件嵌入，传递给 ResNet Block
            context:                未使用，接口兼容
        Returns:
            hidden_states: 输出高分辨率特征 [B, out_channels, ...]
        """
        # 丢弃未使用的 context
        del context

        # 1) 依次弹出对应步的跳跃连接，拼接并通过 ResNet 融合
        for resnet in self.resnets:
            # 从列表尾部取出最近的跳跃特征
            res_hidden_states = res_hidden_states_list.pop()
            # 通道拼接
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            # 通过 ResNet Block，注入时间嵌入
            hidden_states = resnet(hidden_states, temb)

        # 2) 可选上采样
        if self.upsampler is not None:
            hidden_states = self.upsampler(hidden_states, temb)

        return hidden_states

class ConditionalUpBlock(nn.Module):
    """
    U-Net 上采样块：在 Decoder 阶段将低分辨率特征与高分辨率跳跃连接特征融合，
    并可选地做上采样（nearest + optional post-conv 或 ResNet-based upsample）。

    结构 (num_res_blocks 次)：
      1. 从 res_hidden_states_list 弹出跳跃连接特征
      2. 与当前 hidden_states 在通道维拼接
      3. 通过 DiffusionUNetResnetBlock 做融合
    末尾可选：
      4. 上采样（WrappedUpsample 或 ResNetBlock(up=True)）
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_upsample: bool = True,
        resblock_updown: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims:         空间维度 (2 或 3)
            in_channels:          跳跃连接特征的通道数
            prev_output_channel:  上一层输出特征的通道数
            out_channels:         本层输出特征的通道数
            num_res_blocks:       融合块的数量 (拼接 + ResNet 的重复次数)
            norm_num_groups:      GroupNorm 组数
            norm_eps:             GroupNorm eps
            add_upsample:         是否在末尾添加上采样
            resblock_updown:      若 True，使用 ResNetBlock(up=True) 做上采样；否则用 WrappedUpsample
        """
        super().__init__()
        self.resblock_updown = resblock_updown

        # 构建多个 ResNet 融合块 (拼接后通道 = prev_output_channel + in_channels 或 out_channels + in_channels)
        resnets = []
        for i in range(num_res_blocks):
            # 最后一层使用原始跳跃通道数 in_channels；中间层都以 out_channels 作为跳跃通道
            res_skip_channels = in_channels if (i == num_res_blocks - 1) else out_channels
            # 输入给第一个 ResNet 的通道是 prev_output_channel；后续都是 out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            # 融合通道为 resnet_in_channels + res_skip_channels
            resnets.append(
                ConditionalUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )
        # 注册所有 ResNet Block
        self.resnets = nn.ModuleList(resnets)

        # 上采样模块，可选
        if add_upsample:
            if resblock_updown:
                # 用 ResNetBlock(up=True) 进行可学习上采样
                self.upsampler = ConditionalUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    up=True,  # 指定为上采样模式
                )
            else:
                # 用 WrappedUpsample (nearest) + post_conv (3×3) 的方式上采样
                post_conv = Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    strides=1,
                    kernel_size=3,
                    padding=1,
                    conv_only=True,
                )
                self.upsampler = ConditionalWrappedUpsample(
                    spatial_dims=spatial_dims,
                    mode="nontrainable",       # 非训练型插值
                    in_channels=out_channels,
                    out_channels=out_channels,
                    interp_mode="nearest",     # 最近邻插值
                    scale_factor=2.0,          # 放大两倍
                    post_conv=post_conv,       # 插值后跟一个 3×3 卷积
                    align_corners=None,
                )
        else:
            self.upsampler = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_list: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            hidden_states:          当前低分辨率特征 [B, prev_output_channel, ...]
            res_hidden_states_list: 跳跃连接特征列表，最后一个是最靠近当前的高分辨率特征
            temb:                   时间/条件嵌入，传递给 ResNet Block
        Returns:
            hidden_states: 输出高分辨率特征 [B, out_channels, ...]
        """

        # 1) 依次弹出对应步的跳跃连接，拼接并通过 ResNet 融合
        for resnet in self.resnets:
            # 从列表尾部取出最近的跳跃特征
            res_hidden_states = res_hidden_states_list.pop()
            # 通道拼接
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            # 通过 ResNet Block，注入时间嵌入
            hidden_states = resnet(hidden_states)

        # 2) 可选上采样
        if self.upsampler is not None:
            hidden_states = self.upsampler(hidden_states)

        return hidden_states

class AttnUpBlock(nn.Module):
    """
    带空间注意力的 U-Net Decoder 上采样块：
      - 多个 (拼接跳跃 + ResNetBlock + SpatialAttention) 序列
      - 可选上采样（nearest 插值 + post-conv 或 ResNet 上采样）
    用于将低分辨率特征与高分辨率跳跃连接特征融合，并逐步恢复空间分辨率。
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_upsample: bool = True,
        resblock_updown: bool = False,
        num_head_channels: list[int] = [8, 8, 8],
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims:         2 或 3，决定 Conv2d/Conv3d
            in_channels:          每次跳跃连接特征的通道数
            prev_output_channel:  前一层输出（解码器中）特征的通道数
            out_channels:         本层输出特征的通道数
            temb_channels:        时间/条件嵌入维度
            num_res_blocks:       ResNet+Attention 重复次数
            norm_num_groups:      GroupNorm 的组数
            norm_eps:             GroupNorm 的 epsilon
            add_upsample:         是否在末尾添加上采样
            resblock_updown:      True 使用 ResNetBlock(up=True) 上采样，否则使用 WrappedUpsample
            num_head_channels:    每个注意力头的通道数
            include_fc:           SpatialAttentionBlock 是否包含前馈层
            use_combined_linear:  SpatialAttentionBlock 是否合并 QKV 线性层
            use_flash_attention:  SpatialAttentionBlock 是否启用 FlashAttention
        """
        super().__init__()
        self.resblock_updown = resblock_updown
        num_head_channels = list(reversed(num_head_channels))

        # 1. 构造融合块列表：每个块 = 拼接跳跃连接 + ResNetBlock + SpatialAttention
        resnets = []
        attentions = []
        for i in range(num_res_blocks):
            # 最后一轮的跳跃连接通道数仍为 in_channels，其它轮为 out_channels
            res_skip_channels = in_channels if (i == num_res_blocks - 1) else out_channels
            # 第一轮输入通道 = prev_output_channel，其它轮 = out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            # 拼接后的通道数 = resnet_in_channels + res_skip_channels
            combined_in = resnet_in_channels + res_skip_channels

            # ResNetBlock：注入时间/条件嵌入后做卷积
            resnets.append(
                DiffusionUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=combined_in,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )
            # SpatialAttentionBlock：在空间维度分配注意力
            attentions.append(
                SpatialAttentionBlock(
                    spatial_dims=spatial_dims,
                    num_channels=out_channels,
                    num_head_channels=num_head_channels[i],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    include_fc=include_fc,
                    use_combined_linear=use_combined_linear,
                    use_flash_attention=use_flash_attention,
                )
            )

        # 注册模块列表
        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)

        # 2. 上采样模块：可选
        if add_upsample:
            if resblock_updown:
                # 使用可学习的 ResNetBlock(up=True) 实现上采样
                self.upsampler = DiffusionUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    up=True,  # 启用上采样功能
                )
            else:
                # 使用 WrappedUpsample + post_conv (3×3) 的插值上采样
                post_conv = Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    strides=1,
                    kernel_size=3,
                    padding=1,
                    conv_only=True,
                )
                self.upsampler = WrappedUpsample(
                    spatial_dims=spatial_dims,
                    mode="nontrainable",
                    in_channels=out_channels,
                    out_channels=out_channels,
                    interp_mode="nearest",
                    scale_factor=2.0,
                    post_conv=post_conv,
                    align_corners=None,
                )
        else:
            self.upsampler = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_list: list[torch.Tensor],
        temb: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states:          当前解码器特征 [B, prev_output_channel, ...]
            res_hidden_states_list: 跳跃连接特征列表（高分辨率），最后一项最靠近当前层
            temb:                   时间/条件嵌入
            context:                未使用，接口兼容
        Returns:
            hidden_states: [B, out_channels, ...] 融合并上采样后的特征
        """
        # 丢弃未用的 context 参数
        del context

        # 融合每个跳跃连接
        for resnet, attn in zip(self.resnets, self.attentions):
            # 弹出最近的跳跃特征
            res_skip = res_hidden_states_list.pop()
            # 通道拼接：低分辨率 + 对应高分辨率跳跃
            hidden_states = torch.cat([hidden_states, res_skip], dim=1)
            # ResNetBlock (注入 temb)
            hidden_states = resnet(hidden_states, temb)
            # SpatialAttentionBlock
            hidden_states = attn(hidden_states).contiguous()

        # 上采样（若配置）
        if self.upsampler is not None:
            hidden_states = self.upsampler(hidden_states, temb)

        return hidden_states

class ConditionalAttnUpBlock(nn.Module):
    """
    带空间注意力的 U-Net Decoder 上采样块：
      - 多个 (拼接跳跃 + ResNetBlock + SpatialAttention) 序列
      - 可选上采样（nearest 插值 + post-conv 或 ResNet 上采样）
    用于将低分辨率特征与高分辨率跳跃连接特征融合，并逐步恢复空间分辨率。
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_upsample: bool = True,
        resblock_updown: bool = False,
        num_head_channels: int = 1,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims:         2 或 3，决定 Conv2d/Conv3d
            in_channels:          每次跳跃连接特征的通道数
            prev_output_channel:  前一层输出（解码器中）特征的通道数
            out_channels:         本层输出特征的通道数
            num_res_blocks:       ResNet+Attention 重复次数
            norm_num_groups:      GroupNorm 的组数
            norm_eps:             GroupNorm 的 epsilon
            add_upsample:         是否在末尾添加上采样
            resblock_updown:      True 使用 ResNetBlock(up=True) 上采样，否则使用 WrappedUpsample
            num_head_channels:    每个注意力头的通道数
            include_fc:           SpatialAttentionBlock 是否包含前馈层
            use_combined_linear:  SpatialAttentionBlock 是否合并 QKV 线性层
            use_flash_attention:  SpatialAttentionBlock 是否启用 FlashAttention
        """
        super().__init__()
        self.resblock_updown = resblock_updown

        # 1. 构造融合块列表：每个块 = 拼接跳跃连接 + ResNetBlock + SpatialAttention
        resnets = []
        attentions = []
        for i in range(num_res_blocks):
            # 最后一轮的跳跃连接通道数仍为 in_channels，其它轮为 out_channels
            res_skip_channels = in_channels if (i == num_res_blocks - 1) else out_channels
            # 第一轮输入通道 = prev_output_channel，其它轮 = out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            # 拼接后的通道数 = resnet_in_channels + res_skip_channels
            combined_in = resnet_in_channels + res_skip_channels

            # ResNetBlock：注入时间/条件嵌入后做卷积
            resnets.append(
                ConditionalUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=combined_in,
                    out_channels=out_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )
            # SpatialAttentionBlock：在空间维度分配注意力
            attentions.append(
                SpatialAttentionBlock(
                    spatial_dims=spatial_dims,
                    num_channels=out_channels,
                    num_head_channels=num_head_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    include_fc=include_fc,
                    use_combined_linear=use_combined_linear,
                    use_flash_attention=use_flash_attention,
                )
            )

        # 注册模块列表
        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)

        # 2. 上采样模块：可选
        if add_upsample:
            if resblock_updown:
                # 使用可学习的 ResNetBlock(up=True) 实现上采样
                self.upsampler = ConditionalUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    up=True,  # 启用上采样功能
                )
            else:
                # 使用 WrappedUpsample + post_conv (3×3) 的插值上采样
                post_conv = Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    strides=1,
                    kernel_size=3,
                    padding=1,
                    conv_only=True,
                )
                self.upsampler = ConditionalWrappedUpsample(
                    spatial_dims=spatial_dims,
                    mode="nontrainable",
                    in_channels=out_channels,
                    out_channels=out_channels,
                    interp_mode="nearest",
                    scale_factor=2.0,
                    post_conv=post_conv,
                    align_corners=None,
                )
        else:
            self.upsampler = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_list: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            hidden_states:          当前解码器特征 [B, prev_output_channel, ...]
            res_hidden_states_list: 跳跃连接特征列表（高分辨率），最后一项最靠近当前层
        Returns:
            hidden_states: [B, out_channels, ...] 融合并上采样后的特征
        """

        # 融合每个跳跃连接
        for resnet, attn in zip(self.resnets, self.attentions):
            # 弹出最近的跳跃特征
            res_skip = res_hidden_states_list.pop()
            # 通道拼接：低分辨率 + 对应高分辨率跳跃
            hidden_states = torch.cat([hidden_states, res_skip], dim=1)
            # ResNetBlock (注入 temb)
            hidden_states = resnet(hidden_states)
            # SpatialAttentionBlock
            hidden_states = attn(hidden_states).contiguous()

        # 上采样（若配置）
        if self.upsampler is not None:
            hidden_states = self.upsampler(hidden_states)

        return hidden_states
    
class CrossAttnUpBlock(nn.Module):
    """
    带跨注意力 Transformer 的 U-Net Decoder 上采样块：
      - 融合拼接式跳跃连接 + ResNetBlock + SpatialTransformer (Cross-Attention)
      - 可选在末尾做上采样（nearest 插值 + post-conv 或 ResNet-based upsample）
    用于将低分辨率特征与高分辨率跳跃连接特征融合，并在跨注意力机制下恢复空间分辨率。
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_upsample: bool = True,
        resblock_updown: bool = False,
        num_head_channels: int = 1,
        transformer_num_layers: int = 1,
        cross_attention_dim: int | None = None,
        upcast_attention: bool = False,
        dropout_cattn: float = 0.0,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims:          空间维度 (2 或 3)
            in_channels:           跳跃连接特征通道数
            prev_output_channel:   前一层解码输出通道数
            out_channels:          本层输出通道数
            temb_channels:         时间/条件嵌入通道数
            num_res_blocks:        融合块 (拼接+ResNet+Attention) 的数量
            norm_num_groups:       GroupNorm 的组数
            norm_eps:              GroupNorm epsilon
            add_upsample:          是否在末尾添加上采样
            resblock_updown:       若 True，用 ResNetBlock(up=True) 上采样；否则用 WrappedUpsample
            num_head_channels:     每个注意力头的通道数
            transformer_num_layers:SpatialTransformer 中 TransformerBlock 的层数
            cross_attention_dim:   跨注意力上下文维度 (None 则仅自注意力)
            upcast_attention:      是否在注意力计算中使用更高精度 (float32)
            dropout_cattn:         注意力层 dropout 比例
            include_fc:            SpatialTransformer 是否包含前馈层
            use_combined_linear:   SpatialTransformer 是否合并 QKV 线性层
            use_flash_attention:   SpatialTransformer 是否启用 FlashAttention
        """
        super().__init__()
        self.resblock_updown = resblock_updown

        # 1) 构建多个 (拼接跳跃 → ResNetBlock → SpatialTransformer) 序列
        resnets = []
        attentions = []
        for i in range(num_res_blocks):
            # 最后一轮跳跃连接通道数为 in_channels，其余轮为 out_channels
            skip_ch = in_channels if i == num_res_blocks - 1 else out_channels
            # 第一轮输入通道 = prev_output_channel，其余轮 = out_channels
            in_ch = prev_output_channel if i == 0 else out_channels
            # 拼接后通道 = in_ch + skip_ch
            combined_ch = in_ch + skip_ch

            # ResNet 融合块：注入时间嵌入 temb，再做两次 3×3 卷积
            resnets.append(
                DiffusionUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=combined_ch,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )
            # 跨注意力 Transformer：在空间上做自/交叉注意力
            attentions.append(
                SpatialTransformer(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    num_attention_heads=out_channels // num_head_channels,
                    num_head_channels=num_head_channels,
                    num_layers=transformer_num_layers,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    cross_attention_dim=cross_attention_dim,
                    upcast_attention=upcast_attention,
                    dropout=dropout_cattn,
                    include_fc=include_fc,
                    use_combined_linear=use_combined_linear,
                    use_flash_attention=use_flash_attention,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)

        # 2) 可选上采样模块
        if add_upsample:
            if resblock_updown:
                # 使用 ResNetBlock(up=True) 可学习上采样
                self.upsampler = DiffusionUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    up=True,
                )
            else:
                # 使用 WrappedUpsample (nearest + post_conv) 的方式上采样
                post_conv = Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    strides=1,
                    kernel_size=3,
                    padding=1,
                    conv_only=True,
                )
                self.upsampler = WrappedUpsample(
                    spatial_dims=spatial_dims,
                    mode="nontrainable",
                    in_channels=out_channels,
                    out_channels=out_channels,
                    interp_mode="nearest",
                    scale_factor=2.0,
                    post_conv=post_conv,
                    align_corners=None,
                )
        else:
            self.upsampler = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_list: list[torch.Tensor],
        temb: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states:          当前解码器输入 [B, prev_output_channel, ...]
            res_hidden_states_list: 跳跃连接特征列表，最后一项是最靠近当前层
            temb:                   时间/条件嵌入 [B, temb_channels]
            context:                跨注意力上下文 (可选)
        Returns:
            hidden_states: 融合后并上采样的特征 [B, out_channels, ...]
        """
        # 依次融合每个跳跃连接
        for resnet, attn in zip(self.resnets, self.attentions):
            # 从尾部弹出对应的高分辨率跳跃特征
            skip_feat = res_hidden_states_list.pop()
            # 通道拼接：低分辨率 + 高分辨率跳跃
            hidden_states = torch.cat([hidden_states, skip_feat], dim=1)
            # ResNet 块融合 (注入 temb)
            hidden_states = resnet(hidden_states, temb)
            # 空间跨注意力融合
            hidden_states = attn(hidden_states, context=context)

        # 上采样 (若配置)
        if self.upsampler is not None:
            hidden_states = self.upsampler(hidden_states, temb)

        return hidden_states

class ConditionalCrossAttnUpBlock(nn.Module):
    """
    带跨注意力 Transformer 的 U-Net Decoder 上采样块：
      - 融合拼接式跳跃连接 + ResNetBlock + SpatialTransformer (Cross-Attention)
      - 可选在末尾做上采样（nearest 插值 + post-conv 或 ResNet-based upsample）
    用于将低分辨率特征与高分辨率跳跃连接特征融合，并在跨注意力机制下恢复空间分辨率。
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_upsample: bool = True,
        resblock_updown: bool = False,
        num_head_channels: int = 1,
        transformer_num_layers: int = 1,
        cross_attention_dim: int | None = None,
        upcast_attention: bool = False,
        dropout_cattn: float = 0.0,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims:          空间维度 (2 或 3)
            in_channels:           跳跃连接特征通道数
            prev_output_channel:   前一层解码输出通道数
            out_channels:          本层输出通道数
            num_res_blocks:        融合块 (拼接+ResNet+Attention) 的数量
            norm_num_groups:       GroupNorm 的组数
            norm_eps:              GroupNorm epsilon
            add_upsample:          是否在末尾添加上采样
            resblock_updown:       若 True，用 ResNetBlock(up=True) 上采样；否则用 WrappedUpsample
            num_head_channels:     每个注意力头的通道数
            transformer_num_layers:SpatialTransformer 中 TransformerBlock 的层数
            cross_attention_dim:   跨注意力上下文维度 (None 则仅自注意力)
            upcast_attention:      是否在注意力计算中使用更高精度 (float32)
            dropout_cattn:         注意力层 dropout 比例
            include_fc:            SpatialTransformer 是否包含前馈层
            use_combined_linear:   SpatialTransformer 是否合并 QKV 线性层
            use_flash_attention:   SpatialTransformer 是否启用 FlashAttention
        """
        super().__init__()
        self.resblock_updown = resblock_updown

        # 1) 构建多个 (拼接跳跃 → ResNetBlock → SpatialTransformer) 序列
        resnets = []
        attentions = []
        for i in range(num_res_blocks):
            # 最后一轮跳跃连接通道数为 in_channels，其余轮为 out_channels
            skip_ch = in_channels if i == num_res_blocks - 1 else out_channels
            # 第一轮输入通道 = prev_output_channel，其余轮 = out_channels
            in_ch = prev_output_channel if i == 0 else out_channels
            # 拼接后通道 = in_ch + skip_ch
            combined_ch = in_ch + skip_ch

            # ResNet 融合块：注入时间嵌入 temb，再做两次 3×3 卷积
            resnets.append(
                ConditionalUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=combined_ch,
                    out_channels=out_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )
            # 跨注意力 Transformer：在空间上做自/交叉注意力
            attentions.append(
                SpatialTransformer(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    num_attention_heads=out_channels // num_head_channels,
                    num_head_channels=num_head_channels,
                    num_layers=transformer_num_layers,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    cross_attention_dim=cross_attention_dim,
                    upcast_attention=upcast_attention,
                    dropout=dropout_cattn,
                    include_fc=include_fc,
                    use_combined_linear=use_combined_linear,
                    use_flash_attention=use_flash_attention,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)

        # 2) 可选上采样模块
        if add_upsample:
            if resblock_updown:
                # 使用 ResNetBlock(up=True) 可学习上采样
                self.upsampler = ConditionalUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    up=True,
                )
            else:
                # 使用 WrappedUpsample (nearest + post_conv) 的方式上采样
                post_conv = Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    strides=1,
                    kernel_size=3,
                    padding=1,
                    conv_only=True,
                )
                self.upsampler = ConditionalWrappedUpsample(
                    spatial_dims=spatial_dims,
                    mode="nontrainable",
                    in_channels=out_channels,
                    out_channels=out_channels,
                    interp_mode="nearest",
                    scale_factor=2.0,
                    post_conv=post_conv,
                    align_corners=None,
                )
        else:
            self.upsampler = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_list: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            hidden_states:          当前解码器输入 [B, prev_output_channel, ...]
            res_hidden_states_list: 跳跃连接特征列表，最后一项是最靠近当前层
            temb:                   时间/条件嵌入 [B, temb_channels]
            context:                跨注意力上下文 (可选)
        Returns:
            hidden_states: 融合后并上采样的特征 [B, out_channels, ...]
        """
        # 依次融合每个跳跃连接
        for resnet, attn in zip(self.resnets, self.attentions):
            # 从尾部弹出对应的高分辨率跳跃特征
            skip_feat = res_hidden_states_list.pop()
            # 通道拼接：低分辨率 + 高分辨率跳跃
            hidden_states = torch.cat([hidden_states, skip_feat], dim=1)
            # ResNet 块融合 (注入 temb)
            hidden_states = resnet(hidden_states)
            # 空间跨注意力融合
            hidden_states = attn(hidden_states)

        # 上采样 (若配置)
        if self.upsampler is not None:
            hidden_states = self.upsampler(hidden_states)

        return hidden_states