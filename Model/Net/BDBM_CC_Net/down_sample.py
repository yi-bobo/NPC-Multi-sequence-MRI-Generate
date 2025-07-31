
import torch
import torch.nn as nn
from monai.networks.layers.factories import Pool
from monai.networks.blocks import Convolution, SpatialAttentionBlock

from Model.Net.BDBM_CC_Net.resnet import DiffusionUNetResnetBlock, ConditionalUNetResnetBlock
from Model.Net.BDBM_CC_Net.transformer_block import SpatialTransformer

class DiffusionUnetDownsample(nn.Module):
    def __init__(
        self, spatial_dims: int, num_channels: int, use_conv: bool, out_channels: int | None = None, padding: int = 1
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.out_channels = out_channels or num_channels
        self.use_conv = use_conv
        if use_conv:
            self.op = Convolution(
                spatial_dims=spatial_dims,
                in_channels=self.num_channels,
                out_channels=self.out_channels,
                strides=2,
                kernel_size=3,
                padding=padding,
                conv_only=True,
            )
        else:
            if self.num_channels != self.out_channels:
                raise ValueError("num_channels and out_channels must be equal when use_conv=False")
            self.op = Pool[Pool.AVG, spatial_dims](kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor, emb: torch.Tensor | None = None) -> torch.Tensor:
        del emb
        if x.shape[1] != self.num_channels:
            raise ValueError(
                f"Input number of channels ({x.shape[1]}) is not equal to expected number of channels "
                f"({self.num_channels})"
            )
        output: torch.Tensor = self.op(x)
        return output
    
class Downsample(nn.Module):
    def __init__(
        self, spatial_dims: int, 
        in_channels: int,
        out_channels: int | None = None,  
        use_conv: bool = True,
        padding: int = 1
    ) -> None:
        super().__init__()
        # 1. 保存输入参数
        self.in_channels = in_channels 
        self.out_channels = out_channels or in_channels
        self.use_conv = use_conv

        # 2. 根据 use_conv 决定下采样方式
        if use_conv:
            self.op = Convolution(
                spatial_dims=spatial_dims,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                strides=2,
                kernel_size=3,
                padding=padding,
                conv_only=True,
            )
        else:
            # 2b. 用平均池化实现下采样--仅当 in/out 通道一致时才允许，不一致时报错
            if self.in_channels != self.out_channels:
                raise ValueError(
                    "in_channels and out_channels must be equal when use_conv=False"
                )
            # Pool.AVG 表示平均池化，kernel=2, stride=2 即空间尺寸减半
            self.op = Pool[Pool.AVG, spatial_dims](kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # 1. 检查输入通道是否合法
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Input channels ({x.shape[1]}) != expected ({self.in_channels})"
            )
        # 2. 调用 op（卷积或池化）完成下采样
        output: torch.Tensor = self.op(x)
        return output


class DownBlock(nn.Module):
    """
    U-Net Encoder 中的降采样块：
      - 多个连续的 ResNet Block（不改变分辨率）
      - 可选的下采样操作（卷积或 ResNetBlock 下采样）
      - 返回：当前特征 + 各中间特征列表（用于跳跃连接）
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_downsample: bool = True,
        resblock_updown: bool = False,
        downsample_padding: int = 1,
    ) -> None:
        """
        Args:
            spatial_dims:      2 或 3，决定使用 2D/3D 操作
            in_channels:       输入特征通道数
            out_channels:      输出特征通道数（ResNet Block、下采样后通道）
            temb_channels:     时间/条件嵌入的通道数
            num_res_blocks:    ResNet Block 的数量（逐层堆叠）
            norm_num_groups:   GroupNorm 的组数
            norm_eps:          GroupNorm 的 epsilon
            add_downsample:    是否在末尾添加下采样
            resblock_updown:   如果 True，用 ResNetBlock(down=True) 做下采样；否则用 Convolution/Pool
            downsample_padding:卷积下采样时的 padding（仅对 use_conv 下采样有效）
        """
        super().__init__()
        self.resblock_updown = resblock_updown

        # 1️⃣ 构建多个 ResNet Block
        resnets = []
        for i in range(num_res_blocks):
            # 第一层输入通道为 in_channels，其余层输入通道为 out_channels
            block_in_ch = in_channels if i == 0 else out_channels
            resnets.append(
                DiffusionUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=block_in_ch,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )
        self.resnets = nn.ModuleList(resnets)

        # 2️⃣ 是否只在 H 和 W 维度下采样
        if add_downsample:
            if resblock_updown:
                # 使用可学习的 ResNetBlock(down=True) 实现下采样
                self.downsampler = DiffusionUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    down=True,  # 指定为下采样模式
                )
            else:
                # 使用 Convolution (stride=2) 或 Pool 实现下采样
                self.downsampler = DiffusionUnetDownsample(
                    spatial_dims=spatial_dims,
                    num_channels=out_channels,
                    use_conv=True,
                    out_channels=out_channels,
                    padding=downsample_padding,
                )
        else:
            # 只在H和W维度下采样
            self.downsampler = DiffusionUnetDownsample(
                    spatial_dims=spatial_dims,
                    num_channels=out_channels,
                    use_conv=True,
                    out_channels=out_channels,
                    strides=(1, 2, 2),  # 只在 H 和 W 维度下采样
                    padding=downsample_padding,
                )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            hidden_states: [B, C, ...] 输入特征图
            temb:          [B, temb_channels] 时间/条件嵌入
            context:       未使用，仅为接口兼容
        Returns:
            hidden_states_out: [B, out_channels, ...] 最终特征图（空间尺寸可能减半）
            output_states:     List 中包含每个 ResNet Block 后 & 下采样后（若有）的特征
        """
        # 丢弃未使用的 context 参数
        del context

        output_states = []

        # 1️⃣ 依次通过 ResNet Blocks，并收集每层输出
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            output_states.append(hidden_states)

        # 2️⃣ 下采样（若配置）
        if self.downsampler is not None:
            hidden_states = self.downsampler(hidden_states, temb)
            output_states.append(hidden_states)

        # 返回最后的 hidden_states 及所有中间特征列表
        return hidden_states, output_states
    
class ConditionalDownBlock(nn.Module):
    """
    U-Net Encoder 中的降采样块：
      - 多个连续的 ResNet Block（不改变分辨率）
      - 可选的下采样操作（卷积或 ResNetBlock 下采样）
      - 返回：当前特征 + 各中间特征列表（用于跳跃连接）
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_downsample: bool = True,
        resblock_updown: bool = False,
        downsample_padding: int = 1,
    ) -> None:
        """
        Args:
            spatial_dims:      2 或 3，决定使用 2D/3D 操作
            in_channels:       输入特征通道数
            out_channels:      输出特征通道数（ResNet Block、下采样后通道）
            num_res_blocks:    ResNet Block 的数量（逐层堆叠）
            norm_num_groups:   GroupNorm 的组数
            norm_eps:          GroupNorm 的 epsilon
            add_downsample:    是否在末尾添加下采样
            resblock_updown:   如果 True，用 ResNetBlock(down=True) 做下采样；否则用 Convolution/Pool
            downsample_padding:卷积下采样时的 padding（仅对 use_conv 下采样有效）
        """
        super().__init__()
        self.resblock_updown = resblock_updown

        # 1️⃣ 构建多个 ResNet Block
        resnets = []
        for i in range(num_res_blocks):
            # 第一层输入通道为 in_channels，其余层输入通道为 out_channels
            block_in_ch = in_channels if i == 0 else out_channels
            resnets.append(
                ConditionalUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=block_in_ch,
                    out_channels=out_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )
        self.resnets = nn.ModuleList(resnets)

        # 2️⃣ 可选下采样模块
        if add_downsample:
            if resblock_updown:
                # 使用可学习的 ResNetBlock(down=True) 实现下采样
                self.downsampler = ConditionalUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    down=True,  # 指定为下采样模式
                )
            else:
                # 使用 Convolution (stride=2) 或 Pool 实现下采样
                self.downsampler = ConditionalUnetDownsample(
                    spatial_dims=spatial_dims,
                    num_channels=out_channels,
                    use_conv=True,
                    out_channels=out_channels,
                    padding=downsample_padding,
                )
        else:
            # 不做下采样
            self.downsampler = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            hidden_states: [B, C, ...] 输入特征图
            context:       未使用，仅为接口兼容
        Returns:
            hidden_states_out: [B, out_channels, ...] 最终特征图（空间尺寸可能减半）
            output_states:     List 中包含每个 ResNet Block 后 & 下采样后（若有）的特征
        """
        # 丢弃未使用的 context 参数
        del context

        output_states = []

        # 1️⃣ 依次通过 ResNet Blocks，并收集每层输出
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
            output_states.append(hidden_states)

        # 2️⃣ 下采样（若配置）
        if self.downsampler is not None:
            hidden_states = self.downsampler(hidden_states)
            output_states.append(hidden_states)

        # 返回最后的 hidden_states 及所有中间特征列表
        return hidden_states, output_states

class DiffusionUnetDownsample(nn.Module):
    """
    Diffusion U-Net 中的下采样模块。
    - 可选用卷积（stride=2 + padding）或平均池化来实现空间尺寸减半。
    - 在 use_conv=True 时，可同时改变通道数；否则通道数保持不变。
    """

    def __init__(
        self,
        spatial_dims: int,
        num_channels: int,
        use_conv: bool,
        out_channels: int | None = None,
        strides: tuple[int, ...] = (2, 2, 2),
        padding: int = 1
    ) -> None:
        """
        Args:
            spatial_dims: 2 表示 2D，下采样 H×W；3 表示 3D，下采样 D×H×W。
            num_channels: 输入特征图的通道数。
            use_conv:      是否使用卷积实现下采样；否则使用平均池化。
            out_channels:  下采样后输出通道数；若为 None，则保持与 num_channels 一致。
            strides:         下采样的步幅，默认为 (2, 2, 2)，表示在所有空间维度下采样; (1,2,2) 表示只在 H 和 W 维度下采样。
            padding:       卷积时的 padding，通常设为 kernel_size//2 以居中感受野。
        """
        super().__init__()
        # 保存输入的通道数和输出的通道数
        self.num_channels = num_channels
        self.out_channels = out_channels or num_channels
        self.use_conv = use_conv

        if use_conv:
            # 1) 使用 3×3 卷积 + stride=2 实现下采样
            #    - conv_only=True：只保留卷积算子，不带归一化/激活
            #    - 通过 in_channels/out_channels 可以在下采样时改变通道数
            self.op = Convolution(
                spatial_dims=spatial_dims,
                in_channels=self.num_channels,
                out_channels=self.out_channels,
                strides=strides,
                kernel_size=3,
                padding=padding,
                conv_only=True,
            )
        else:
            # 2) 使用平均池化实现下采样（kernel=2, stride=2），要求通道数不变
            if self.num_channels != self.out_channels:
                raise ValueError(
                    "num_channels and out_channels must be equal when use_conv=False"
                )
            # Pool[Pool.AVG, spatial_dims] 会根据 spatial_dims 自动选择 AvgPool2d 或 AvgPool3d
            self.op = Pool[Pool.AVG, spatial_dims](
                kernel_size=2, stride=2
            )

    def forward(self, x: torch.Tensor, emb: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x:   输入特征图，shape = [B, C, ...]，C 应等于 num_channels
            emb: 时间/条件嵌入（不使用，仅为接口兼容）
        Returns:
            output: 下采样后的特征图，shape = [B, out_channels, ...]，空间尺寸减半
        """
        # 丢弃多余的 emb 参数，保持与其它模块 forward 接口一致
        del emb

        # 检查输入通道数是否与初始化时一致，防止误用
        if x.shape[1] != self.num_channels:
            raise ValueError(
                f"Input number of channels ({x.shape[1]}) "
                f"is not equal to expected number ({self.num_channels})"
            )

        # 调用选定的下采样算子（卷积或池化）
        output: torch.Tensor = self.op(x)
        return output

class ConditionalUnetDownsample(nn.Module):
    """
    Diffusion U-Net 中的下采样模块。
    - 可选用卷积（stride=2 + padding）或平均池化来实现空间尺寸减半。
    - 在 use_conv=True 时，可同时改变通道数；否则通道数保持不变。
    """

    def __init__(
        self,
        spatial_dims: int,
        num_channels: int,
        use_conv: bool,
        out_channels: int | None = None,
        padding: int = 1
    ) -> None:
        """
        Args:
            spatial_dims: 2 表示 2D，下采样 H×W；3 表示 3D，下采样 D×H×W。
            num_channels: 输入特征图的通道数。
            use_conv:      是否使用卷积实现下采样；否则使用平均池化。
            out_channels:  下采样后输出通道数；若为 None，则保持与 num_channels 一致。
            padding:       卷积时的 padding，通常设为 kernel_size//2 以居中感受野。
        """
        super().__init__()
        # 保存输入的通道数和输出的通道数
        self.num_channels = num_channels
        self.out_channels = out_channels or num_channels
        self.use_conv = use_conv

        if use_conv:
            # 1) 使用 3×3 卷积 + stride=2 实现下采样
            #    - conv_only=True：只保留卷积算子，不带归一化/激活
            #    - 通过 in_channels/out_channels 可以在下采样时改变通道数
            self.op = Convolution(
                spatial_dims=spatial_dims,
                in_channels=self.num_channels,
                out_channels=self.out_channels,
                strides=2,
                kernel_size=3,
                padding=padding,
                conv_only=True,
            )
        else:
            # 2) 使用平均池化实现下采样（kernel=2, stride=2），要求通道数不变
            if self.num_channels != self.out_channels:
                raise ValueError(
                    "num_channels and out_channels must be equal when use_conv=False"
                )
            # Pool[Pool.AVG, spatial_dims] 会根据 spatial_dims 自动选择 AvgPool2d 或 AvgPool3d
            self.op = Pool[Pool.AVG, spatial_dims](
                kernel_size=2, stride=2
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   输入特征图，shape = [B, C, ...]，C 应等于 num_channels
        Returns:
            output: 下采样后的特征图，shape = [B, out_channels, ...]，空间尺寸减半
        """
        # 检查输入通道数是否与初始化时一致，防止误用
        if x.shape[1] != self.num_channels:
            raise ValueError(
                f"Input number of channels ({x.shape[1]}) "
                f"is not equal to expected number ({self.num_channels})"
            )

        # 调用选定的下采样算子（卷积或池化）
        output: torch.Tensor = self.op(x)
        return output

class AttnDownBlock(nn.Module):
    """
    包含 ResNet 与空间注意力（Spatial Attention）模块的下采样块。
    结构：
      - 若干个 (ResNet Block → Spatial Attention) 串联
      - 可选一个下采样（卷积或 ResNet-based）
    用于 Diffusion U-Net 的降分辨率阶段，同时保留多尺度特征。
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_downsample: bool = True,
        resblock_updown: bool = False,
        downsample_padding: int = 1,
        num_head_channels: int = 1,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims:         2 或 3，决定使用 2D/3D 操作
            in_channels:          输入通道数
            out_channels:         输出通道数（各子模块通道统一）
            temb_channels:        时间嵌入维度，用于 ResNet Block
            num_res_blocks:       每个阶段的 ResNet+Attention 重复次数
            norm_num_groups:      GroupNorm 中的组数
            norm_eps:             GroupNorm 的 epsilon
            add_downsample:       是否在末尾添加下采样
            resblock_updown:      若 True，用 ResNet Block 实现下采样（down=True）
                                  否则用 DiffusionUnetDownsample
            downsample_padding:   卷积下采样时的 padding（仅对 use_conv 下采样有效）
            num_head_channels:    注意力头的通道数（而非头数）
            include_fc:           SpatialAttentionBlock 中是否包含前馈层
            use_combined_linear:  是否在注意力中使用合并的 QKV 线性层
            use_flash_attention:  是否使用 Flash Attention 优化
        """
        super().__init__()
        self.resblock_updown = resblock_updown

        # 按照 num_res_blocks 次数构建平行的 ResNet+Attention 列表
        resnets = []
        attentions = []
        for i in range(num_res_blocks):
            # 第一层的输入通道为用户指定 in_channels，之后都为 out_channels
            in_ch = in_channels if i == 0 else out_channels

            # 1) ResNet Block (支持注入时间嵌入 temb)
            resnets.append(
                DiffusionUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=in_ch,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )
            # 2) 空间注意力模块
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

        # 将列表封装为 ModuleList，保证参数注册
        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)

        # 下采样模块：可选添加
        if add_downsample:
            if resblock_updown:
                # 使用 ResNet Block（down=True）来做下采样
                self.downsampler = DiffusionUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    down=True,                # 指定下采样行为
                )
            else:
                # 使用专门的 Conv/Pool 下采样
                self.downsampler = DiffusionUnetDownsample(
                    spatial_dims=spatial_dims,
                    num_channels=out_channels,
                    use_conv=True,            # 用卷积方式下采样
                    out_channels=out_channels,
                    padding=downsample_padding,
                )
        else:
            self.downsampler = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            hidden_states: [B, C, ...]，输入特征图
            temb:          [B, temb_channels]，时间/条件嵌入
            context:       可选上下文（此处忽略，仅为接口一致）
        Returns:
            hidden_states:   经所有模块处理后的特征图
            output_states:   包含每次 ResNet+Attention 后（及最终下采样后）的中间特征列表
        """
        # 丢弃未使用的 context
        del context

        output_states = []

        # 1) 对每个 ResNet Block + Attention 作前向，并收集输出
        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)    # 注入时间嵌入
            hidden_states = attn(hidden_states).contiguous()
            output_states.append(hidden_states)

        # 2) 下采样（若配置了）
        if self.downsampler is not None:
            hidden_states = self.downsampler(hidden_states, temb)
            output_states.append(hidden_states)

        # 返回最终特征与所有中间特征
        return hidden_states, output_states
    
class ConditionalAttnDownBlock(nn.Module):
    """
    包含 ResNet 与空间注意力（Spatial Attention）模块的下采样块。
    结构：
      - 若干个 (ResNet Block → Spatial Attention) 串联
      - 可选一个下采样（卷积或 ResNet-based）
    用于 Diffusion U-Net 的降分辨率阶段，同时保留多尺度特征。
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_downsample: bool = True,
        resblock_updown: bool = False,
        downsample_padding: int = 1,
        num_head_channels: int = 1,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims:         2 或 3，决定使用 2D/3D 操作
            in_channels:          输入通道数
            out_channels:         输出通道数（各子模块通道统一）
            num_res_blocks:       每个阶段的 ResNet+Attention 重复次数
            norm_num_groups:      GroupNorm 中的组数
            norm_eps:             GroupNorm 的 epsilon
            add_downsample:       是否在末尾添加下采样
            resblock_updown:      若 True，用 ResNet Block 实现下采样（down=True）
                                  否则用 DiffusionUnetDownsample
            downsample_padding:   卷积下采样时的 padding（仅对 use_conv 下采样有效）
            num_head_channels:    注意力头的通道数（而非头数）
            include_fc:           SpatialAttentionBlock 中是否包含前馈层
            use_combined_linear:  是否在注意力中使用合并的 QKV 线性层
            use_flash_attention:  是否使用 Flash Attention 优化
        """
        super().__init__()
        self.resblock_updown = resblock_updown

        # 按照 num_res_blocks 次数构建平行的 ResNet+Attention 列表
        resnets = []
        attentions = []
        for i in range(num_res_blocks):
            # 第一层的输入通道为用户指定 in_channels，之后都为 out_channels
            in_ch = in_channels if i == 0 else out_channels

            # 1) ResNet Block (支持注入时间嵌入 temb)
            resnets.append(
                ConditionalUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=in_ch,
                    out_channels=out_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )
            # 2) 空间注意力模块
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

        # 将列表封装为 ModuleList，保证参数注册
        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)

        # 下采样模块：可选添加
        if add_downsample:
            if resblock_updown:
                # 使用 ResNet Block（down=True）来做下采样
                self.downsampler = ConditionalUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    down=True,                # 指定下采样行为
                )
            else:
                # 使用专门的 Conv/Pool 下采样
                self.downsampler = ConditionalUnetDownsample(
                    spatial_dims=spatial_dims,
                    num_channels=out_channels,
                    use_conv=True,            # 用卷积方式下采样
                    out_channels=out_channels,
                    padding=downsample_padding,
                )
        else:
            self.downsampler = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            hidden_states: [B, C, ...]，输入特征图
            context:       可选上下文（此处忽略，仅为接口一致）
        Returns:
            hidden_states:   经所有模块处理后的特征图
            output_states:   包含每次 ResNet+Attention 后（及最终下采样后）的中间特征列表
        """
        # 丢弃未使用的 context
        del context

        output_states = []

        # 1) 对每个 ResNet Block + Attention 作前向，并收集输出
        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states)    # 注入时间嵌入
            hidden_states = attn(hidden_states).contiguous()
            output_states.append(hidden_states)

        # 2) 下采样（若配置了）
        if self.downsampler is not None:
            hidden_states = self.downsampler(hidden_states)
            output_states.append(hidden_states)

        # 返回最终特征与所有中间特征
        return hidden_states, output_states


class CrossAttnDownBlock(nn.Module):
    """
    包含 ResNet、Cross‐Attention Transformer 和可选下采样的 U-Net 下采样块。

    结构:
      for each of num_res_blocks:
        1) ResNetBlock (inject time embedding)
        2) SpatialTransformer (cross‐attention with optional context)
      optional:
        3) Downsample (卷积或 ResNet 降采样)
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_downsample: bool = True,
        resblock_updown: bool = False,
        downsample_padding: int = 1,
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
            spatial_dims:         2 或 3，决定 2D/3D 操作
            in_channels:          输入特征图通道数
            out_channels:         经过此块后输出的通道数
            temb_channels:        时间嵌入 (temb) 的维度
            num_res_blocks:       ResNet+Transformer 的重复次数
            norm_num_groups:      GroupNorm 的组数
            norm_eps:             GroupNorm 的 epsilon
            add_downsample:       是否在末尾添加下采样层
            resblock_updown:      如果 True，用 ResNetBlock(down=True) 做下采样
            downsample_padding:   卷积下采样时的 padding (仅在 use_conv 下采样有效)
            num_head_channels:    每个注意力头的通道数
            transformer_num_layers: Transformer 中的层数
            cross_attention_dim:  跨注意力键/值的特征维度 (None 则使用 out_channels)
            upcast_attention:     是否在注意力计算中使用高精度 upcasting
            dropout_cattn:        注意力层的 dropout 比例
            include_fc:           SpatialTransformer 是否包含前馈层
            use_combined_linear:  是否合并 QKV 的线性层
            use_flash_attention:  是否使用 FlashAttention 优化
        """
        super().__init__()
        self.resblock_updown = resblock_updown

        # 1. 构建 ResNetBlocks 与 SpatialTransformers 列表
        resnets = []
        attentions = []
        for i in range(num_res_blocks):
            # 第一层输入通道为 in_channels，其后都为 out_channels
            in_ch = in_channels if i == 0 else out_channels

            # 1) ResNet Block: 支持注入时间嵌入 temb, 可选上下采样
            resnets.append(
                DiffusionUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=in_ch,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )

            # 2) Cross‐Attention Transformer: 支持跨注意力融合 context
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

        # 注册模块
        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)

        # 2. 可选下采样层
        if add_downsample:
            if resblock_updown:
                # 用 ResNetBlock(down=True) 实现下采样
                self.downsampler = DiffusionUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    down=True,
                )
            else:
                # 用卷积下采样
                self.downsampler = DiffusionUnetDownsample(
                    spatial_dims=spatial_dims,
                    num_channels=out_channels,
                    use_conv=True,
                    out_channels=out_channels,
                    padding=downsample_padding,
                )
        else:
            self.downsampler = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            hidden_states: [B, C, ...]，输入特征图
            temb:          [B, temb_channels]，时间/条件嵌入
            context:       可选上下文，用于 cross‐attention
        Returns:
            hidden_states: 经过本块处理后的特征图
            output_states: 包含每次 ResNet+Attention (及下采样) 后的特征列表
        """
        output_states: list[torch.Tensor] = []

        # 1) 依次执行 ResNet Block -> SpatialTransformer
        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)           # 注入时间嵌入
            hidden_states = attn(hidden_states, context=context).contiguous()  # 跨注意力
            output_states.append(hidden_states)

        # 2) 下采样 (如果配置)
        if self.downsampler is not None:
            hidden_states = self.downsampler(hidden_states, temb)
            output_states.append(hidden_states)

        # 返回最终特征与各中间特征（用于跳跃连接）
        return hidden_states, output_states

class ConditionalCrossAttnDownBlock(nn.Module):
    """
    包含 ResNet、Cross‐Attention Transformer 和可选下采样的 U-Net 下采样块。

    结构:
      for each of num_res_blocks:
        1) ResNetBlock (inject time embedding)
        2) SpatialTransformer (cross‐attention with optional context)
      optional:
        3) Downsample (卷积或 ResNet 降采样)
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_downsample: bool = True,
        resblock_updown: bool = False,
        downsample_padding: int = 1,
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
            spatial_dims:         2 或 3，决定 2D/3D 操作
            in_channels:          输入特征图通道数
            out_channels:         经过此块后输出的通道数
            temb_channels:        时间嵌入 (temb) 的维度
            num_res_blocks:       ResNet+Transformer 的重复次数
            norm_num_groups:      GroupNorm 的组数
            norm_eps:             GroupNorm 的 epsilon
            add_downsample:       是否在末尾添加下采样层
            resblock_updown:      如果 True，用 ResNetBlock(down=True) 做下采样
            downsample_padding:   卷积下采样时的 padding (仅在 use_conv 下采样有效)
            num_head_channels:    每个注意力头的通道数
            transformer_num_layers: Transformer 中的层数
            cross_attention_dim:  跨注意力键/值的特征维度 (None 则使用 out_channels)
            upcast_attention:     是否在注意力计算中使用高精度 upcasting
            dropout_cattn:        注意力层的 dropout 比例
            include_fc:           SpatialTransformer 是否包含前馈层
            use_combined_linear:  是否合并 QKV 的线性层
            use_flash_attention:  是否使用 FlashAttention 优化
        """
        super().__init__()
        self.resblock_updown = resblock_updown

        # 1. 构建 ResNetBlocks 与 SpatialTransformers 列表
        resnets = []
        attentions = []
        for i in range(num_res_blocks):
            # 第一层输入通道为 in_channels，其后都为 out_channels
            in_ch = in_channels if i == 0 else out_channels

            # 1) ResNet Block: 支持注入时间嵌入 temb, 可选上下采样
            resnets.append(
                DiffusionUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=in_ch,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )

            # 2) Cross‐Attention Transformer: 支持跨注意力融合 context
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

        # 注册模块
        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)

        # 2. 可选下采样层
        if add_downsample:
            if resblock_updown:
                # 用 ResNetBlock(down=True) 实现下采样
                self.downsampler = DiffusionUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    down=True,
                )
            else:
                # 用卷积下采样
                self.downsampler = DiffusionUnetDownsample(
                    spatial_dims=spatial_dims,
                    num_channels=out_channels,
                    use_conv=True,
                    out_channels=out_channels,
                    padding=downsample_padding,
                )
        else:
            self.downsampler = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            hidden_states: [B, C, ...]，输入特征图
            temb:          [B, temb_channels]，时间/条件嵌入
            context:       可选上下文，用于 cross‐attention
        Returns:
            hidden_states: 经过本块处理后的特征图
            output_states: 包含每次 ResNet+Attention (及下采样) 后的特征列表
        """
        output_states: list[torch.Tensor] = []

        # 1) 依次执行 ResNet Block -> SpatialTransformer
        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)           # 注入时间嵌入
            hidden_states = attn(hidden_states, context=context).contiguous()  # 跨注意力
            output_states.append(hidden_states)

        # 2) 下采样 (如果配置)
        if self.downsampler is not None:
            hidden_states = self.downsampler(hidden_states, temb)
            output_states.append(hidden_states)

        # 返回最终特征与各中间特征（用于跳跃连接）
        return hidden_states, output_states