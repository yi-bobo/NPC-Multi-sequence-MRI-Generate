import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPModel

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

## 测试
gpu = 7         # 使用的GPU编号
is_text = True  # 是否启用文本特征提取
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
text_feature_extraction = TextFeature(device=device, is_text=is_text)         # 初始化文本特征提取模型
