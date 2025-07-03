import os
import torch


def save_model(model_components, epoch, save_dir, file_name="model_checkpoint.pth"):
    """
    通用的模型保存函数。

    Parameters:
        model_components (dict) -- 包含模型和优化器状态字典的字典。例如：
                                   {
                                       'model': model.state_dict(),
                                       'optimizer_G': optimizer_G.state_dict(),
                                       'optimizer_D': optimizer_D.state_dict(),
                                   }
        epoch (int) -- 当前训练的epoch
        save_dir (str) -- 保存模型的文件夹路径
        file_name (str) -- 保存模型的文件名（默认 "model_checkpoint.pth"）
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file_name)

    # 将epoch以及模型组件的状态字典保存到文件
    torch.save({'epoch': epoch, **model_components}, save_path)
    print(f"Model saved to {save_path}")


def load_model(model_components, checkpoint_path, device):
    """
    通用的模型加载函数。

    Parameters:
        model_components (dict) -- 包含需要加载状态字典的字典。例如：
                                   {
                                       'model': model,
                                       'optimizer_G': optimizer_G,
                                       'optimizer_D': optimizer_D,
                                   }
        checkpoint_path (str) -- 保存的模型文件路径
        device (str) -- 加载模型的设备
    """
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"Loading model from {checkpoint_path}...")

    # 恢复各个组件的状态字典
    for key, component in model_components.items():
        if hasattr(component, 'load_state_dict'):
            component.load_state_dict(checkpoint[f"{key}"])
        else:
            raise ValueError(f"Component '{key}' does not have a 'load_state_dict' method.")

    # 返回保存的epoch
    epoch = checkpoint.get('epoch', 0)
    print(f"Model loaded from epoch {epoch}")
    return epoch