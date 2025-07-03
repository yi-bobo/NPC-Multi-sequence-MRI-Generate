
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from medpy.metric.binary import hd95, assd
import numpy as np

# region 生成图像质量评价指标
def compute_metrics(pred, target):
    data_range = max(pred.max()-pred.min(), target.max()-target.min())
    mae = np.mean(np.abs(pred-target))
    mse = np.mean(((pred-target)**2))
    SSIM = ssim(pred, target, data_range=data_range)
    PSNR = psnr(pred, target, data_range=data_range)
    return mae, mse, SSIM, PSNR

# region 分割性能指标
def compute_dice(pred, target, eps=1e-8):
    pred = pred.float()
    target = target.float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2. * intersection + eps) / (union + eps)
    return dice.item()

def compute_hd95(pred, target, spacing=(1.0, 1.0, 1.0)):
    """
    pred, target: binary mask, shape = [D, H, W]
    spacing: voxel spacing in mm
    """
    pred = pred.astype(np.bool_)
    target = target.astype(np.bool_)
    if pred.any() and target.any():
        return hd95(pred, target, voxelspacing=spacing)
    else:
        return np.nan  # 无法定义 HD

def compute_asd(pred, target, spacing=(1.0, 1.0, 1.0)):
    pred = pred.astype(np.bool_)
    target = target.astype(np.bool_)
    if pred.any() and target.any():
        return assd(pred, target, voxelspacing=spacing)
    else:
        return np.nan

# endregion