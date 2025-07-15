
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

# endregion


# region 分割性能指标

def cal_dice(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """
    计算 Dice 系数
    pred, target: 二值 mask，shape (H, W)，值为 0 或 1
    """
    pred = pred.astype(np.bool_)
    target = target.astype(np.bool_)
    intersection = np.logical_and(pred, target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def cal_iou(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """
    计算 Intersection over Union (IoU)
    """
    pred = pred.astype(np.bool_)
    target = target.astype(np.bool_)
    inter = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return (inter + smooth) / (union + smooth)

def cal_precision(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """
    计算精确率 Precision = TP / (TP + FP)
    """
    pred = pred.astype(np.bool_)
    target = target.astype(np.bool_)
    tp = np.logical_and(pred, target).sum()
    fp = np.logical_and(pred, np.logical_not(target)).sum()
    return (tp + smooth) / (tp + fp + smooth)

def cal_recall(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """
    计算召回率 Recall = TP / (TP + FN)
    """
    pred = pred.astype(np.bool_)
    target = target.astype(np.bool_)
    tp = np.logical_and(pred, target).sum()
    fn = np.logical_and(np.logical_not(pred), target).sum()
    return (tp + smooth) / (tp + fn + smooth)

def cal_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """
    计算像素准确率 Pixel Accuracy
    """
    assert pred.shape == target.shape
    return (pred == target).sum() / pred.size

# endregion