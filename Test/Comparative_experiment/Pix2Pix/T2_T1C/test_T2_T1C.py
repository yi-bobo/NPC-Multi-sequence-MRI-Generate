import sys
import os
import torch
import random
import logging
import warnings
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader

# 自定义模块导入
warnings.filterwarnings('ignore')    # 忽略警告
sys.path.append('/data1/weiyibo/NPC-MRI/Code/NPC-Multi-sequence-MRI-Generate/')
from Model.Pix2Pix_model import Pix2Pix3DModel
from Dataset.patch_dataset import npy_3D_dataset
from Utils.path_util import create_directories
from Utils.config_util import load_yaml_config, log_namespace
from Utils.logging_util import setup_logging, save_loss_csv
from Utils.save_load_model_util import save_model, load_model


def set_random_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def initialize_dataloader(opt):
    """初始化数据加载器"""
    logging.info("========== ⏳加载数据集⏳ ==========")
    test_dataset = npy_3D_dataset(opt.data, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    logging.info(f"测试集大小: {len(test_dataset)}, 批次数: {len(test_loader)}")
    logging.info("========== ⌛数据集加载完毕⌛ ==========")

    return test_loader


def validate_model(model, val_loader, epoch, opt, best_metrics):
    """验证模型"""
    logging.info(f"😃开始验证epoch{epoch}模型")
    mae_list, ssim_list, psnr_list = [], [], []

    with torch.no_grad():
        for i, data in enumerate(val_loader):
        # for i, data in zip(range(2), val_loader):

            input, target, patient_id, data_mapping = model.set_input(data)
            input, target, pred_target, mae, ssim, psnr = model.val(input, target)
            
            mae_list.append(mae)
            ssim_list.append(ssim)
            psnr_list.append(psnr)

            # 保存单个样本验证结果
            test_cal_dict = {'patient_id': patient_id, 'MAE': mae, 'SSIM': ssim, 'PSNR': psnr}
            logging.info(f"测试集epoch_{epoch}模型的patient_id:{patient_id}的验证结果:"
                         f"MAE:{mae:.4f}, SSIM:{ssim:.4f}, PSNR:{psnr:.4f}, ")
            save_loss_csv(
                file_path=opt.path.test_metric_csv_path,
                epoch=epoch,
                header=['patient_id', 'MAE', 'SSIM', 'PSNR'],
                loss_dict=test_cal_dict
            )

            save_path_dir = os.path.join(opt.path.img_path_dir, 'test', f"epoch_{epoch}", patient_id)
            os.makedirs(save_path_dir, exist_ok=True)
            model.plot(input, target, pred_target, data_mapping, save_path_dir)

    # 计算验证集平均值与标准差
    metrics_avg = {
        'MAE': (np.mean(mae_list), np.std(mae_list)),
        'SSIM': (np.mean(ssim_list), np.std(ssim_list)),
        'PSNR': (np.mean(psnr_list), np.std(psnr_list)),
        }
    # 格式化输出结果
    metrics_str = (
        f"MAE: {metrics_avg['MAE'][0]:.6f} ± {metrics_avg['MAE'][1]:.6f}, "
        f"SSIM: {metrics_avg['SSIM'][0]:.6f} ± {metrics_avg['SSIM'][1]:.6f}, "
        f"PSNR: {metrics_avg['PSNR'][0]:.6f} ± {metrics_avg['PSNR'][1]:.6f}"
    )
    logging.info(f"验证集epoch_{epoch}模型的验证结果: {metrics_str}")
    
    # 保存验证集结果
    save_loss_csv(
        file_path=opt.path.val_avg_metric_csv_path,
        epoch=epoch,
        header=['epoch', 'MAE', 'SSIM', 'PSNR'],
        loss_dict={key: f"{val[0]:.4f} ± {val[1]:.4f}" for key, val in metrics_avg.items()}
    )
    return best_metrics

def random_sliding_window_image(model, data, patch_size, overlap):
    """
    对 3D 图像进行随机裁剪，并计算每个图像所有随机块的损失均值。

    Args:
        model: 模型对象，包含 `set_input` 和前向传播逻辑。
        data: 输入数据，包含 x_0, x_T, txt_con, img_con 等。
        patch_size (tuple): 每个块的大小 (patch_d, patch_h, patch_w)。
        overlap (float): 用于计算随机裁剪块的数量 (0 <= overlap < 1)。
        num_random_crops (int): 每张图像要随机裁剪的块数量。

    Returns:
        avg_loss (float): 所有随机块的损失均值。
        avg_loss_dict (dict): 所有随机块的损失字典均值。
    """
    # 提取模型输入
    input, target, _, _ = model.set_input(data)

    b, c, d, h, w = input.shape
    patch_d, patch_h, patch_w = patch_size

    # 确保图像尺寸足够裁剪
    assert d >= patch_d and h >= patch_h and w >= patch_w, "图像尺寸必须大于块大小"
    num_random_crops = 8   # 随机裁剪块的数量，这里设置为 4，可以根据需要调整

    # 初始化损失累加
    total_loss_dict = defaultdict(lambda:0.0)

    for _ in range(num_random_crops):
        # 随机裁剪起始位置，确保不会越界
        start_d = random.randint(0, d - patch_d)
        start_h = random.randint(0, h - patch_h)
        start_w = random.randint(0, w - patch_w)

        # 裁剪块
        input_patch = input[:, :, start_d:start_d + patch_d, start_h:start_h + patch_h, start_w:start_w + patch_w]
        target_patch = target[:, :, start_d:start_d + patch_d, start_h:start_h + patch_h, start_w:start_w + patch_w]
        
        # 确保块在设备上
        input_patch = input_patch.to(input.device)
        target_patch = target_patch.to(input.device)

        # 计算单个块的损失
        loss_dict = model(input_patch, target_patch)

        # 累加损失字典中的每项
        for key in loss_dict:
            total_loss_dict[key] += loss_dict[key]

    # 计算所有随机块的平均损失
    avg_loss_dict = {key: value / num_random_crops for key, value in total_loss_dict.items()}

    return avg_loss_dict

def main():
    # 1️⃣ 加载配置
    config_path = "./Config/Comparative_experiment/Pix2Pix/T2_T1C.yaml"
    opt = load_yaml_config(config_path)

    # 1.1 设置随机种子与保存路径
    set_random_seed(opt.train.seed)
    create_directories(opt)

    # 2️⃣ 配置日志
    setup_logging(opt, mode='test')
    logging.info("========== ⏳训练配置参数⏳ ==========")
    log_namespace(opt)

    # 3️⃣ 选择设备
    if opt.train.multi_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{opt.train.gpu}'
        opt.train.device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
        available_gpus = torch.cuda.device_count()
        opt.train.batch_size = int(opt.train.batch_size * available_gpus)  # 动态调整 batch_size
        logging.info(f"使用 {available_gpus} 个 GPU, 更新 batch_size 为 {opt.train.batch_size}")
    else:
        opt.train.device = torch.device(f"cuda:{opt.train.gpu}" if torch.cuda.is_available() else "cpu")
        logging.info(f"使用设备: {opt.train.device}")

    # 4️⃣ 加载数据集
    test_loader = initialize_dataloader(opt)

    # 5️⃣ 创建模型
    model = Pix2Pix3DModel(opt).to(opt.train.device)
    
    logging.info("加载已保存模型参数")
    val_model = {'net_G': model.net_G, 'net_D': model.net_D, 'optimizer_G': model.optimizer_G, 'optimizer_D': model.optimizer_D}
    epoch = load_model(val_model, checkpoint_path=opt.path.checkpoint_path, device=opt.train.device)
    best_metrics = {metric: 0 for metric in ['MAE', 'SSIM', 'PSNR']}

    # 6️⃣ 开始训测试
    
    torch.cuda.empty_cache()
    model.to(opt.train.device)
    best_metrics = validate_model(model, test_loader, epoch, opt, best_metrics)


if __name__ == '__main__':
    main()