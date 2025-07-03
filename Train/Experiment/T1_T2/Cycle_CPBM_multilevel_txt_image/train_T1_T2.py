import sys
import os
import torch
import random
import logging
import warnings
import numpy as np
from torch.utils.data import DataLoader

# 自定义模块导入
warnings.filterwarnings('ignore')    # 忽略警告
sys.path.append('/data1/weiyibo/NPC-MRI/Code/Pctch_model/')
from Model.Cycle_CPBM_model import Cycle_CPBM_model
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
    train_dataset = npy_3D_dataset(opt.data, mode='train')
    train_loader = DataLoader(
        train_dataset, 
        batch_size=opt.train.batch_size, 
        shuffle=True, 
        num_workers=opt.train.num_workers, 
        drop_last=True
    )
    logging.info(f"训练集大小: {len(train_dataset)}, 批次数: {len(train_loader)}")

    val_dataset = npy_3D_dataset(opt.data, mode='val')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
    logging.info(f"验证集大小: {len(val_dataset)}, 批次数: {len(val_loader)}")

    test_dataset = npy_3D_dataset(opt.data, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    logging.info(f"测试集大小: {len(test_dataset)}, 批次数: {len(test_loader)}")
    logging.info("========== ⌛数据集加载完毕⌛ ==========")

    return train_loader, val_loader, test_loader


def validate_model(model, val_loader, epoch, opt, best_metrics, model_components):
    """验证模型"""
    logging.info(f"😃开始验证epoch{epoch}模型")
    mae_x0_list, ssim_x0_list, psnr_x0_list, mae_xT_list, ssim_xT_list, psnr_xT_list = [], [], [], [], [], []

    with torch.no_grad():
        for i, data in enumerate(val_loader):
        # for i, data in zip(range(2), val_loader):

            x_0, x_T, txt_con, txt_con_mask, img_con, img_con_mask, patient_id, _ = model.set_input(data)
            x0, xT, pred_x0, pred_xT, mae_x0, ssim_x0, psnr_x0, mae_xT, ssim_xT, psnr_xT = model.val(x_0, x_T, txt_con, img_con )
            
            mae_x0_list.append(mae_x0)
            ssim_x0_list.append(ssim_x0)
            psnr_x0_list.append(psnr_x0)
            mae_xT_list.append(mae_xT)
            ssim_xT_list.append(ssim_xT)
            psnr_xT_list.append(psnr_xT)

            # 保存单个样本验证结果
            val_cal_dict = {'patient_id': patient_id, 'MAE_x0': mae_x0, 'SSIM_x0': ssim_x0, 'PSNR_x0': psnr_x0, 
                            'MAE_xT': mae_xT, 'SSIM_xT': ssim_xT, 'PSNR_xT': psnr_xT}
            logging.info(f"验证集epoch_{epoch}模型的patient_id:{patient_id}的验证结果:"
                         f"MAE_x0:{mae_x0:.4f}, SSIM_x0:{ssim_x0:.4f}, PSNR_x0:{psnr_x0:.4f}, "
                         f"MAE_xT:{mae_xT:.4f}, SSIM_xT:{ssim_xT:.4f}, PSNR_xT:{psnr_xT:.4f}")
            save_loss_csv(
                file_path=opt.path.val_metric_csv_path,
                epoch=epoch,
                header=['patient_id', 'MAE_x0', 'SSIM_x0', 'PSNR_x0', 'MAE_xT', 'SSIM_xT', 'PSNR_xT'],
                loss_dict=val_cal_dict
            )

            # save_path_dir = os.path.join(opt.path.img_path_dir, 'val', f"epoch_{epoch}", patient_id)
            # os.makedirs(save_path_dir, exist_ok=True)
            # model.plot(x0, xT, pred_x0, pred_xT, save_path_dir)

    # 计算验证集平均值与标准差
    metrics_avg = {
        'MAE_x0': (np.mean(mae_x0_list), np.std(mae_x0_list)),
        'SSIM_x0': (np.mean(ssim_x0_list), np.std(ssim_x0_list)),
        'PSNR_x0': (np.mean(psnr_x0_list), np.std(psnr_x0_list)),
        'MAE_xT': (np.mean(mae_xT_list), np.std(mae_xT_list)),
        'SSIM_xT': (np.mean(ssim_xT_list), np.std(ssim_xT_list)),
        'PSNR_xT': (np.mean(psnr_xT_list), np.std(psnr_xT_list)),
    }
    # 格式化输出结果
    metrics_str = (
        f"MAE_x0: {metrics_avg['MAE_x0'][0]:.6f} ± {metrics_avg['MAE_x0'][1]:.6f}, "
        f"SSIM_x0: {metrics_avg['SSIM_x0'][0]:.6f} ± {metrics_avg['SSIM_x0'][1]:.6f}, "
        f"PSNR_x0: {metrics_avg['PSNR_x0'][0]:.6f} ± {metrics_avg['PSNR_x0'][1]:.6f}, "
        f"MAE_xT: {metrics_avg['MAE_xT'][0]:.6f} ± {metrics_avg['MAE_xT'][1]:.6f}, "
        f"SSIM_xT: {metrics_avg['SSIM_xT'][0]:.6f} ± {metrics_avg['SSIM_xT'][1]:.6f}, "
        f"PSNR_xT: {metrics_avg['PSNR_xT'][0]:.6f} ± {metrics_avg['PSNR_xT'][1]:.6f}"
    )
    logging.info(f"验证集epoch_{epoch}模型的验证结果: {metrics_str}")
    
    # 保存验证集结果
    save_loss_csv(
        file_path=opt.path.val_avg_metric_csv_path,
        epoch=epoch,
        header=['epoch', 'MAE_x0', 'SSIM_x0', 'PSNR_x0', 'MAE_xT', 'SSIM_xT', 'PSNR_xT'],
        loss_dict={key: f"{val[0]:.4f} ± {val[1]:.4f}" for key, val in metrics_avg.items()}
    )

    # 更新最佳指标并保存模型
    for metric, (avg, _) in metrics_avg.items():
        if avg > best_metrics[metric]:
            best_metrics[metric] = avg
            save_model(
                model_components=model_components,
                epoch=epoch,
                save_dir=opt.path.checkpoint_path_dir,
                file_name=f"best_{metric.lower()}.pth"
            )
            logging.info(f"🎉保存epoch{epoch}模型，{metric}最佳模型")

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
    x_0, x_T, txt_con, _, img_con, _, patient_id, _ = model.set_input(data)

    b, c, d, h, w = x_0.shape
    patch_d, patch_h, patch_w = patch_size

    # 确保图像尺寸足够裁剪
    assert d >= patch_d and h >= patch_h and w >= patch_w, "图像尺寸必须大于块大小"
    num_random_crops = 1   # 随机裁剪块的数量，这里设置为 4，可以根据需要调整
    # # 计算每个维度的 stride 和总块数
    # stride_d = int(patch_d * (1 - overlap))
    # stride_h = int(patch_h * (1 - overlap))
    # stride_w = int(patch_w * (1 - overlap))

    # # 总块数，用 stride 模拟覆盖区域
    # num_d = (d - patch_d) // stride_d + 1
    # num_h = (h - patch_h) // stride_h + 1
    # num_w = (w - patch_w) // stride_w + 1
    # num_random_crops = num_d * num_h * num_w

    # 初始化损失累加
    total_loss = 0.0
    total_loss_dict = {key: 0.0 for key in model.get_loss_names()}

    for _ in range(num_random_crops):
        # 随机裁剪起始位置，确保不会越界
        start_d = random.randint(0, d - patch_d)
        start_h = random.randint(0, h - patch_h)
        start_w = random.randint(0, w - patch_w)

        # 裁剪块
        x0_patch = x_0[:, :, start_d:start_d + patch_d, start_h:start_h + patch_h, start_w:start_w + patch_w]
        xT_patch = x_T[:, :, start_d:start_d + patch_d, start_h:start_h + patch_h, start_w:start_w + patch_w]
        img_con_patch = img_con[:, :, start_d:start_d + patch_d, start_h:start_h + patch_h, start_w:start_w + patch_w]

        # 确保块在设备上
        x0_patch = x0_patch.to(x_0.device)
        xT_patch = xT_patch.to(x_0.device)
        img_con_patch = img_con_patch.to(x_0.device)

        # 计算单个块的损失
        loss, loss_dict = model(x0_patch, xT_patch, txt_con, img_con_patch)

        # 累加损失
        total_loss += loss  # 将损失值累加
        
        # 累加损失字典中的每项
        for key in loss_dict:
            total_loss_dict[key] += loss_dict[key]

    # 计算所有随机块的平均损失
    avg_loss = total_loss / num_random_crops
    avg_loss_dict = {key: value / num_random_crops for key, value in total_loss_dict.items()}

    return avg_loss, avg_loss_dict

def main():
    # 1️⃣ 加载配置
    config_path = "./Config/Cycle_CPBM_multilevel_txt_image/T1_T2.yaml"
    opt = load_yaml_config(config_path)

    # 1.1 设置随机种子与保存路径
    set_random_seed(opt.train.seed)
    create_directories(opt)

    # 2️⃣ 配置日志
    setup_logging(opt)
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
    train_loader, val_loader, test_loader = initialize_dataloader(opt)

    # 5️⃣ 创建模型
    model = Cycle_CPBM_model(opt).to(opt.train.device)
    model_components = {'net_f': model.net_f.state_dict(), 'net_b': model.net_b.state_dict(), 'optimizer': model.optimizer.state_dict()}
    if opt.train.continue_train:
        logging.info("加载已保存模型参数")
        val_model = {'net_f': model.net_f, 'net_b': model.net_b, 'optimizer': model.optimizer}
        opt.train.epoch_start = load_model(val_model, checkpoint_path=opt.path.checkpoint_path, device=opt.train.device)
    best_metrics = {metric: 0 for metric in ['MAE_x0', 'SSIM_x0', 'PSNR_x0', 'MAE_xT', 'SSIM_xT', 'PSNR_xT']}

    # 6️⃣ 开始训练
    logging.info("========== ⏳开始训练⏳ ==========")
    for epoch in range(opt.train.epoch_start + 1, opt.train.max_epochs + 1):
        if opt.loss.is_perceptual:
            loss_dict = {'loss','loss_perceptual'}  # 保存每个batch的损失值
        else:
            loss_dict = {'loss'}  # 保存每个batch的损失值
        loss_sums = {key: 0.0 for key in loss_dict} # 初始化每个损失的累加值为0

        for i, data in enumerate(train_loader):
        # for i, data in zip(range(1), train_loader):
            loss, loss_dict = random_sliding_window_image(model, data, opt.data.patch_image_shape, opt.data.overlap)
            # 累加每个损失函数的值
            loss_dict = {'loss': loss}
            if i % opt.train.log_freq == 0:
                lr = model.optimizer.param_groups[0]['lr']
                train_str = f"epoch:{epoch}|{opt.train.max_epochs}; batch:{i+1}/{len(train_loader)}; Lr:{lr:.7f}; " + ", ".join([f"{key}:{value:.6f}" for key, value in loss_dict.items()])
                logging.info(train_str)

        # 计算平均损失并保存
        avg_loss_dict = {key: loss_sums[key] / len(train_loader) for key in loss_sums.keys()}
        avg_loss_str = f"epoch:{epoch}|{opt.train.max_epochs};" + ", ".join([f"{key}:{value:.6f}" for key, value in avg_loss_dict.items()])
        logging.info(avg_loss_str)
        save_loss_csv(opt.path.train_avg_loss_csv_path, epoch,  ['epoch'] + list(avg_loss_dict.keys()), {'epoch': epoch, **avg_loss_dict})

        # 学习率调度与模型保存
        model.scheduler.step()
        save_model(model_components, epoch, opt.path.checkpoint_path_dir, file_name=f"latest.pth")

        # 验证模型
        if epoch % opt.train.val_freq == 0:
            torch.cuda.empty_cache()
            model.to(opt.train.device)
            best_metrics = validate_model(model, val_loader, epoch, opt, best_metrics, model_components)


if __name__ == '__main__':
    main()