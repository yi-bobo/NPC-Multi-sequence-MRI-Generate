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
from Utils.save_load_model_util import load_model


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


def test_model(model, test_loader, epoch, opt, model_name):
    """验证模型"""
    logging.info(f"🍀开始对模型{model_name}进行测试")
    mae_x0_list, ssim_x0_list, psnr_x0_list, mae_xT_list, ssim_xT_list, psnr_xT_list = [], [], [], [], [], []

    with torch.no_grad():
        for i, data in enumerate(test_loader):
        # for i, data in zip(range(2), test_loader):
            x_0, x_T, txt_con, _, img_con, _, patient_id, data_mapping = model.set_input(data)
            x0, xT, pred_x0, pred_xT, mae_x0, ssim_x0, psnr_x0, mae_xT, ssim_xT, psnr_xT = model.val(x_0, x_T, txt_con, img_con )
            
            mae_x0_list.append(mae_x0)
            ssim_x0_list.append(ssim_x0)
            psnr_x0_list.append(psnr_x0)
            mae_xT_list.append(mae_xT)
            ssim_xT_list.append(ssim_xT)
            psnr_xT_list.append(psnr_xT)

            # 保存单个样本验证结果
            test_cal_dict = {'patient_id': patient_id, 'MAE_x0': mae_x0, 'SSIM_x0': ssim_x0, 'PSNR_x0': psnr_x0, 
                            'MAE_xT': mae_xT, 'SSIM_xT': ssim_xT, 'PSNR_xT': psnr_xT}
            logging.info(f"测试集epoch_{epoch}模型的patient_id:{patient_id}的测试结果:"
                         f"MAE_x0:{mae_x0:.4f}, SSIM_x0:{ssim_x0:.4f}, PSNR_x0:{psnr_x0:.4f}, "
                         f"MAE_xT:{mae_xT:.4f}, SSIM_xT:{ssim_xT:.4f}, PSNR_xT:{psnr_xT:.4f}")
            save_loss_csv(
                file_path=opt.path.test_metric_csv_path,
                epoch=epoch,
                header=['epoch', 'patient_id', 'MAE_x0', 'SSIM_x0', 'PSNR_x0', 'MAE_xT', 'SSIM_xT', 'PSNR_xT'],
                loss_dict=test_cal_dict
            )

            save_path_dir = os.path.join(opt.path.img_path_dir, 'test', f"{model_name}_{epoch}", patient_id)
            os.makedirs(save_path_dir, exist_ok=True)
            model.plot(x0, xT, pred_x0, pred_xT, data_mapping, save_path_dir)

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
    logging.info(f"测试集{model_name}在epoch_{epoch}模型的测试结果: {metrics_str}")
    
    # 保存验证集结果
    save_loss_csv(
        file_path=opt.path.test_avg_metric_csv_path,
        epoch=epoch,
        header=['epoch', 'MAE_x0', 'SSIM_x0', 'PSNR_x0', 'MAE_xT', 'SSIM_xT', 'PSNR_xT'],
        loss_dict={key: f"{test[0]:.4f} ± {test[1]:.4f}" for key, test in metrics_avg.items()}
    )


def main():
    # 1️⃣ 加载配置
    config_path = "./Config/Cycle_CPBM_multilevel_txt_image/T1_T2.yaml"
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
    _, _, test_loader = initialize_dataloader(opt)

    # 5️⃣ 创建模型
    model = Cycle_CPBM_model(opt).to(opt.train.device)
    logging.info("加载已保存模型参数")
    test_model_dict = {'net_f': model.net_f, 'net_b': model.net_b, 'optimizer': model.optimizer}
    epoch = load_model(test_model_dict, checkpoint_path=opt.path.checkpoint_path, device=opt.train.device)
    model_name = opt.path.checkpoint_path.split('/')[-1].split('.')[0]
    
    test_model(model, test_loader, epoch, opt, model_name)


if __name__ == '__main__':
    main()