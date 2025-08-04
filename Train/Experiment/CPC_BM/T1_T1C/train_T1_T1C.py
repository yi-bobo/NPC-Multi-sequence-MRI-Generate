import sys
import os
import torch
import random
import logging
import warnings
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist

# 自定义模块导入
warnings.filterwarnings('ignore')
sys.path.append('/data1/weiyibo/NPC-MRI/Code/NPC-Multi-sequence-MRI-Generate/')
from Model.CPC_BM import CPC_BM
from Dataset.pt_dataset import npy_3D_dataset
from Utils.path_util import create_directories
from Utils.config_util import load_yaml_config, log_namespace
from Utils.logging_util import setup_logging, save_loss_csv
from Utils.save_load_model_util import save_model, load_model


def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def initialize_dataloader(opt, rank=0, world_size=1):
    """自动支持单/多卡的 DataLoader"""
    train_dataset = npy_3D_dataset(opt.data, mode='train')
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.train.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=opt.train.num_workers,
        drop_last=True,
        pin_memory=True
    )

    val_dataset = npy_3D_dataset(opt.data, mode='val')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    test_dataset = npy_3D_dataset(opt.data, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    
    return train_loader, val_loader, test_loader


def validate_model(model, val_loader, epoch, opt, best_metrics, rank, use_ddp):
    """验证模型 (仅 rank=0 输出日志和保存 best 模型)"""
    if rank == 0:
        logging.info(f"😃 开始验证 epoch {epoch}")

    # 取出 DDP 包裹的真实模型
    real_model = model.module if use_ddp else model
    real_model.eval()

    mae_x0_list, ssim_x0_list, psnr_x0_list, mae_xT_list, ssim_xT_list, psnr_xT_list = [], [], [], [], [], []

    with torch.no_grad():
        for data in val_loader:
            x_0, x_T, txt, _, img_con1, img_con2, _, patient_id, _ = real_model.set_input(data)
            (
                _, _, _, _,
                mae_x0, ssim_x0, psnr_x0,
                mae_xT, ssim_xT, psnr_xT
            ) = real_model.val(x_0, x_T, txt, img_con1, img_con2)

            mae_x0_list.append(mae_x0)
            ssim_x0_list.append(ssim_x0)
            psnr_x0_list.append(psnr_x0)
            mae_xT_list.append(mae_xT)
            ssim_xT_list.append(ssim_xT)
            psnr_xT_list.append(psnr_xT)

            if rank == 0:
                logging.info(
                    f"val epoch {epoch} | patient:{patient_id} -> "
                    f"MAE_x0:{mae_x0:.4f}, SSIM_x0:{ssim_x0:.4f}, PSNR_x0:{psnr_x0:.4f}, "
                    f"MAE_xT:{mae_xT:.4f}, SSIM_xT:{ssim_xT:.4f}, PSNR_xT:{psnr_xT:.4f}"
                )

    if rank == 0:
        # 计算平均值和标准差
        metrics_avg = {
            'MAE_x0': (np.mean(mae_x0_list), np.std(mae_x0_list)),
            'SSIM_x0': (np.mean(ssim_x0_list), np.std(ssim_x0_list)),
            'PSNR_x0': (np.mean(psnr_x0_list), np.std(psnr_x0_list)),
            'MAE_xT': (np.mean(mae_xT_list), np.std(mae_xT_list)),
            'SSIM_xT': (np.mean(ssim_xT_list), np.std(ssim_xT_list)),
            'PSNR_xT': (np.mean(psnr_xT_list), np.std(psnr_xT_list)),
        }

        metrics_str = ", ".join([f"{k}:{v[0]:.4f}±{v[1]:.4f}" for k, v in metrics_avg.items()])
        logging.info(f"验证集 epoch {epoch} -> {metrics_str}")

        # 保存 best 模型
        for metric, (avg, _) in metrics_avg.items():
            if avg > best_metrics[metric]:
                best_metrics[metric] = avg
                state_dict = {
                    'net_f': real_model.net_f.state_dict(),
                    'net_b': real_model.net_b.state_dict(),
                    'optimizer': real_model.optimizer.state_dict()
                }
                save_model(state_dict, epoch, opt.path.checkpoint_path_dir, file_name=f"best_{metric.lower()}.pth")
                logging.info(f"🎉 保存 epoch {epoch} 最佳 {metric} 模型")

    return best_metrics

def random_sliding_window_image(model, iters, data, patch_size, autocast_dtype, rank, world_size, use_ddp):
    """随机裁剪训练 patch (兼容单/多卡)"""
    real_model = model.module if use_ddp else model
    input, target, txt, _, img_con1, img_con2, _, _, _ = real_model.set_input(data)

    _, _, d, h, w = input.shape
    patch_d, patch_h, patch_w = patch_size
    num_random_crops = 4

    total_loss_dict = defaultdict(lambda: 0.0)
    real_model.train()

    for m in range(num_random_crops):
        start_d = random.randint(0, d - patch_d)
        start_h = random.randint(0, h - patch_h)
        start_w = random.randint(0, w - patch_w)

        input_patch = input[:, :, start_d:start_d + patch_d, start_h:start_h + patch_h, start_w:start_w + patch_w]
        target_patch = target[:, :, start_d:start_d + patch_d, start_h:start_h + patch_h, start_w:start_w + patch_w]
        img_con1_patch = img_con1[:, :, start_d:start_d + patch_d, start_h:start_h + patch_h, start_w:start_w + patch_w]
        img_con2_patch = img_con2[:, :, start_d:start_d + patch_d, start_h:start_h + patch_h, start_w:start_w + patch_w]

        input_patch, target_patch = input_patch.cuda(int(rank)), target_patch.cuda(int(rank))
        img_con1_patch, img_con2_patch = img_con1_patch.cuda(int(rank)), img_con2_patch.cuda(int(rank))

        with torch.amp.autocast(device_type='cuda', dtype=autocast_dtype):
            _, loss_dict = real_model(input_patch, target_patch, txt, img_con1_patch, img_con2_patch, iters)

        for key in loss_dict:
            total_loss_dict[key] += loss_dict[key]

    return {key: value / num_random_crops for key, value in total_loss_dict.items()}


def train_worker(rank, world_size, opt):
    """通用训练逻辑 (单卡或多卡 DDP)"""
    use_ddp = world_size > 1

    if use_ddp:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    opt.train.device = device

    set_random_seed(opt.train.seed)
    
    create_directories(opt)
    setup_logging(opt)
    logging.info("========== ⏳训练配置参数⏳ ==========")
    log_namespace(opt)

    train_loader, val_loader, _ = initialize_dataloader(opt, rank, world_size)

    # 构建模型
    model = CPC_BM(opt).to(device)
    if use_ddp:
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    optimizer_f = model.optimizer_f if not use_ddp else model.module.optimizer_f
    optimizer_b = model.optimizer_b if not use_ddp else model.module.optimizer_b
    # scheduler = model.scheduler if not use_ddp else model.module.scheduler

    scaler = torch.amp.GradScaler('cuda')
    autocast_dtype = torch.float16

    # 恢复训练
    if opt.train.continue_train and rank == 0:
        logging.info("加载已保存模型参数")
        val_model = {
            'net_f': model.net_f if not use_ddp else model.module.net_f,
            'net_b': model.net_b if not use_ddp else model.module.net_b,
            'optimizer_f': optimizer_f, 'optimizer_b': optimizer_b
        }
        opt.train.epoch_start = load_model(val_model, checkpoint_path=opt.path.checkpoint_path, device=rank)

    best_metrics = {metric: 0 for metric in ['MAE_x0', 'SSIM_x0', 'PSNR_x0', 'MAE_xT', 'SSIM_xT', 'PSNR_xT']}
    global steps
    steps = 1

    # 开始训练
    for epoch in range(opt.train.epoch_start + 1, opt.train.max_epochs + 1):
        if use_ddp and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        loss_sums = defaultdict(lambda: 0.0)
        for i, data in enumerate(train_loader):
        # for i, data in zip(range(3), train_loader):
            loss_dict = random_sliding_window_image(model, i, data, opt.data.patch_image_shape,
                                                    autocast_dtype, rank, world_size, use_ddp)
            for key in loss_dict:
                loss_sums[key] += loss_dict[key]

            if i % opt.train.log_freq == 0:
                lr_f = optimizer_f.param_groups[0]['lr']
                lr_b = optimizer_b.param_groups[0]['lr']
                train_str = f"[rank{rank}] epoch:{epoch}/{opt.train.max_epochs}; batch:{i+1}/{len(train_loader)}; lr_f:{lr_f:.7f}; lr_b:{lr_b: .7f} " + \
                            ", ".join([f"{key}:{value:.6f}" for key, value in loss_dict.items()])
                logging.info(train_str)

        
        avg_loss_dict = {key: loss_sums[key] / len(train_loader) for key in loss_sums.keys()}
        avg_loss_str = f"epoch:{epoch}/{opt.train.max_epochs}; " + \
                        ", ".join([f"{k}:{v:.6f}" for k, v in avg_loss_dict.items()])
        logging.info(avg_loss_str)
        save_loss_csv(opt.path.train_avg_loss_csv_path, epoch, ['epoch'] + list(avg_loss_dict.keys()),
                        {'epoch': epoch, **avg_loss_dict})

        # scheduler.step()

        # 验证 & 保存模型
        if epoch % opt.train.val_freq == 0:
            torch.cuda.empty_cache()
            best_metrics = validate_model(model, val_loader, epoch, opt, best_metrics, rank, use_ddp)
            if rank == 0:
                # 保存 latest.pth
                real_model = model.module if use_ddp else model
                state_dict = {
                    'net_f': real_model.net_f.state_dict(),
                    'net_b': real_model.net_b.state_dict(),
                    'optimizer_f': optimizer_f.state_dict(), 'optimizer_b': optimizer_b.state_dict()
                }
                save_model(state_dict, epoch, opt.path.checkpoint_path_dir, file_name=f"latest.pth")

    if use_ddp:
        dist.destroy_process_group()


def main():
    config_path = "/data1/weiyibo/NPC-MRI/Code/NPC-Multi-sequence-MRI-Generate/Config/CPC_BM/T1_T1C.yaml"
    opt = load_yaml_config(config_path)

    # 自动多卡判断
    if opt.train.multi_gpu:
        gpu_list = list(map(int, (opt.train.gpu).split(',')))
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_list))
        world_size = len(gpu_list)

        if world_size < 1:
            raise RuntimeError("至少需要一块GPU！")

        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        mp.spawn(train_worker, nprocs=world_size, args=(world_size, opt), join=True)
    else:
        # 单 GPU / CPU 模式
        train_worker(rank=opt.train.gpu[0], world_size=1, opt=opt)


if __name__ == '__main__':
    main()
