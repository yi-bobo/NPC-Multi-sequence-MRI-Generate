import os
import sys
import csv
import logging
from datetime import datetime

def setup_logging(opt, mode='train'):
    """
    设置日志记录系统，创建带时间戳的日志文件，并将日志同时输出到文件和终端。
    
    Parameters:
        opt (object): 一个包含配置的对象，应该有 `save.save_dir` 字段
    """
    # 生成带时间戳的日志文件
    log_filename = os.path.join(opt.path.log_path_dir, f"{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # 配置 logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_filename, mode='w'),  # 记录到文件
            logging.StreamHandler(sys.stdout)  # 在终端输出
        ]
    )
    logging.info(f"Logging setup complete. Logs will be saved to: {log_filename}")

def save_loss_csv(file_path, epoch, header, loss_dict):
    """
    file_path: 保存路径
    epoch: 训练轮数
    iter: 训练批次
    header: 表头
    loss_dict: 损失字典
    """
    with open(file_path, 'a', newline='') as f: # 使用 'w' 模式清空文件
        writer = csv.writer(f)
        if epoch == 1 and header[1]=='iter' and loss_dict['iter']==1:
            writer.writerow(header)
        writer.writerow([epoch] + [loss_dict[key] for key in header[1:]])
        