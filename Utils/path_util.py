import os
import datetime

def create_directories(opt):
    """
    创建保存训练过程中日志、模型、图像和CSV文件的目录。
    
    :param opt: 配置对象，其中包含保存路径的相关信息
    """
    # 构造各类保存路径
    opt.path.save_dir = os.path.join(opt.path.save_dir, f'{opt.data.sour_img_name}_to_{opt.data.targ_img_name}', opt.train.model_name)
    opt.path.log_path_dir = os.path.join(opt.path.save_dir, 'Logs')  # 日志保存路径
    opt.path.csv_path_dir = os.path.join(opt.path.save_dir, 'Cal')  # 损失，验证指标csv保存根路径
    opt.path.checkpoint_path_dir = os.path.join(opt.path.save_dir, 'Checkpoints')  # 模型保存路径
    opt.path.img_path_dir = os.path.join(opt.path.save_dir, 'Images')  # 预测结果保存路径

    opt.path.train_loss_csv_path = os.path.join(opt.path.csv_path_dir, f'train_loss_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')  # 训练损失保存路径
    opt.path.train_avg_loss_csv_path = os.path.join(opt.path.csv_path_dir, f'train_avg_loss_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')  # 训练损失保存路径
    opt.path.val_loss_csv_path = os.path.join(opt.path.csv_path_dir, f'val_loss_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')  # 验证损失保存路径
    opt.path.val_metric_csv_path = os.path.join(opt.path.csv_path_dir, f'val_metric_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')  # 验证指标保存路径
    opt.path.val_avg_metric_csv_path = os.path.join(opt.path.csv_path_dir, f'val_avg_metric_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')  # 验证指标保存路径
    opt.path.test_metric_csv_path = os.path.join(opt.path.csv_path_dir, f'test_metric_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')  # 验证指标保存路径
    opt.path.test_avg_metric_csv_path = os.path.join(opt.path.csv_path_dir, f'test_avg_metric_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')  # 验证指标保存路径

    # 创建目录，若目录已存在则不报错
    os.makedirs(opt.path.log_path_dir, exist_ok=True)
    os.makedirs(opt.path.csv_path_dir, exist_ok=True)
    os.makedirs(opt.path.checkpoint_path_dir, exist_ok=True)
    os.makedirs(opt.path.img_path_dir, exist_ok=True)
