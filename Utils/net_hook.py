"""
给出已经训练好的模型的中间某层输出作为最终输出，查看中间层的输出。
"""
# 定义特征提取钩子
class FeatureExtractor:
    def __init__(self, model, layer_names):
        """
        初始化特征提取器。
        :param model: 已训练好的模型
        :param layer_names: 需要捕获输出的层名列表
        """
        self.model = model.net_f
        self.layer_names = layer_names
        self.outputs = {}

        # 注册 forward hook
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                module.register_forward_hook(self.save_output_hook(name))

    def save_output_hook(self, name):
        """
        定义保存中间层输出的 hook。
        :param name: 层的名称
        """
        def hook(module, input, output):
            self.outputs[name] = output.detach()  # 保存中间层输出
        return hook

    def get_layer_output(self, layer_name):
        """
        获取某一层的输出。
        :param layer_name: 层的名称
        :return: 该层的输出
        """
        return self.outputs.get(layer_name, None)

# region 测试代码
# 实例化特征提取器

import sys
import torch
import logging
from itertools import islice
from torch.utils.data import DataLoader
sys.path.append('/data1/weiyibo/NPC-MRI/Code/Pctch_model/')

from Dataset.patch_dataset import npy_3D_dataset

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

def main():
    from Model.Cycle_CPBM_model import Cycle_CPBM_model
    from Utils.config_util import load_yaml_config
    # 预训练模型加载
    config_path = "./Config/Cycle_CPBM_multilevel_txt_image/T1_T2.yaml"
    opt = load_yaml_config(config_path)
    opt.train.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = opt.train.device
    model = Cycle_CPBM_model(opt).to(device)

    # 初始化数据集
    train_loader, val_loader, test_loader = initialize_dataloader(opt)

    # 指定需要捕获输出的中间层名称
    layer_names = ["middle_block"]

    # 创建特征提取器
    feature_extractor = FeatureExtractor(model, layer_names)

    with torch.no_grad():
        for i, data in enumerate(islice(train_loader, 35)):
            x_0, x_T, txt_con, _, img_con, _, _, _ = model.set_input(data)
            _ = model.val(x_0, x_T, txt_con, img_con)

            # 获取指定中间层的输出
            middle_block_output = feature_extractor.get_layer_output("middle_block")
            print(middle_block_output.shape)

if __name__ == '__main__':
    main()
# endregion