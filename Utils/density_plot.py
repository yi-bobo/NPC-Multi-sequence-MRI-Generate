import os
import sys
sys.path.append('/data1/weiyibo/NPC-MRI/Code/Pctch_model/')
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from Model.Cycle_CPBM_model import Cycle_CPBM_model
import warnings
from itertools import islice
warnings.filterwarnings('ignore')    # 忽略警告
from tqdm import tqdm
from Utils.config_util import load_yaml_config
# 数据加载和处理
def read_npy_data(npy_path, z=None):
    data = np.load(npy_path)
    if z is not None:
        data = data[z, :, :]
    return data


def resize_arr(data, resize_shape):
    from scipy.ndimage import zoom
    data_shape = data.shape
    resize_data = zoom(
        data,
        (resize_shape[0] / data_shape[0], resize_shape[1] / data_shape[1], resize_shape[2] / data_shape[2]),
        order=1,
    )
    return resize_data


def listdir_data(path_dir, z=None):
    T1_path = os.path.join(path_dir, "T1.npy")
    T1C_path = os.path.join(path_dir, "T1C.npy")
    T2_path = os.path.join(path_dir, "T2.npy")
    T1_data = read_npy_data(T1_path)
    T1C_data = read_npy_data(T1C_path)
    T2_data = read_npy_data(T2_path)

    T1_data = resize_arr(T1_data, (36, 256, 256))
    T1C_data = resize_arr(T1C_data, (36, 256, 256))
    T2_data = resize_arr(T2_data, (36, 256, 256))

    max_value = max(T1_data.max(), T1C_data.max(), T2_data.max())
    min_value = min(T1_data.min(), T1C_data.min(), T2_data.min())
    T1_data = (T1_data - min_value) / (max_value - min_value)
    T1C_data = (T1C_data - min_value) / (max_value - min_value)
    T2_data = (T2_data - min_value) / (max_value - min_value)

    if z is not None:
        T1_data = T1_data[z, :, :]
        T1C_data = T1C_data[z, :, :]
        T2_data = T2_data[z, :, :]

    return T1_data, T1C_data, T2_data


class NPC_Dataset(Dataset):
    def __init__(self, split_path):
        self.img_size = [36, 256, 256]
        with open(split_path, 'r') as f:
            self.data_list = [line.strip() for line in f.readlines()]

    def __getitem__(self, index):
        path = (self.data_list[index].strip()).split('&')[0]
        txt = (self.data_list[index].strip()).split('&')[1]

        path_dir = path.strip()
        patient_id = path_dir.split('/')[-1]

        # 加载数据
        T1_data = np.load(os.path.join(path_dir, 'T1.npy'))
        T1C_data = np.load(os.path.join(path_dir, 'T1C.npy'))
        T1C_mask_data = np.load(os.path.join(path_dir, 'T1C_mask.npy'))
        T2_data = np.load(os.path.join(path_dir, 'T2.npy'))
        T2_mask_data = np.load(os.path.join(path_dir, 'T2_mask.npy'))
        T1C_tumor_data = T1C_data * T1C_mask_data
        T2_tumor_data = T2_data * T2_mask_data

        # 计算单个病人的最大最小值
        max_value = max(np.max(T1_data), np.max(T1C_data), np.max(T2_data))
        min_value = min(np.min(T1_data), np.min(T1C_data), np.min(T2_data))

        # 归一化
        T1_data = (T1_data-min_value)/(max_value-min_value)
        T1C_data = (T1C_data-min_value)/(max_value-min_value)
        T2_data = (T2_data-min_value)/(max_value-min_value)
        T1C_tumor_data = (T1C_tumor_data-min_value)/(max_value-min_value)
        T2_tumor_data = (T2_tumor_data-min_value)/(max_value-min_value)

        # resize
        T1_data = resize_arr(T1_data, self.img_size)
        T1C_data = resize_arr(T1C_data, self.img_size)
        T1C_mask_data = resize_arr(T1C_mask_data, self.img_size)
        T2_data = resize_arr(T2_data, self.img_size)
        T2_mask_data = resize_arr(T2_mask_data, self.img_size)
        T1C_tumor_data = resize_arr(T1C_tumor_data, self.img_size)
        T2_tumor_data = resize_arr(T2_tumor_data, self.img_size)

        # 转换为 tensor
        T1_data = torch.from_numpy(T1_data).float().unsqueeze(0)
        T1C_data = torch.from_numpy(T1C_data).float().unsqueeze(0)
        T1C_mask_data = torch.from_numpy(T1C_mask_data).float().unsqueeze(0)
        T2_data = torch.from_numpy(T2_data).float().unsqueeze(0)
        T2_mask_data = torch.from_numpy(T2_mask_data).float().unsqueeze(0)
        T1C_tumor_data = torch.from_numpy(T1C_tumor_data).float().unsqueeze(0)
        T2_tumor_data = torch.from_numpy(T2_tumor_data).float().unsqueeze(0)

        return T1_data, T1C_data, T1C_mask_data, T2_data, T2_mask_data, T1C_tumor_data, T2_tumor_data, patient_id, txt

    def __len__(self):
        return len(self.data_list)


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


# 映射到二维平面
def project_to_2d(data_3d, method='mean'):
    if method == 'mean':
        data_2d = np.mean(data_3d, axis=0)
    elif method == 'max':
        data_2d = np.max(data_3d, axis=0)
    elif method == 'sum':
        data_2d = np.sum(data_3d, axis=0)
    else:
        raise ValueError("Invalid method. Choose from 'mean', 'max', 'sum'.")
    return data_2d


# t-SNE 降维
def apply_tsne(data_2d, n_samples=5000):
    y, x = data_2d.shape
    x_coords, y_coords = np.meshgrid(range(x), range(y))
    coords = np.stack([x_coords.flatten(), y_coords.flatten()], axis=1)
    values = data_2d.flatten()

    features = np.hstack([coords, values[:, np.newaxis]])

    if features.shape[0] > n_samples:
        indices = np.random.choice(features.shape[0], n_samples, replace=False)
        features = features[indices]

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    tsne_results = tsne.fit_transform(features)
    return tsne_results, features[:, 2]


# 绘制 t-SNE 散点图
def plot_2d_scatter(data_2d, values, title="t-SNE Scatter Plot", file_name=None):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=values, cmap='viridis', s=5, alpha=0.7)
    plt.colorbar(scatter, label="Intensity")
    plt.title(title, fontsize=16)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.grid(alpha=0.3)
    if file_name:
        plt.savefig(file_name, dpi=300)
    plt.show()



# 主函数
def main():
    # 数据集路径
    split_path = "Split/zhongshan2/val_with_info.txt"  # 自定义分割文件路径
    dataset = NPC_Dataset(split_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # 预训练模型加载
    config_path = "./Config/Cycle_CPBM_multilevel_txt_image/T1_T2.yaml"
    opt = load_yaml_config(config_path)
    opt.train.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = opt.train.device
    model = Cycle_CPBM_model(opt).to(device)

    # 指定需要捕获输出的中间层名称
    layer_names = ["middle_block"]

    # 创建特征提取器
    feature_extractor = FeatureExtractor(model, layer_names)

    # 提取潜在特征
    print("Extracting latent features...")
    latent_features = []
    model.eval()

    with torch.no_grad():
    # 用 tqdm 包裹 dataloader，显示进度条
        for i, data in tqdm(enumerate(islice(dataloader, 35)), total=35, desc="Loading Data"):
            x_0, x_T, txt_con, _, img_con, _, _, _ = model.set_input(data)
            _ = model.val(x_0, x_T, txt_con, img_con)

            # 获取指定中间层的输出
            middle_block_output = feature_extractor.get_layer_output("middle_block")
            middle_block_output = middle_block_output.squeeze(1)  # 去掉通道维度
            latent = middle_block_output.flatten(1)  # 展平为 (B, features_dim)
            latent_features.append(latent.cpu().numpy())

    # 将所有批次的特征拼接起来
    latent_features = np.concatenate(latent_features, axis=0)
    print(f"Extracted latent features shape: {latent_features.shape}")

    # 绘制潜在空间分布
    plot_tsne(latent_features, output_path="latent_space_visualization.png")


if __name__ == "__main__":
    main()











































# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from scipy.stats import gaussian_kde

# def plot_latent_space_4d(latent_tensor, output_path="latent_space_visualization.png"):
#     """
#     绘制潜在空间的 t-SNE 可视化图
#     :param latent_tensor: 输入潜在特征，形状为 (N, C, H, W)
#     :param output_path: 保存图像的路径
#     """
#     # Step 1: 展平潜在特征
#     # 输入形状为 (1024, 1, 32, 32)，将每个样本展平为 (1024,)
#     N, C, H, W = latent_tensor.shape  # 获取维度信息
#     latent_vectors = latent_tensor.reshape(N, -1)  # 展平为形状 (1024, 1024)

#     # Step 2: 使用 t-SNE 将高维特征降维到二维
#     print("Performing t-SNE...")
#     tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
#     tsne_results = tsne.fit_transform(latent_vectors)  # 输出形状为 (1024, 2)

#     # Step 3: 核密度估计（KDE）计算点的密度
#     print("Calculating density...")
#     kde = gaussian_kde(tsne_results.T)
#     density = kde(tsne_results.T)  # 计算每个点的密度值

#     # Step 4: 绘制散点图
#     print("Plotting t-SNE visualization...")
#     plt.figure(figsize=(10, 8))
#     scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=density, cmap='viridis', alpha=0.6)

#     # Step 5: 图形美化
#     plt.colorbar(scatter, label='Density')  # 添加颜色条
#     plt.title('Visualization of Latent Space with t-SNE', fontsize=16)
#     plt.xlabel('t-SNE Dimension 1', fontsize=12)
#     plt.ylabel('t-SNE Dimension 2', fontsize=12)
#     plt.tight_layout()

#     # Step 6: 保存或展示图像
#     plt.savefig(output_path, dpi=300)
#     plt.close()
#     print(f"Visualization saved to {output_path}")


# # 示例数据生成与绘制
# if __name__ == "__main__":
#     # Step 1: 生成模拟潜在特征数据 (1024, 1, 32, 32)
#     np.random.seed(42)
#     latent_tensor = np.random.rand(1024, 1, 32, 32)  # 随机生成潜在特征

#     # Step 2: 绘制潜在空间分布图
#     plot_latent_space_4d(latent_tensor, output_path="latent_space_visualization.png")