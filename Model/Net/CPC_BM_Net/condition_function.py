import torch

def compute_cost_matrix(A, B, cost_function="euclidean"):
    """
    计算文本特征 A 和 图像特征 B 之间的代价矩阵，形状为 [L, L]。
    
    参数：
    - A: 文本特征，形状 [L, N]，L 是样本数，N 是特征维度
    - B: 图像特征，形状 [L, M]，L 是样本数，M 是特征维度
    - cost_function: 选择代价函数类型 ("euclidean" | "cosine")
    
    返回：
    - cost_matrix: 代价矩阵，形状为 [N, M]，每对文本和图像特征之间的代价
    """
    L1, N = A.shape
    L2, M = B.shape
    assert L1 == L2  # "Features must have the same dimension."
    cost_matrix = torch.zeros(N, M).to(A.device)
    cost_matrix = torch.cdist(A.T, B.T, p=2)  # 计算欧几里得距离

    return cost_matrix