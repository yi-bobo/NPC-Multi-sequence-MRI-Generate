import torch
import torch.nn.functional as F
import numpy as np

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
    L = A.shape[0]
    cost_matrix = torch.zeros(N, M).to(A.device)

    for i in range(N):
        for j in range(M):
            if cost_function == "euclidean":
                cost_matrix[i, j] = torch.norm(A[:, i] - B[:, j], p=2)  # 计算欧几里得距离

    cost_matrix2 = torch.cdist(A.T, B.T, p=2)

    return cost_matrix

def optimal_transport(A, B, epsilon=0.1, max_iter=1000):
    """
    使用最优传输算法计算文本特征 A 和 图像特征 B 之间的对齐。
    
    参数：
    - A: 文本特征，形状 [L, N]
    - B: 图像特征，形状 [L, M]
    - epsilon: Sinkhorn 算法的正则化因子
    - max_iter: Sinkhorn 算法的最大迭代次数
    
    返回：
    - aligned_A: 对齐后的文本特征，形状与 B 相同 [L, M]
    """
    L, N = A.shape
    _, M = B.shape
    
    # 计算代价矩阵
    cost_matrix = compute_cost_matrix(A, B, cost_function="euclidean")
    exp_term = torch.exp(-cost_matrix / epsilon)  # 转换成 Sinkhorn 算法的形式
    
    # Sinkhorn 算法：计算最优传输矩阵
    u = torch.ones(cost_matrix.shape[0], 1).to(A.device)  # 初始化 u
    v = torch.ones(cost_matrix.shape[1], 1).to(A.device)  # 初始化 v
    
    for _ in range(max_iter):
        u = 1.0 / (torch.sum(exp_term * v.T, dim=1, keepdim=True)+ 1e-10)  # 更新 u_i
        v = 1.0 / (torch.sum(exp_term.T * u.T, dim=1, keepdim=True)+ 1e-10)  # 更新 v_j
    
    # 计算最优传输矩阵
    transport_matrix = torch.exp(-cost_matrix * epsilon)*u*v.T   # 计算最优传输矩阵
    
    # 将最优传输矩阵应用于 A，以对齐 A 到 B
    aligned_A = torch.matmul(B, transport_matrix.T)  # 对齐特征 A 到 B
    
    return aligned_A

# 示例用法
L = 32  # 5 个样本
N = 21  # 文本特征维度
M = 33  # 图像特征维度

# 随机初始化文本和图像特征
A = torch.randn(L, N)  # 文本特征
B = torch.randn(L, M)  # 图像特征

# 计算最优传输对齐
aligned_A = optimal_transport(A, B)
print(aligned_A.shape)  # 输出对齐后的文本特征形状 [L, M]
