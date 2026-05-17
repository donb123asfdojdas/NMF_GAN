import os
import scipy.io as sio
import numpy as np
import pandas as pd
import re

def matrix_factorization_completion(matrix, rank=2, max_iter=500, tol=1e-4, lr=0.01, reg=0.1):
    """
    使用矩阵因子分解 (Matrix Factorization) 进行矩阵补全
    :param matrix: 需要补全的矩阵 (4 x n)，其中缺失值为 nan
    :param rank: 近似矩阵的秩
    :param max_iter: 最大迭代次数
    :param tol: 迭代收敛阈值
    :param lr: 学习率
    :param reg: 正则化系数
    :return: 补全后的矩阵
    """
    matrix =  np.vstack(matrix[0]).astype(np.float64)
    nan_mask = np.isnan(matrix)

    # 初始化缺失值
    col_mean = np.nanmean(matrix, axis=0)
    col_mean[np.isnan(col_mean)] = 0
    matrix[nan_mask] = np.take(col_mean, np.where(nan_mask)[1])

    m, n = matrix.shape
    U = np.random.rand(m, rank)  # 随机初始化 U
    V = np.random.rand(n, rank)  # 随机初始化 V

    for _ in range(max_iter):
        # 计算 V (固定 U，最小化 ||M - UV^T||)
        for i in range(n):
            valid_idx = ~nan_mask[:, i]  # 找到非缺失值的行索引
            if np.any(valid_idx):
                U_valid = U[valid_idx, :]
                M_valid = matrix[valid_idx, i]
                V[i, :] = np.linalg.solve(U_valid.T @ U_valid + reg * np.eye(rank), U_valid.T @ M_valid)

        # 计算 U (固定 V，最小化 ||M - UV^T||)
        for i in range(m):
            valid_idx = ~nan_mask[i, :]
            if np.any(valid_idx):
                V_valid = V[valid_idx, :]
                M_valid = matrix[i, valid_idx]
                U[i, :] = np.linalg.solve(V_valid.T @ V_valid + reg * np.eye(rank), V_valid.T @ M_valid)

        # 计算补全矩阵
        completed_matrix = U @ V.T
        diff = np.linalg.norm(completed_matrix - matrix, ord='fro')

        # 更新缺失值
        matrix[nan_mask] = completed_matrix[nan_mask]

        if diff < tol:
            break

    return matrix

def process_mat_files(input_folder, output_folder, rank=2, max_iter=500, tol=1e-4):
    """
    读取.mat文件，进行矩阵因子分解补全，并保存为CSV文件
    :param input_folder: 存放.mat文件的文件夹
    :param output_folder: 输出CSV文件的文件夹
    :param rank: 近似矩阵的秩
    :param max_iter: 最大迭代次数
    :param tol: 迭代收敛阈值
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in os.listdir(input_folder):
        if file.endswith('.mat'):
            file_path = os.path.join(input_folder, file)
            data = sio.loadmat(file_path)

            if 'incompleteData' in data:
                matrix = data['incompleteData']
                completed_matrix = matrix_factorization_completion(matrix, rank=rank, max_iter=max_iter, tol=tol)
                completed_matrix=completed_matrix/10000
                # 生成符合命名要求的文件名
                match = re.search(r'IncompleteFusionData(\d+)', file)
                if match:
                    file_number = match.group(1)
                    output_filename = f"output{file_number}.csv"
                else:
                    output_filename = file.replace('.mat', '.csv')  # 兜底策略

                output_file = os.path.join(output_folder, output_filename)
                pd.DataFrame(completed_matrix).to_csv(output_file, index=False, header=False)
                print(f"Processed: {file} -> {output_file}")

input_folder = "../data"   # 存放.mat文件的文件夹
output_folder = "../NMFdata&totalindex/output_HMM" # 输出CSV文件的文件夹
process_mat_files(input_folder, output_folder)
