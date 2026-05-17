import os
import scipy.io as sio
import numpy as np
import pandas as pd
from numpy.linalg import svd
import re

def nuclear_norm_minimization(matrix, max_iter=500, tol=1e-4, lambda_factor=0.1):
    """
    使用核范数最小化（Soft-Impute）进行矩阵补全
    :param matrix: 需要补全的矩阵 (4 x n)，其中缺失值为 nan
    :param max_iter: 最大迭代次数
    :param tol: 迭代收敛阈值
    :param lambda_factor: 控制奇异值收缩的参数
    :return: 补全后的矩阵
    """
    matrix = np.vstack(matrix[0]).astype(np.float64)  # 确保为 float64 类型
    nan_mask = np.isnan(matrix)

    # 初步填充 NaN 值（列均值填充）
    col_mean = np.nanmean(matrix, axis=0)
    col_mean[np.isnan(col_mean)] = 0  # 处理某些列全是 NaN 的情况
    matrix[nan_mask] = np.take(col_mean, np.where(nan_mask)[1])

    # 计算动态正则化参数 lambda
    lambda_val = lambda_factor * np.mean(svd(matrix, compute_uv=False))

    # 迭代优化
    for _ in range(max_iter):
        U, S, Vt = svd(matrix, full_matrices=False)
        S = np.maximum(S - lambda_val, 0)  # 奇异值收缩（Soft-Thresholding）
        completed_matrix = U @ np.diag(S) @ Vt  # 近似补全的矩阵

        # 仅更新缺失值位置
        diff = np.linalg.norm(completed_matrix - matrix, ord='fro')
        matrix[nan_mask] = completed_matrix[nan_mask]

        if diff < tol:  # 判断是否收敛
            break

    return matrix

def process_mat_files(input_folder, output_folder, max_iter=500, tol=1e-4):
    """
    读取.mat文件，进行核范数最小化补全，并保存为CSV文件
    :param input_folder: 存放.mat文件的文件夹
    :param output_folder: 输出CSV文件的文件夹
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
                matrix=matrix
                print(matrix)
                completed_matrix = nuclear_norm_minimization(matrix, max_iter=max_iter, tol=tol)
                completed_matrix=completed_matrix/ 10000
                # 生成符合命名要求的文件名
                match = re.search(r'IncompleteFusionData(\d+)', file)
                if match:
                    file_number = match.group(1)
                    output_filename = f"output{file_number}.csv"
                    #output_filename = f"L{file_number}.csv"
                else:
                    output_filename = file.replace('.mat', '.csv')  # 兜底策略
                    raise ValueError(f"{file_number}文件未找到",file_number)
                output_file = os.path.join(output_folder, output_filename)
                pd.DataFrame(completed_matrix).to_csv(output_file, index=False, header=False)
                print(f"Processed: {file} -> {output_file}")

# 示例用法
input_folder = "../data"   # 存放.mat文件的文件夹
output_folder = "../NMFdata&totalindex/output" # 输出CSV文件的文件夹
#output_folder = '../GAN'
process_mat_files(input_folder, output_folder)