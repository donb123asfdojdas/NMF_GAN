# -*- coding: utf-8 -*-
import os
import scipy.io as sio
import numpy as np
import pandas as pd
import re
import tensorly as tl
from tensorly.decomposition import parafac

def high_order_correlation_completion(matrix, rank=3, max_iter=500, tol=1e-4):
    """
    使用高阶张量分解 (CP分解) 进行矩阵补全，捕捉高阶相关性
    :param matrix: 需要补全的矩阵 (4 x n)，其中缺失值为 nan
    :param rank: 张量分解秩
    :param max_iter: 最大迭代次数
    :param tol: 迭代收敛阈值
    :return: 补全后的矩阵
    """
    matrix = np.vstack(matrix[0]).astype(np.float64)  # 确保为 float64 类型
    nan_mask = np.isnan(matrix)  # 获取缺失数据的位置

    # 初步填充 NaN 值（列均值填充）
    col_mean = np.nanmean(matrix, axis=0)
    col_mean[np.isnan(col_mean)] = 0
    matrix[nan_mask] = np.take(col_mean, np.where(nan_mask)[1])

    # 将矩阵转为张量（增加一个维度）
    tensor = tl.tensor(matrix[np.newaxis, :, :])  # 张量形状为 (1, 4, n)

    # 使用 CP 分解进行张量分解
    factors = parafac(tensor, rank=rank, n_iter_max=max_iter, tol=tol)
    
    # 通过分解结果重构张量
    completed_tensor = tl.cp_to_tensor(factors)

    # 获取补全后的矩阵（降维）
    completed_matrix = completed_tensor[0, :, :]

    # 仅替换缺失值的位置
    matrix[nan_mask] = completed_matrix[nan_mask]

    return matrix

def process_mat_files(input_folder, output_folder, rank=3, max_iter=500, tol=1e-4):
    """
    读取.mat文件，进行高阶相关性矩阵补全，并保存为CSV文件
    :param input_folder: 存放.mat文件的文件夹
    :param output_folder: 输出CSV文件的文件夹
    :param rank: 分解秩
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
                completed_matrix = high_order_correlation_completion(matrix, rank=rank, max_iter=max_iter, tol=tol)
                completed_matrix=completed_matrix/ 10000
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

# 示例用法
input_folder = "../data"   # 存放.mat文件的文件夹
output_folder = "../NMFdata&totalindex/output_HOC" # 输出CSV文件的文件夹
process_mat_files(input_folder, output_folder)
