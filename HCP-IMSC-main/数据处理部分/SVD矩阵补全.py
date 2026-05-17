import os
import scipy.io as sio
import numpy as np
import pandas as pd
from numpy.linalg import svd
import re
def svd_matrix_completion(matrix, rank=2):
    """
    使用SVD进行矩阵补全
    :param matrix: 需要补全的矩阵 (4 x n)，其中缺失值为 nan
    :param rank: 近似矩阵的秩
    :return: 补全后的矩阵
    """
    # 确保matrix是数值类型的ndarray
    matrix = np.vstack(matrix[0]).astype(np.float64)
    
    # 计算均值填充初始缺失值（列均值）
    nan_mask = np.isnan(matrix)
    col_mean = np.nanmean(matrix, axis=0)
    
    # 处理某些列全是NaN的情况
    col_mean[np.isnan(col_mean)] = 0
    matrix[nan_mask] = np.take(col_mean, np.where(nan_mask)[1])
    
    # 进行SVD分解
    U, S, Vt = svd(matrix, full_matrices=False)
    S = np.diag(S[:rank])  # 只保留前 rank 个奇异值
    U = U[:, :rank]
    Vt = Vt[:rank, :]
    
    # 计算近似矩阵
    completed_matrix = U @ S @ Vt
    return completed_matrix

def process_mat_files(input_folder, output_folder, rank=2):
    """
    读取.mat文件，进行SVD补全，并保存为CSV文件
    :param input_folder: 存放.mat文件的文件夹
    :param output_folder: 输出CSV文件的文件夹
    :param rank: SVD分解时的秩
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for file in os.listdir(input_folder):
        if file.endswith('.mat'):
            file_path = os.path.join(input_folder, file)
            data = sio.loadmat(file_path)
            
            if 'incompleteData' in data:
                matrix = data['incompleteData']
                #print(matrix)
                completed_matrix = svd_matrix_completion(matrix, rank=rank)
                completed_matrix=completed_matrix/10000
                # 保存为CSV
                match = re.search(r'IncompleteFusionData(\d+)', file)
                if match:
                    file_number = match.group(1)  # 提取文件编号
                    new_filename = f"output{file_number}.csv"
                else:
                    new_filename = file.replace('.mat', '.csv')  # 兜底策略
                
                output_file = os.path.join(output_folder, new_filename)
                pd.DataFrame(completed_matrix).to_csv(output_file, index=False, header=False)
                print(f"Processed: {file} -> {output_file}")
# 示例用法
input_folder = "../data"   # 存放.mat文件的文件夹
output_folder = "../NMFdata&totalindex/output_SVD" # 输出CSV文件的文件夹
process_mat_files(input_folder, output_folder)
