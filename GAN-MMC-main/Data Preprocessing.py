import pandas as pd
import numpy as np
import os

# 定义填充函数 - 边缘填充
def fill_with_edge_padding(matrix):
    rows, cols = matrix.shape
    if cols == 8:
        return matrix
    filled_matrix = np.zeros((4, 8))
    filled_matrix[:, :cols] = matrix
    for i in range(4):
        filled_matrix[i, cols:] = matrix[i, cols - 1]
    return filled_matrix

# 定义填充函数 - 零填充
def fill_with_zero_padding(matrix):
    rows, cols = matrix.shape
    if cols == 8:
        return matrix
    filled_matrix = np.zeros((4, 8))
    filled_matrix[:, :cols] = matrix
    return filled_matrix

# 设置文件路径
input_path = "../HCP-IMSC-main/GAN/"
#input_path="data/train_data"
input_files = sorted([f for f in os.listdir(input_path) if f.endswith(".csv")])

# 获取文件名列表
L_files = sorted([f for f in input_files if f.startswith("L")])
M_files = sorted([f for f in input_files if f.startswith("M")])

# 提取文件编号
L_file_numbers = {f[1:-4] for f in L_files}  # 提取 L 文件的编号
M_file_numbers = {f[1:-4] for f in M_files}  # 提取 M 文件的编号

# 检查 L 和 M 文件是否一一对应
if L_file_numbers != M_file_numbers:
    unmatched_files = L_file_numbers.symmetric_difference(M_file_numbers)
    raise ValueError(f"文件不匹配，未找到对应的文件编号: {unmatched_files}")

# 逐个处理文件
for file_name in input_files:
    file_path = os.path.join(input_path, file_name)
    df = pd.read_csv(file_path, header=None)
    matrix = df.values
    rows, cols = matrix.shape
    
    # 检查是否为4行，且列数不超过8
    if rows == 4 and 0 < cols < 8:
        # 判断文件类型并选择填充方法
        if file_name.startswith("L") or file_name.startswith("H"):
            # L 和 H 文件采用重复边缘填充
            filled_matrix = fill_with_edge_padding(matrix)
            print(f"{file_name} 使用边缘填充。")
        elif file_name.startswith("M"):
            # M 文件使用零填充
            filled_matrix = fill_with_zero_padding(matrix)
            print(f"{file_name} 使用零填充。")
        else:
            print(f"{file_name} 不符合 L/H/M 类型，未处理。")
            continue
        
        # 保存填充后的矩阵，覆盖原文件
        filled_df = pd.DataFrame(filled_matrix)
        filled_df.to_csv(file_path, header=False, index=False)
        print(f"{file_name} 已填充并覆盖保存。")
    else:
        print(f"{file_name} 不符合要求，未处理")

print("所有文件处理完毕。")
