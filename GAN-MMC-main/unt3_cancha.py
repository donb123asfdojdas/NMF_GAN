# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:52:35 2024

@author: Administrator
"""

import torch
from torch import nn
import glob
import pandas as pd
import numpy as np
import os
import re
from sklearn.preprocessing import MinMaxScaler


# 假设这是 Generator 模型的定义
class Config():
    # name = 'shubert'20
    # num = 400
    alpha = 500
    beta = 200
    train_batch_size = 64
    train_number_epochs = 1000
    Data_size = (1, 4, 8)
    Data_area = 32

    lr_G = 0.001  # G网络学习率
    lr_D = 0.008  # D网络学习系率
    gamma = 0.9  # 更新D网络学习率
    epsilon = 1e-15
    use_gpu = True  # set it to True to use GPU and False to use CPU


class Generator(nn.Module):
    """
    Residual Generator that outputs small, data-dependent residuals
    to enhance high-frequency details of completed matrices.
    Input x should have shape (batch_size, 1, depth, width) with values in [0,1].
    Residual amplitude for each sample is computed as mean(x) * residual_factor.
    """

    def __init__(self, dropout_prob=0.3, residual_factor=0.1):
        super(Generator, self).__init__()
        self.dropout_prob = dropout_prob
        self.residual_factor = residual_factor  # 百分比，如0.1表示10%

        # 编码/解码模块构造函数
        def conv3d_block(in_ch, out_ch, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), use_bn=True):
            layers = [
                nn.Conv3d(in_ch, out_ch, kernel_size, stride, padding),
                nn.BatchNorm3d(out_ch) if use_bn else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Dropout3d(p=self.dropout_prob)
            ]
            return nn.Sequential(*layers)

        def upsample3d_block(in_ch, out_ch, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)):
            layers = [
                nn.ConvTranspose3d(in_ch, out_ch, kernel_size, stride, padding),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout3d(p=self.dropout_prob)
            ]
            return nn.Sequential(*layers)

        # Encoder: 提取特征
        self.enc1 = conv3d_block(1, 64)
        self.enc2 = conv3d_block(64, 128)
        self.enc3 = conv3d_block(128, 256)

        # Decoder: 上采样并跳跃连接
        self.dec1 = upsample3d_block(256, 128)
        self.dec2 = upsample3d_block(128, 64)

        # 最终卷积: 生成残差
        self.final_conv = nn.Conv3d(64, 1, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x: (batch_size, 1, depth, width)
        # 1) 保存原始输入
        x = x.view(x.size(0), 1, 4, 8)
        orig = x.clone()
        # 2) 为3D卷积添加高度维度 -> (batch_size,1,depth,1,width)
        x3d = orig.unsqueeze(3)
        # 3) 编码
        e1 = self.enc1(x3d)  # -> (N,64,D,1,W)
        e2 = self.enc2(e1)  # -> (N,128,D,1,W)
        e3 = self.enc3(e2)  # -> (N,256,D,1,W)

        # 4) 解码 + 跳跃连接
        d1 = self.dec1(e3) + e2  # -> (N,128,D,1,W)
        d2 = self.dec2(d1) + e1  # -> (N,64,D,1,W)

        # 5) 生成原始残差 r in [-1,1]
        r = self.final_conv(d2)  # (N,1,D,1,W)
        # print("r>",r)
        r = self.tanh(r)
        # print("r>>>",r)
        # 6) 计算每个矩阵的均值 -> (N,1,1,1)
        # mean_val = orig.mean(dim=(2,3), keepdim=True)
        # # 动态缩放残差幅度
        # scale = mean_val * self.residual_factor
        # r = r * scale.unsqueeze(3) if scale.dim() == 4 else r * scale

        # 7) 去掉高度维度 -> (N,1,D,W)
        r = r.squeeze(3)

        # 8) 残差相加并 clamp 到 [0,1]
        out = torch.clamp(orig + r, 0.0, 1.0)
        return out


device = torch.device('cpu')
# 初始化模型实例
generator = Generator().to(device)

# 加载保存的模型参数
generator.load_state_dict(torch.load('best_model/history/unet3/tanh.pth', map_location=device))

# 将模型设置为评估模式
generator.eval()


# 自定义排序函数，用于提取文件名中的数字并按数字大小排序
def natural_sort_key(file_path):
    # 使用正则表达式提取文件名中的数字
    match = re.search(r"(\d+)", file_path)
    return int(match.group(1)) if match else float('inf')  # 提取数字部分用于排序


# -------------------------------获取H-----------------------data/Generate/H*.csv
file_paths = sorted(glob.glob("../HCP-IMSC-main/GAN/H*.csv"), key=natural_sort_key)
H = []
for file_path in file_paths:
    data = pd.read_csv(file_path, header=None).values
    H.append(data)
H_data = torch.tensor(H, dtype=torch.float32)
H_data = H_data.float().unsqueeze(1).to(device)

# ------------------------------获取L------------------------
file_paths = sorted(glob.glob("../HCP-IMSC-main/GAN/L*.csv"), key=natural_sort_key)

# 获取原始的L数据
last_L = []
for file_path in file_paths:
    data = pd.read_csv(file_path, header=None).values
    last_L.append(data)
# 获取归一化后的L数据
L = []
scalers = []  # 保存每个矩阵自己的Scaler
for file_path in file_paths:
    # 2. 读成 DataFrame（shape = [4, 8]）
    df = pd.read_csv(file_path, header=None)
    # 3. 对每一列做 Min–Max 归一化（映射到 [0,1]）
    scaler = MinMaxScaler(feature_range=(0, 1))
    # fit_transform 接受形如 [n_samples, n_features] 的数组
    data_norm = scaler.fit_transform(df.values)
    scalers.append(scaler)  # 保存这个Scaler，用来逆归一化
    L.append(data_norm)
L_data = torch.tensor(L, dtype=torch.float32).to(device)

# -----------------------------获取M-------------------------
file_paths = sorted(glob.glob("../HCP-IMSC-main/GAN/M*.csv"), key=natural_sort_key)
M_all = []
for file_path in file_paths:
    data = pd.read_csv(file_path, header=None).values
    M_all.append(data)
M_data = torch.tensor(M_all)
M_data = M_data.float().unsqueeze(1).to(device)


# ----------------------------获取output对应文件名的数字部分-----
def get_sorted_numbers_from_filenames(folder_path):
    files = os.listdir(folder_path)  # 获取文件夹内所有文件名
    pattern = re.compile(r"L(\d+)\.csv")  # 正则匹配 L+数字.csv 格式

    numbers = []
    for file in files:
        match = pattern.match(file)
        if match:
            numbers.append(int(match.group(1)))  # 提取数字部分并转换为整数

    return sorted(numbers)  # 按照数字大小排序


# 示例用法
folder_path = "../HCP-IMSC-main/GAN/"
sorted_numbers = get_sorted_numbers_from_filenames(folder_path)
###############################################################
# 通过模型进行预测
x = generator(L_data)
x = x.squeeze(1).detach().cpu().numpy()
x_recovered = []
for i in range(len(x)):  # 遍历每个样本
    scaler = scalers[i]  # 拿到对应的Scaler
    x_orig = scaler.inverse_transform(x[i])  # 逆归一化
    x_recovered.append(x_orig)

x_recovered = np.array(x_recovered)  # [batch_size, 4, 8]，原始数据
# print(x_recovered)
output_test = x_recovered
last_L = np.array(last_L)
M_test = np.array(M_data).squeeze(1)
H_test = torch.detach(H_data).numpy().squeeze(1)

B, H, W = output_test.shape  # B=batch_size, H=4, W=8
# 遍历每个样本
# for i in range(B):
#     # M_test[i] 的形状 (4,8)，对每一列统计 1 的数量
#     col_counts = M_test[i].sum(axis=0)  # 结果形状 (8,)

#     # 找到需要替换的列索引：恰好出现 1 次
#     cols_to_replace = np.where((col_counts > 0) & (col_counts < 2))[0]

#     if cols_to_replace.size == 0:
#         continue  # 该样本没有需替换的列

#     # 构造布尔掩码，形状 (4,8)，只在这些列上且 M_test==1 的位置为 True
#     mask = np.zeros((H, W), dtype=bool)
#     mask[:, cols_to_replace] = (M_test[i][:, cols_to_replace] == 1)

#     # 在 mask 为 True 的位置，将 output_test 替换为 last_L
#     output_test[i][mask] = last_L[i][mask]
# print(M_test.shape)
# print(x_recovered.shape)
# print(last_L.shape)
# print(M_test*H_test)
# print("------------")
# print(M_test*last_L)
# print(abs(M_test*H_test-M_test*last_L))
# print("-------------------------")
# print(H_test)
# print(abs(M_test*H_test-M_test*last_L)/H_test)
# print("############################")
# print(abs(M_test*H_test-M_test*last_L))
# print(output_test)
# print("-------------------")
# print(abs(M_test*H_test-M_test*output_test))
# print("------------------------")
# print(abs(M_test*H_test-M_test*output_test)/H_test)
# print("----------------------------")
print("生成的数据与高精度之间的均方根误差:",
      np.sum(abs(M_test * H_test - M_test * output_test) / H_test) / np.sum(M_test))
print('-----------------')
print("低精度与高精度之间均方根误差:", np.sum(abs(M_test * H_test - M_test * last_L) / H_test) / np.sum(M_test))

M_data = M_data.squeeze(1).detach().cpu().numpy()
# 遍历每个数据，逐一保存

for i in range(x_recovered.shape[0]):

    x_np = ((M_data[i] * x_recovered[i]) + ((1 - M_data[i]) * last_L[i]))
    # 从后向前扫描，删除重复的列
    count = 1
    while x_np.shape[1] > 1:
        # 如果最后一列与前一列完全相同，则删除最后一列
        if np.all(L[i][:, -count] == L[i][:, -(count + 1)]):
            x_np = x_np[:, :-1]  # 删除最后一列
            count = count + 1
        else:
            break  # 当前列与前一列不同，停止删除
    # 保存为 CSV 文件
    file_path = os.path.join("../HCP-IMSC-main/NMFdata&totalindex/output/", f"output{sorted_numbers[i]}.csv")
    pd.DataFrame(x_np).to_csv(file_path, header=False, index=False)
    # print(f"Saved {file_path} successfully!")

print("All files have been saved.")