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
# 假设这是 Generator 模型的定义
class Config():
    #name = 'shubert'20
    #num = 400
    alpha = 500
    beta = 200
    train_batch_size = 64
    train_number_epochs = 1000
    Data_size = (1, 4, 8)
    Data_area = 32
    
    lr_G=0.001#G网络学习率
    lr_D=0.008#D网络学习系率
    gamma=0.9#更新D网络学习率
    epsilon = 1e-15
    use_gpu = True  # set it to True to use GPU and False to use CPU

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        # Dropout概率
        self.dropout_prob = 0.2

        # 定义3D卷积块，使用 (3, 1, 1) 卷积核和新的填充逻辑
        def conv3d_block(in_channels, out_channels, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), use_bn=True):
            layers = [
                nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm3d(out_channels) if use_bn else nn.Identity(),
                nn.LeakyReLU(0.2),
                nn.Dropout3d(p=self.dropout_prob)  # 使用类中定义的Dropout概率
            ]
            return nn.Sequential(*layers)

        # 定义上采样块（保持 W 维度不变）
        def upsample3d_block(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
                nn.BatchNorm3d(out_channels),
                nn.LeakyReLU(0.2),
                nn.Dropout3d(p=self.dropout_prob)  # 使用类中定义的Dropout概率
            )

        # 编码路径 (Encoder)
        self.enc1 = conv3d_block(1, 64)     # 输入: 1x4x1x8 -> 输出: 64x4x1x8
        self.enc2 = conv3d_block(64, 128)   # 输出: 128x4x1x8
        self.enc3 = conv3d_block(128, 256)  # 输出: 256x4x1x8
        # 解码路径 (Decoder)
        self.dec1 = upsample3d_block(256, 128)   # 输出: 128x4x1x8
        self.dec2 = upsample3d_block(128, 64)    # 输出: 64x4x1x8
        self.final_conv = nn.Conv3d(64, 1, kernel_size=(3, 1, 1), padding=(1, 0, 0))  # 输出: 1x4x1x8

        # 输出激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. 输入格式转换
        x = x.view(x.size(0), 1, 4, 8)  # 转换为 (batch_size, channels, depth, height, width)
        x = x.unsqueeze(3)  # 在高度维度 H 上增加一个维度，变为 (batch_size, 1, 4, 1, 8)

        # 2. 编码路径 (3D卷积编码)
        x1 = self.enc1(x)   # 64x4x1x8
        x2 = self.enc2(x1)  # 128x4x1x8
        x3 = self.enc3(x2)  # 256x4x1x8

        # 3. 解码路径 (3D卷积解码)
        d1 = self.dec1(x3) + x2  # 跳跃连接，128x4x1x8
        d2 = self.dec2(d1) + x1  # 跳跃连接，64x4x1x8

        # 4. 最后一层卷积和激活函数
        out = self.final_conv(d2)  # 1x4x1x8
        out = self.sigmoid(out)    # 激活函数

        # 5. 调整输出形状为 (batch_size, 1, 4, 8)
        out = out.squeeze(3)  # 去掉 H 维度
        return out
'''
#unt2
class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        # Dropout概率
        self.dropout_prob = 0.3

        # 定义3D卷积块，使用 (3, 1, 1) 卷积核和新的填充逻辑
        def conv3d_block(in_channels, out_channels, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), use_bn=True):
            layers = [
                nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm3d(out_channels) if use_bn else nn.Identity(),
                #nn.LeakyReLU(0.2),
                nn.ReLU(),
                nn.Dropout3d(p=self.dropout_prob)  # 使用类中定义的Dropout概率
            ]
            return nn.Sequential(*layers)

        
# 定义上采样块（保持 W 维度不变）
        def upsample3d_block(in_channels, out_channels,kernel_size=(3, 1, 1), stride=1,padding=(1, 0, 0)):
            return nn.Sequential(
                nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride,padding),
                nn.BatchNorm3d(out_channels),
                #nn.LeakyReLU(0.2),
                nn.ReLU(),
                nn.Dropout3d(p=self.dropout_prob)  # 使用类中定义的Dropout概率
            )
        # 编码路径 (Encoder)
        self.enc1 = conv3d_block(1, 64)     # 输入: 1x4x1x8 -> 输出: 64x4x1x8
        self.enc2 = conv3d_block(64, 128,kernel_size=(3, 1, 1),padding=(1, 0, 0))   # 输出: 128x4x1x8
        self.enc3 = conv3d_block(128, 256,kernel_size=(3, 1, 1),padding=(1, 0, 0))  # 输出: 256x4x1x8
        self.enc4 = conv3d_block(256, 512,kernel_size=(3, 1, 3),padding=(1, 0, 1))  # 输出: 256x4x1x8
        # 解码路径 (Decoder)
        self.dec0 = upsample3d_block(512, 256,kernel_size=(3, 1, 1),padding=(1, 0, 0))
        self.dec1 = upsample3d_block(256, 128,kernel_size=(3, 1, 1),padding=(1, 0, 0))   # 输出: 128x4x1x8
        self.dec2 = upsample3d_block(128, 64,kernel_size=(3, 1, 1),padding=(1, 0, 0))    # 输出: 64x4x1x8
        self.final_conv = nn.Conv3d(64, 1,kernel_size=(3,1,1), padding=(1,0,0))  # 输出: 1x4x1x8

        # 输出激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. 输入格式转换
        x = x.view(x.size(0), 1, 4, 8)  # 转换为 (batch_size, channels, depth, height, width)
        x = x.unsqueeze(3)  # 在高度维度 H 上增加一个维度，变为 (batch_size, 1, 4, 1, 8)

        # 2. 编码路径 (3D卷积编码)
        x1 = self.enc1(x)   # 64x4x1x8
        x2 = self.enc2(x1)  # 128x4x1x8
        x3 = self.enc3(x2)  # 256x4x1x8
        x4 = self.enc4(x3)
        # 3. 解码路径 (3D卷积解码)
        d0 = self.dec0(x4) + x3
        d1 = self.dec1(x3) + x2  # 跳跃连接，128x4x1x8
        d2 = self.dec2(d1) + x1  # 跳跃连接，64x4x1x8

        # 4. 最后一层卷积和激活函数
        out = self.final_conv(d2)  # 1x4x1x8
        out = self.sigmoid(out)    # 激活函数

        # 5. 调整输出形状为 (batch_size, 1, 4, 8)
        out = out.squeeze(3)  # 去掉 H 维度
        return out
'''
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device=torch.device('cpu')
# 初始化模型实例
generator = Generator().to(device)

# 加载保存的模型参数
generator.load_state_dict(torch.load('best_model/history/unet3/MF-GAN.pth',map_location=device))

# 将模型设置为评估模式
generator.eval()
# 自定义排序函数，用于提取文件名中的数字并按数字大小排序
def natural_sort_key(file_path):
    # 使用正则表达式提取文件名中的数字
    match = re.search(r"(\d+)", file_path)
    return int(match.group(1)) if match else float('inf')  # 提取数字部分用于排序

#-------------------------------获取H-----------------------data/Generate/H*.csv
file_paths = sorted(glob.glob("../HCP-IMSC-main/GAN/H*.csv"), key=natural_sort_key)
H = []
for file_path in file_paths:
    data = pd.read_csv(file_path, header=None).values
    H.append(data)
H_data = torch.tensor(H, dtype=torch.float32)
H_data = H_data.float().unsqueeze(1).to(device)

#------------------------------获取L------------------------
file_paths = sorted(glob.glob("../HCP-IMSC-main/GAN/L*.csv"), key=natural_sort_key)

L = []
last_L = []
xmin = 0
xmax = 0
for file_path in file_paths:
    data = pd.read_csv(file_path, header=None).values
    last_L.append(data)
    xmin = np.min(data)
    xmax = np.max(data)
    data = (data - xmin + Config.epsilon) / (xmax - xmin)
    L.append(data)
L_data = torch.tensor(L, dtype=torch.float32).to(device)

#-----------------------------获取M-------------------------
file_paths = sorted(glob.glob("../HCP-IMSC-main/GAN/M*.csv"), key=natural_sort_key)
M_all = []
for file_path in file_paths:
    data = pd.read_csv(file_path, header=None).values
    M_all.append(data)
M_data = torch.tensor(M_all)
M_data = M_data.float().unsqueeze(1).to(device)
#----------------------------获取output对应文件名的数字部分-----
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

output_x=x* (xmax - xmin) + xmin
last_L=np.array(last_L)
MSE_test_loss = torch.sqrt(torch.mean((M_data * H_data - M_data * output_x) ** 2) / torch.mean(M_data))
origin_loss = torch.sqrt(torch.mean((M_data * H_data - M_data * last_L) ** 2) / torch.mean(M_data))
print("生成的数据与高精度之间的均方根误差:",np.array((MSE_test_loss ).detach().numpy())/10000)
print('-----------------')
print("低精度与高精度之间均方根误差:",np.array((origin_loss ).detach().numpy())/10000)

M_data=M_data.squeeze(1).detach().cpu().numpy()
output_x=output_x.squeeze(1).detach().cpu().numpy()
# 遍历每个数据，逐一保存

for i in range(output_x.shape[0]):

    x_np = ((M_data[i] * output_x[i]) + ((1 - M_data[i]) * last_L[i]))/ 10000
    # 从后向前扫描，删除重复的列
    count=1
    while x_np.shape[1] > 1:
        
        # 如果最后一列与前一列完全相同，则删除最后一列
        if np.all(L[i][:, -count] == L[i][:, -(count+1)]):  
            x_np = x_np[:, :-1]  # 删除最后一列
            count=count+1
        else:
            break  # 当前列与前一列不同，停止删除
    
    # 保存为 CSV 文件
    file_path = os.path.join("../HCP-IMSC-main/NMFdata&totalindex/output/", f"output{sorted_numbers[i]}.csv")
    pd.DataFrame(x_np).to_csv(file_path, header=False, index=False)
    #print(f"Saved {file_path} successfully!")

print("All files have been saved.")