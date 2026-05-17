import os
import numpy as np
import pandas as pd
output_folder = "../NMFdata&totalindex/output_HMM"
gan_folder = "../GAN"
# 获取所有 outputi.csv 文件
output_files = sorted([f for f in os.listdir(output_folder) if f.startswith("output") and f.endswith(".csv")])

f_norm_errors = []

for output_file in output_files:
    i = output_file[6:-4]  # 提取数字 i
    output_path = os.path.join(output_folder, output_file)
    h_path = os.path.join(gan_folder, f"H{i}.csv")
    m_path = os.path.join(gan_folder, f"M{i}.csv")

    # 读取 CSV 文件
    O = pd.read_csv(output_path, header=None).values
    H = pd.read_csv(h_path, header=None).values
    M = pd.read_csv(m_path, header=None).values

    # 确保 O 维度是 (4, 8)，如果列数不足 8，填充 0
    if O.shape[1] < 8:
        O = np.pad(O, ((0, 0), (0, 8 - O.shape[1])), mode='constant', constant_values=0)
    if H.shape[1] < 8:
        H = np.pad(H, ((0, 0), (0, 8 - H.shape[1])), mode='constant', constant_values=0)
    if M.shape[1] < 8:
        M = np.pad(M, ((0, 0), (0, 8 - M.shape[1])), mode='constant', constant_values=0)
    # 计算误差矩阵（仅考虑 M == 1 的部分）
    E = M * (O - H)

    # 计算 F 范数
    f_norm = np.linalg.norm(E, 'fro')
    f_norm_errors.append(f_norm)

# 计算平均 F 范数误差
average_f_norm = np.mean(f_norm_errors)
print(f"所有 output 补全矩阵的平均 F 范数误差: {average_f_norm}")
'''
SNR=7
MF-GAN 0.006644195645744668
SVD 0.013356770462198392
HOC 0.007817740534488703
HMM 0.019080205735876851

#SINR=10
MF-GAN 0.005050568668182190
SVD 0.011316530061661555
HOC 0.007874156345679176
HMM 0.017608370551411612

SNR=13
MF-GAN 0.005004274046763363
SVD 0.010200021067694872
HOC 0.007783363945710668
HMM 0.017177441861984264
'''