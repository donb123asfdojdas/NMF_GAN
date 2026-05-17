# -*- coding: utf-8 -*-
"""
Inference for DCAE 4x8 matrix completion

- 读取 ../../HCP-IMSC-main/GAN/ 目录中的 L*, M*, H* CSV
- 使用训练好的 dcae_4x8.pth 做矩阵补全
- 输出补全后的 CSV 到 ../HCP-IMSC-main/NMFdata&totalindex/output_DCAE/
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# --------------------- DCAE 模型定义（与训练时保持一致） ---------------------
class DCAE(nn.Module):
    """
    深度补全自动编码器（DCAE）模型
    输入: concat(L_clean.flatten(), M.flatten()) -> [64]
    输出: 补全的矩阵 (H_flat) -> [32]
    """

    def __init__(self, input_dim=64, hidden=(128, 64, 32), output_dim=32):
        super().__init__()
        dims = [input_dim] + list(hidden) + [output_dim]
        layers = []

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        hidden_acts = []
        out = x
        for layer in self.net:
            out = layer(out)
            if isinstance(layer, nn.ReLU):
                hidden_acts.append(out)
        return out, hidden_acts


# --------------------- 文件排序工具 ---------------------
def natural_sort_key(file_path):
    match = re.search(r"(\d+)", os.path.basename(file_path))
    return int(match.group(1)) if match else float("inf")


def get_sorted_numbers_from_filenames(folder_path):
    files = os.listdir(folder_path)
    pattern = re.compile(r"L(\d+)\.csv")
    numbers = []
    for f in files:
        m = pattern.match(f)
        if m:
            numbers.append(int(m.group(1)))
    return sorted(numbers)


# --------------------- 主推理流程 ---------------------
def main():
    device = torch.device("cuda")  # 有 GPU 可改为 torch.device("cuda")

    # 1. 加载训练好的模型
    model = DCAE().to(device)
    state = torch.load("dcae_model.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()
    print("模型已加载: dcae_4x8.pth")

    # 2. 路径配置
    gan_folder = "../../HCP-IMSC-main/GAN/"
    output_folder = "../../HCP-IMSC-main/NMFdata&totalindex/output_DMF/"
    os.makedirs(output_folder, exist_ok=True)

    # 3. 读取 H 数据
    H_files = sorted(glob.glob(os.path.join(gan_folder, "H*.csv")),
                     key=natural_sort_key)
    H_list = [pd.read_csv(fp, header=None).values.astype(np.float32)
              for fp in H_files]
    H_np = np.stack(H_list, axis=0)  # [N,4,8]
    H_tensor = torch.tensor(H_np, dtype=torch.float32).to(device)

    # 4. 读取 L 数据（原始，不做归一化）
    L_files = sorted(glob.glob(os.path.join(gan_folder, "L*.csv")),
                     key=natural_sort_key)
    L_list = [pd.read_csv(fp, header=None).values.astype(np.float32)
              for fp in L_files]
    L_np = np.stack(L_list, axis=0)  # [N,4,8]

    # 5. 读取 M 数据
    M_files = sorted(glob.glob(os.path.join(gan_folder, "M*.csv")),
                     key=natural_sort_key)
    M_list = [pd.read_csv(fp, header=None).values.astype(np.float32)
              for fp in M_files]
    M_np = np.stack(M_list, axis=0)  # [N,4,8]

    # 一些形状检查
    N = L_np.shape[0]
    assert H_np.shape == (N, 4, 8)
    assert M_np.shape == (N, 4, 8), "L/M/H 样本数量或形状不一致"

    # 6. 构造模型输入: L_clean = L * (1 - M), x = concat(L_clean, M)
    L_clean = L_np * (1.0 - M_np)             # 去掉缺失位置的随机值
    L_flat = L_clean.reshape(N, -1)           # [N,32]
    M_flat = M_np.reshape(N, -1)              # [N,32]
    x = np.concatenate([L_flat, M_flat], 1)   # [N,64]

    x_tensor = torch.tensor(x, dtype=torch.float32).to(device)

    # 7. 前向推理
    with torch.no_grad():
        pred_flat, _ = model(x_tensor)        # [N,32]
    pred_np = pred_flat.cpu().numpy().reshape(N, 4, 8)  # 预测的完整矩阵

    # 8. 根据掩码 M 完成矩阵:
    #    缺失位置(M==1) 用预测值，观测位置(M==0) 保留原始 L
    completed_np = M_np * pred_np + (1.0 - M_np) * L_np  # [N,4,8]

    # 9. 计算误差（只在缺失位置上）
    H_np = H_tensor.cpu().numpy()  # [N,4,8]
    missing_mask = M_np  # 1=缺失

    # 避免除 0
    total_missing = missing_mask.sum()
    if total_missing > 0:
        # 生成数据 vs 高精度
        err_gen = np.abs(missing_mask * H_np - missing_mask * completed_np) / (H_np + 1e-12)
        rmse_gen = err_gen.sum() / total_missing

        # 低精度 L vs 高精度
        err_low = np.abs(missing_mask * H_np - missing_mask * L_np) / (H_np + 1e-12)
        rmse_low = err_low.sum() / total_missing

        print("生成的数据与高精度之间的均方误差(缺失部分):", rmse_gen)
        print("低精度与高精度之间的均方误差(缺失部分):", rmse_low)
    else:
        print("掩码中没有缺失位置(M==1)，不计算误差。")

    # 10. 获取输出文件编号
    sorted_numbers = get_sorted_numbers_from_filenames(gan_folder)

    # 11. 逐样本保存补全后的 CSV
    for i in range(N):
        x_np = completed_np[i].copy()  # [4,8]

        # 可选：和你原代码一样，删除末尾连续重复列（基于原始 L）
        count = 1
        while x_np.shape[1] > 1:
            if np.all(L_np[i][:, -count] == L_np[i][:, -(count + 1)]):
                x_np = x_np[:, :-1]
                count += 1
            else:
                break

        out_idx = sorted_numbers[i] if i < len(sorted_numbers) else i
        out_path = os.path.join(output_folder, f"output{out_idx}.csv")
        pd.DataFrame(x_np).to_csv(out_path, header=False, index=False)

    print("所有补全后的 CSV 文件已保存到:", output_folder)


if __name__ == "__main__":
    main()
