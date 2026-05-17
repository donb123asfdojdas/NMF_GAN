# -*- coding: utf-8 -*-
"""
Inference for AEMC-NE 4x8 matrix completion

- 读取 ../../HCP-IMSC-main/GAN/ 目录中的 L*, M*, H* CSV
- 使用训练好的 aemcne_4x8_model.pth 做矩阵补全
- 输出补全后的 CSV 到 ../../HCP-IMSC-main/NMFdata&totalindex/output_AEMCNE/
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# --------------------- AEMC-NE 模型定义（与训练时保持一致） ---------------------
class ElementWiseNN(nn.Module):
    """
    逐元素小网络（所有元素共享参数）:
    输入: 标量 scalar
    输出: 标量 scalar
    """
    def __init__(self, hidden_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, D) 或 (B, 4, 8)
        返回形状与 x 相同
        """
        orig_shape = x.shape          # 例如 (B, 32) 或 (B, 4, 8)
        x_flat = x.view(-1, 1)        # (B*D, 1)
        out = self.net(x_flat)        # (B*D, 1)
        return out.view(*orig_shape)  # reshape 回原形状


class AEMCNE4x8(nn.Module):
    """
    AEMC-NE 风格 4x8 模型:
    - 主自编码器：32 -> latent_dim -> 32
    - 逐元素网络：对 decoder 输出的每个标量再做一次非线性变换
    """
    def __init__(self,
                 n_rows=4,
                 n_cols=8,
                 hidden_dim=64,
                 latent_dim=16,
                 elem_hidden_dim=8):
        super().__init__()
        self.n_rows = n_rows
        self.n_cols = n_cols
        input_dim = n_rows * n_cols

        # encoder: 32 -> hidden_dim -> latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )

        # decoder: latent_dim -> hidden_dim -> 32
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

        # 逐元素小网络
        self.elem_nn = ElementWiseNN(hidden_dim=elem_hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 4, 8) 或 (B, 32)
        返回: (B, 4, 8)
        """
        if x.dim() == 3:
            B = x.shape[0]
            x_flat = x.view(B, -1)  # (B, 32)
        else:
            B = x.shape[0]
            x_flat = x              # (B, 32)

        z = self.encoder(x_flat)        # (B, latent_dim)
        recon = self.decoder(z)         # (B, 32)
        recon = self.elem_nn(recon)     # (B, 32)

        recon = recon.view(B, self.n_rows, self.n_cols)  # (B, 4, 8)

        # 注意：这里和训练代码保持一致。
        # 如果你在训练时加了 sigmoid，这里也要加。
        # recon = torch.sigmoid(recon)

        return recon


# --------------------- 文件排序工具 ---------------------
def natural_sort_key(file_path):
    """按文件名中的数字自然排序，例如 L0, L1, L10"""
    match = re.search(r"(\d+)", os.path.basename(file_path))
    return int(match.group(1)) if match else float("inf")


def get_sorted_numbers_from_filenames(folder_path):
    """从 L<num>.csv 文件名中提取 num，并排序返回 [num0, num1, ...]"""
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载训练好的模型
    model = AEMCNE4x8(
        n_rows=4,
        n_cols=8,
        hidden_dim=64,
        latent_dim=16,
        elem_hidden_dim=8
    ).to(device)

    state = torch.load("aemcne_4x8_model.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()
    print("模型已加载: aemcne_4x8_model.pth")

    # 2. 路径配置（按你的 DCAE 代码风格）
    gan_folder = "../../HCP-IMSC-main/GAN/"
    output_folder = "../../HCP-IMSC-main/NMFdata&totalindex/output_AEMCNE/"
    os.makedirs(output_folder, exist_ok=True)

    # 3. 读取 H 数据（ground truth）
    H_files = sorted(glob.glob(os.path.join(gan_folder, "H*.csv")),
                     key=natural_sort_key)
    H_list = [pd.read_csv(fp, header=None).values.astype(np.float32)
              for fp in H_files]
    H_np = np.stack(H_list, axis=0)  # [N,4,8]
    H_tensor = torch.tensor(H_np, dtype=torch.float32).to(device)

    # 4. 读取 L 数据（原始，包含缺失位置的随机值）
    L_files = sorted(glob.glob(os.path.join(gan_folder, "L*.csv")),
                     key=natural_sort_key)
    L_list = [pd.read_csv(fp, header=None).values.astype(np.float32)
              for fp in L_files]
    L_np = np.stack(L_list, axis=0)  # [N,4,8]

    # 5. 读取 M 数据（掩码）
    M_files = sorted(glob.glob(os.path.join(gan_folder, "M*.csv")),
                     key=natural_sort_key)
    M_list = [pd.read_csv(fp, header=None).values.astype(np.float32)
              for fp in M_files]
    M_np = np.stack(M_list, axis=0)  # [N,4,8]

    # 一些形状检查
    N = L_np.shape[0]
    assert H_np.shape == (N, 4, 8), "H 的形状或数量与 L 不一致"
    assert M_np.shape == (N, 4, 8), "M 的形状或数量与 L 不一致"

    # 6. 构造模型输入: L_clean = L，缺失位置(M==1)置为 0
    L_clean = L_np.copy()
    L_clean[M_np == 1.0] = 0.0                 # 去掉缺失位置的随机值
    L_tensor = torch.tensor(L_clean, dtype=torch.float32).to(device)  # [N,4,8]

    # 7. 前向推理
    with torch.no_grad():
        pred_tensor = model(L_tensor)          # [N,4,8]
    pred_np = pred_tensor.cpu().numpy()        # 预测的完整矩阵

    # 8. 根据掩码 M 完成矩阵:
    #    缺失位置(M==1) 用预测值，观测位置(M==0) 保留原始 L
    completed_np = M_np * pred_np + (1.0 - M_np) * L_np  # [N,4,8]

    # 9. 计算误差（只在缺失位置上）
    H_np = H_tensor.cpu().numpy()  # [N,4,8]
    missing_mask = M_np            # 1=缺失

    total_missing = missing_mask.sum()
    if total_missing > 0:
        # 平均相对误差（缺失位置）
        err_gen = np.abs(missing_mask * H_np - missing_mask * completed_np) / (H_np + 1e-12)
        mean_rel_err_gen = err_gen.sum() / total_missing

        err_low = np.abs(missing_mask * H_np - missing_mask * L_np) / (H_np + 1e-12)
        mean_rel_err_low = err_low.sum() / total_missing

        print("生成的数据与高精度之间的平均相对误差(缺失部分):", mean_rel_err_gen)
        print("低精度与高精度之间的平均相对误差(缺失部分):", mean_rel_err_low)
    else:
        print("掩码中没有缺失位置(M==1)，不计算误差。")

    # 10. 获取输出文件编号（与原始 L<num>.csv 对齐）
    sorted_numbers = get_sorted_numbers_from_filenames(gan_folder)

    # 11. 逐样本保存补全后的 CSV
    for i in range(N):
        x_np = completed_np[i].copy()  # [4,8]

        # 可选：和你 DCAE 代码一样，删除末尾连续重复列（基于原始 L）
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
