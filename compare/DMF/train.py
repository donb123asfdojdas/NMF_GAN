# -*- coding: utf-8 -*-
"""
Deep Completion Autoencoder (DCAE) for Radio Map Completion

- 输入: ../../HCP-IMSC-main/GAN 下的 L*, M*, H* CSV 文件
- 任务: 基于掩码矩阵补全缺失部分
- 输出: 补全后的矩阵，存储到指定文件夹
"""

import os
import glob
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split


# --------------------- DCAE 模型定义 ---------------------
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


# --------------------- 数据集定义 ---------------------
class LMHDataset(Dataset):
    """
    读取 L, M, H 文件，L 为带缺失值的矩阵，M 为掩码，H 为真实矩阵
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.items = []

        # 获取 L, M, H 文件
        L_files = sorted(glob.glob(os.path.join(root_dir, "L*.csv")))
        for lf in L_files:
            idx = os.path.basename(lf)[1:-4]  # L0.csv -> "0"
            mf = os.path.join(root_dir, f"M{idx}.csv")
            hf = os.path.join(root_dir, f"H{idx}.csv")
            if os.path.exists(mf) and os.path.exists(hf):
                self.items.append((lf, mf, hf))

        if len(self.items) == 0:
            raise RuntimeError(f"在 {root_dir} 下没有找到 L/M/H*.csv 文件")

        print(f"✔ 读取到 {len(self.items)} 个样本")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        l_path, m_path, h_path = self.items[idx]

        # 加载数据
        L = np.loadtxt(l_path, delimiter=",").astype(np.float32)  # [4,8]
        M = np.loadtxt(m_path, delimiter=",").astype(np.float32)
        H = np.loadtxt(h_path, delimiter=",").astype(np.float32)

        # 去除缺失位置的随机值: L_clean = L * (1 - M)
        L_clean = L * (1.0 - M)

        L_flat = L_clean.reshape(-1)  # [32]
        M_flat = M.reshape(-1)        # [32]
        H_flat = H.reshape(-1)        # [32]

        x = np.concatenate([L_flat, M_flat], axis=0)  # [64]
        y = H_flat

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(M_flat, dtype=torch.float32),
        )


# --------------------- 训练与评估 ---------------------
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    loss_fn = nn.MSELoss()

    for x, y, M in loader:
        x = x.to(device)
        y = y.to(device)
        M = M.to(device)

        optimizer.zero_grad()
        pred, _ = model(x)  # [B, 32]

        # 只在缺失位置计算 MSE
        mask = M
        num_missing = mask.sum()
        if num_missing.item() > 0:
            main_loss = ((pred - y) ** 2 * mask).sum() / num_missing
        else:
            main_loss = ((pred - y) ** 2).mean()

        loss = main_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    loss_fn = nn.MSELoss()

    for x, y, M in loader:
        x = x.to(device)
        y = y.to(device)
        M = M.to(device)

        pred, _ = model(x)

        mask = M
        num_missing = mask.sum()
        if num_missing.item() > 0:
            main_loss = ((pred - y) ** 2 * mask).sum() / num_missing
        else:
            main_loss = ((pred - y) ** 2).mean()

        total_loss += main_loss.item()
        n_batches += 1

    return total_loss / n_batches


# --------------------- 主函数 ---------------------
def main():
    data_root = "../../HCP-IMSC-main/GAN_for_train/"
    output_folder = "../../HCP-IMSC-main/NMFdata&totalindex/output_DMF/"
    os.makedirs(output_folder, exist_ok=True)

    # 训练配置
    batch_size = 16
    epochs = 50
    lr = 1e-3
    train_ratio = 0.8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    # 1. 数据集
    dataset = LMHDataset(data_root)
    n_total = len(dataset)
    n_train = int(n_total * train_ratio)
    n_val = n_total - n_train

    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"✔ 训练集: {n_train} 样本, 验证集: {n_val} 样本")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # 2. 模型 & 优化器
    model = DCAE(input_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 3. 训练
    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)

        print(f"Epoch {epoch:03d}/{epochs} | Train Loss: {train_loss:.8f} | Val Loss: {val_loss:.8f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "dcae_model.pth")
            print(f"✔ 保存最佳模型: dcae_model.pth")

    print("训练完成，最佳验证损失:", best_val_loss)


if __name__ == "__main__":
    main()
