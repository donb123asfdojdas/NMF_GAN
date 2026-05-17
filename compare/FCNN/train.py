# train_test_dnn_nsr_hcp.py
import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


# ----------------------- 模型定义 -----------------------
class DNNNSR(nn.Module):
    """
    参考:
    S. Faramarzi et al.,
    'Matrix Completion via Nonsmooth Regularization of Fully Connected Neural Networks',
    arXiv:2403.10232, 2024.

    输入: concat(L_clean.flatten(), M.flatten()) -> [64]
    输出: 预测完整矩阵 flatten -> [32]
    """

    def __init__(self, input_dim=64, hidden=(128, 64, 32), output_dim=32):
        super().__init__()
        dims = [input_dim] + list(hidden) + [output_dim]
        layers = []

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())  # 中间层 ReLU

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        hidden_acts = []
        out = x
        for layer in self.net:
            out = layer(out)
            if isinstance(layer, nn.ReLU):
                hidden_acts.append(out)
        return out, hidden_acts

    def nuclear_norm(self):
        total = 0.0
        for m in self.modules():
            if isinstance(m, nn.Linear):
                s = torch.linalg.svdvals(m.weight)
                total += s.sum()
        return total


# ----------------------- 数据集定义 -----------------------
class LMHDataset(Dataset):
    """
    root 下存在文件:
      L0.csv, M0.csv, H0.csv
      L1.csv, M1.csv, H1.csv
      ...
    每个 csv 是 4x8 矩阵

    输出:
      x: [64] = concat(L_clean.flatten(), M.flatten())
      y: [32] = H.flatten()
      M_flat: [32] = 掩码 (1=缺失, 0=观测)
    """

    def __init__(self, root):
        self.root = root
        L_files = sorted(glob.glob(os.path.join(root, "L*.csv")))
        self.items = []

        for lf in L_files:
            idx = os.path.basename(lf)[1:-4]  # L12.csv -> "12"
            mf = os.path.join(root, f"M{idx}.csv")
            hf = os.path.join(root, f"H{idx}.csv")
            if os.path.exists(mf) and os.path.exists(hf):
                self.items.append((lf, mf, hf))

        if len(self.items) == 0:
            raise RuntimeError(f"在 {root} 下没有找到 L/M/H*.csv")

        print(f"✔ 读取到 {len(self.items)} 个样本")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        l_path, m_path, h_path = self.items[idx]

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


# ----------------------- 训练与测试 -----------------------
def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    lambda_l1,
    lambda_nuc,
):
    model.train()
    total_loss = 0.0
    total_main = 0.0
    total_l1 = 0.0
    total_nuc = 0.0
    n_batches = 0

    for x, y, M in loader:
        x = x.to(device)
        y = y.to(device)
        M = M.to(device)  # 1=缺失

        optimizer.zero_grad()
        pred, hidden = model(x)  # pred: [B,32]

        # 只在缺失位置计算 MSE
        mask = M
        num_missing = mask.sum()
        if num_missing.item() > 0:
            main_loss = ((pred - y) ** 2 * mask).sum() / num_missing
        else:
            main_loss = ((pred - y) ** 2).mean()

        # L1 正则: 中间层激活
        if len(hidden) > 0:
            l1_reg = sum(h.abs().mean() for h in hidden)
        else:
            l1_reg = torch.tensor(0.0, device=device)

        # 核范数正则
        nuc_reg = model.nuclear_norm()

        loss = main_loss + lambda_l1 * l1_reg + lambda_nuc * nuc_reg
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_main += main_loss.item()
        total_l1 += l1_reg.item()
        total_nuc += nuc_reg.item()
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "main": total_main / n_batches,
        "l1": total_l1 / n_batches,
        "nuc": total_nuc / n_batches,
    }


@torch.no_grad()
def evaluate_missing_mse(model, loader, device):
    """
    在测试集上评估:
    只计算缺失位置 (M==1) 上的 MSE
    """
    model.eval()
    total_se = 0.0   # squared error
    total_count = 0.0

    for x, y, M in loader:
        x = x.to(device)
        y = y.to(device)
        M = M.to(device)

        pred, _ = model(x)

        mask = M  # 1=缺失
        se = ((pred - y) ** 2) * mask
        total_se += se.sum().item()
        total_count += mask.sum().item()

    if total_count == 0:
        print("测试集中没有缺失位置(M==1)，返回 0。")
        return 0.0

    mse_missing = total_se / total_count
    return mse_missing


def main():
    data_root = "../../HCP-IMSC-main/GAN/"
    device = torch.device("cpu")  # 有 GPU 可以改为 torch.device("cuda")

    # 超参数
    epochs = 200
    batch_size = 16
    lr = 1e-3
    lambda_l1_max = 1e-3
    lambda_nuc_max = 1e-4
    train_ratio = 0.8  # 8:2 划分训练/测试

    # 数据集
    full_dataset = LMHDataset(data_root)
    n_total = len(full_dataset)
    n_train = int(n_total * train_ratio)
    n_test = n_total - n_train

    # 随机划分训练/测试
    train_set, test_set = random_split(
        full_dataset,
        [n_train, n_test],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"✔ 训练集: {n_train} 样本, 测试集: {n_test} 样本")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # 模型 & 优化器
    model = DNNNSR().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        alpha = epoch / epochs
        lambda_l1 = lambda_l1_max * alpha
        lambda_nuc = lambda_nuc_max * alpha

        stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            lambda_l1=lambda_l1,
            lambda_nuc=lambda_nuc,
        )

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train_loss={stats['loss']:.6f} | "
            f"main={stats['main']:.6f} | "
            f"L1={stats['l1']:.6f} | "
            f"nuc={stats['nuc']:.6f}"
        )

    # ------- 训练结束后，测试集评估(只在缺失位置上的 MSE) -------
    test_mse_missing = evaluate_missing_mse(model, test_loader, device)
    print(f"\n=== 测试集缺失部分 MSE: {test_mse_missing:.6f} ===")

    # 保存模型
    torch.save(model.state_dict(), "dnn_nsr_4x8.pth")
    print("✔ 已保存模型到 dnn_nsr_4x8.pth")


if __name__ == "__main__":
    main()
