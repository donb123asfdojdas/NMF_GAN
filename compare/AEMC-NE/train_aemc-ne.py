import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================== 配置区域 ===========================

# 数据所在目录：里面包含 L0.csv / M0.csv / H0.csv 等
ROOT_DIR   = "../../HCP-IMSC-main/GAN_for_train/"

# 训练相关超参数
BATCH_SIZE = 16
NUM_EPOCHS = 50
LR         = 1e-3
WEIGHT_DECAY = 1e-5

# 模型结构超参数
N_ROWS         = 4
N_COLS         = 8
HIDDEN_DIM     = 64
LATENT_DIM     = 16
ELEM_HIDDEN_DIM = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========================= 模型定义区域 =========================

class ElementWiseNN(nn.Module):
    """
    逐元素小网络（所有元素共享同一组参数）:
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

        # 如果你的 H 的数值范围比较大，可以不加 sigmoid；
        # 如果 H 在 [0,1] 或类似范围，可以考虑加 sigmoid 约束输出。
        # recon = torch.sigmoid(recon)

        return recon


# ========================= 数据集定义区域 =========================

class TripletCSV4x8Dataset(Dataset):
    """
    每个样本由三个文件组成:
        L<num>.csv : 带缺失的矩阵 (4x8)，缺失位置为随机数
        M<num>.csv : 掩码矩阵 (4x8)，1=缺失位置, 0=真实值位置
        H<num>.csv : 真实完整矩阵 (4x8)，作为 ground truth

    训练目标:
        - 输入: L_input (其中 M==1 的地方已被置为 0 或占位)
        - 标签: H
        - loss: 只在 M==1 的缺失位置计算 MSE
    """
    def __init__(self, root_dir: str, index_list=None):
        """
        root_dir: csv 所在目录
        index_list: 例如 [0,1,2,...]，对应 L0/M0/H0 等；
                    若为 None，则会自动扫描 root_dir 中的 L*.csv
        """
        super().__init__()
        self.root_dir = root_dir

        if index_list is not None:
            self.ids = index_list
        else:
            pattern = os.path.join(root_dir, "L*.csv")
            files = sorted(glob.glob(pattern))
            ids = []
            for f in files:
                base = os.path.basename(f)         # L0.csv
                num_str = base[1:base.rfind('.')]  # 从第 1 位到 '.'
                if num_str.isdigit():
                    ids.append(int(num_str))
            self.ids = sorted(ids)

        if len(self.ids) == 0:
            raise ValueError(f"在目录 {root_dir} 中未找到任何 L<num>.csv 文件。")

    def __len__(self) -> int:
        return len(self.ids)

    def _load_csv(self, prefix: str, num: int) -> np.ndarray:
        """
        prefix: 'L' / 'M' / 'H'
        num   : 整数，如 0,1,2...
        """
        path = os.path.join(self.root_dir, f"{prefix}{num}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到文件: {path}")
        arr = np.loadtxt(path, delimiter=",")
        return arr

    def __getitem__(self, idx: int):
        num = self.ids[idx]

        # 读取 L/M/H
        L = self._load_csv("L", num)  # (4,8)
        M = self._load_csv("M", num)  # (4,8)
        H = self._load_csv("H", num)  # (4,8)

        # 转为 float32
        L = L.astype(np.float32)
        M = M.astype(np.float32)
        H = H.astype(np.float32)

        # 构造模型输入：
        #   对缺失位置 (M==1) 把 L 中的随机数去掉，置为 0（或者你也可以换成全局均值等）
        L_input = L.copy()
        L_input[M == 1.0] = 0.0

        # 转为 tensor
        L_input_t = torch.from_numpy(L_input)  # (4,8)
        H_t       = torch.from_numpy(H)        # (4,8)
        M_missing = torch.from_numpy(M)        # (4,8) 1=缺失位置

        # 返回: 模型输入, 标签, 缺失 mask
        return L_input_t, H_t, M_missing


# ========================= 损失函数定义 =========================

def masked_mse_on_missing(pred: torch.Tensor,
                          target: torch.Tensor,
                          missing_mask: torch.Tensor,
                          eps: float = 1e-8) -> torch.Tensor:
    """
    只在缺失位置上计算 MSE。

    pred        : (B, 4, 8) 模型输出
    target      : (B, 4, 8) H (ground truth)
    missing_mask: (B, 4, 8) 1=缺失位置, 0=真实位置

    返回: 一个标量 loss
    """
    diff = (pred - target) * missing_mask
    mse = (diff ** 2).sum() / (missing_mask.sum() + eps)
    return mse


# ========================= 推理函数（可选） =========================

def infer_one_num(model: nn.Module,
                  root_dir: str,
                  num: int,
                  save_completed: bool = True):
    """
    使用训练好的模型对单个样本 (L<num>, M<num>) 做补全。

    - 读取 L<num>.csv 和 M<num>.csv
    - 将缺失位置 (M==1) 的随机值置为 0，作为输入
    - 模型预测完整矩阵 pred
    - 用 pred 填回缺失位置，得到 completed
    - 可选: 将 completed 保存为 C<num>.csv

    返回:
        L: 原始 L 矩阵 (4,8)
        M: 掩码矩阵 (4,8)
        completed: 缺失位置由模型预测填充后的矩阵 (4,8)
        pred: 模型的完整预测矩阵 (4,8)
    """
    model.eval()

    L_path = os.path.join(root_dir, f"L{num}.csv")
    M_path = os.path.join(root_dir, f"M{num}.csv")

    L = np.loadtxt(L_path, delimiter=",").astype(np.float32)
    M = np.loadtxt(M_path, delimiter=",").astype(np.float32)

    # 构造输入
    L_input = L.copy()
    L_input[M == 1.0] = 0.0

    L_input_t = torch.from_numpy(L_input).unsqueeze(0).to(DEVICE)  # (1,4,8)

    with torch.no_grad():
        pred = model(L_input_t)[0].cpu().numpy()  # (4,8)

    completed = L.copy()
    completed[M == 1.0] = pred[M == 1.0]

    if save_completed:
        out_path = os.path.join(root_dir, f"C{num}.csv")
        np.savetxt(out_path, completed, delimiter=",")
        print(f"[Infer] 补全矩阵已保存到: {out_path}")

    return L, M, completed, pred


# ========================== 训练主程序 ==========================

def main():
    print(f"使用设备: {DEVICE}")

    # 1) 构建 Dataset 和 DataLoader
    dataset = TripletCSV4x8Dataset(root_dir=ROOT_DIR, index_list=None)
    train_loader = DataLoader(dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              drop_last=False)

    print(f"数据集大小: {len(dataset)} 个样本")

    # 2) 初始化模型和优化器
    model = AEMCNE4x8(
        n_rows=N_ROWS,
        n_cols=N_COLS,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        elem_hidden_dim=ELEM_HIDDEN_DIM
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    # 3) 训练循环
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_count = 0

        for L_in, H_gt, M_missing in train_loader:
            L_in      = L_in.to(DEVICE)        # (B,4,8)
            H_gt      = H_gt.to(DEVICE)        # (B,4,8)
            M_missing = M_missing.to(DEVICE)   # (B,4,8)

            optimizer.zero_grad()
            pred = model(L_in)                 # (B,4,8)

            loss = masked_mse_on_missing(pred, H_gt, M_missing)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * L_in.size(0)
            total_count += L_in.size(0)

        avg_loss = total_loss / max(total_count, 1)
        print(f"[Epoch {epoch:03d}/{NUM_EPOCHS}] "
              f"Train masked MSE (on missing): {avg_loss:.9f}")

    # 4) 训练结束后，简单示范对第 0 号样本做一次补全
    try:
        print("\n开始对样本 num=0 做一次推理示例...")
        L0, M0, C0, P0 = infer_one_num(model, ROOT_DIR, num=0, save_completed=True)
        print("原始 L0:")
        print(L0)
        print("掩码 M0 (1=缺失位置):")
        print(M0)
        print("补全后的矩阵 C0:")
        print(C0)
    except Exception as e:
        print(f"推理示例失败: {e}")

    # 如需保存模型参数:
    save_path = "aemcne_4x8_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"模型参数已保存到: {save_path}")


if __name__ == "__main__":
    main()
