import numpy as np
from scipy.optimize import nnls

# 场景：4个监测点 × 16个网格，3个信号
np.random.seed(0)
W = np.random.rand(4, 16) * 2 + 0.5

# 构造稀疏激活矩阵 H_true
H_true = np.zeros((16, 3))
positions = [2, 5, 12]
powers = [4.0, 2.0, 6.0]
for j, (pos, p) in enumerate(zip(positions, powers)):
    H_true[pos, j] = p

V = W.dot(H_true)               # 4×3 观测矩阵
alpha = V.max(axis=1)           # 每行最大值归一化因子
Wp = W / alpha[:, None]         # 归一化后的 W
Vp = V / alpha[:, None]         # 归一化后的 V

# 对每个信号（每列 Vp）做 NNLS，固定 Wp 求 H_recovered
H_recovered = np.zeros_like(H_true)
for j in range(3):
    h, _ = nnls(Wp, Vp[:, j])   # 非负最小二乘
    H_recovered[:, j] = h

# 验证结果
for j, pos in enumerate(positions):
    print(f"信号 {j+1}：真实位置 网格{pos}, 真功率 {powers[j]}")
    print(f"          恢复位置 最大索引 {H_recovered[:, j].argmax()}, "
          f"恢复功率 {H_recovered[pos, j]:.2f}\n")
