# Lightweight NMF Refiner

这个版本已经简化为纯 Python 主流程，不再依赖外部粗补全 `L`。

当前流程是：

1. `generate_dataset.py`
   生成模拟信号、第一层 NMF、缺失观测矩阵 `X_obs`、缺失掩码 `M`、真值 `H`、以及第二层求解需要的中间结果。
2. `train_refiner.py`
   直接用 `X_obs + M -> H` 训练轻量补全模型 `TinyMaskedResidualRefiner`。
3. `predict_with_nmf_refiner.py`
   直接从缺失输入预测补全结果 `output*.csv`，再替换回 `NMF_H`，执行第二层 NMF / 求解，输出最终位置和功率。
4. `evaluate.py`
   统一评估补全误差、位置误差、功率误差。

## 目录里保留的核心代码

- `src/nmf/`
  第一层 NMF、第二层求解、定位相关逻辑
- `src/models/residual_refiner.py`
  默认轻量残差补全模型
- `src/data/`
  padding、normalization、dataset
- `src/training/`
  模型训练逻辑
- `src/evaluation/`
  评估逻辑
- `compare/`
  原有对比方法，保持不动

## 运行顺序

### 1. 生成数据集

```bash
python scripts/generate_dataset.py \
  --config configs/default.yaml \
  --num-samples 1000 \
  --output-dir data/processed
```

输出包括：

- `data/processed/sample_*.npz`
- `data/splits/train.txt`
- `data/splits/val.txt`
- `data/splits/test.txt`

### 2. 训练补全模型

```bash
python scripts/train_refiner.py \
  --config configs/default.yaml \
  --data-dir data/processed \
  --checkpoint-dir outputs/checkpoints
```

输出包括：

- `outputs/checkpoints/best_residual.pt`
- `outputs/checkpoints/normalization.json`
- `outputs/checkpoints/train_log_residual.csv`

### 3. 预测补全结果并执行第二层 NMF

```bash
python scripts/predict_with_nmf_refiner.py \
  --config configs/default.yaml \
  --checkpoint outputs/checkpoints/best_residual.pt \
  --data-dir data/processed \
  --output-dir outputs/nmf_final
```

输出包括：

- `outputs/nmf_final/completion/output*.csv`
- `outputs/nmf_final/predictions/prediction_*.json`

### 4. 统一评估

```bash
python scripts/evaluate.py \
  --config configs/default.yaml \
  --data-dir data/processed \
  --prediction-dir outputs/nmf_final \
  --output-dir outputs/metrics
```

输出包括：

- `outputs/metrics/metrics.csv`
- `outputs/metrics/metrics.json`

## 现在不再需要的步骤

- 不再需要先生成核范数最小化/HCP-IMSC 的粗补全 `L`
- 不再需要 MATLAB 入口
- 不再需要先粗补全、再精修的双补全流程

## compare

`compare/` 目录仍然保留，用于和现有方法做横向比较。
