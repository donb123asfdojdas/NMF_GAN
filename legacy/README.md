旧项目源码保留在仓库根目录的 `HCP-IMSC-main/` 与 `GAN-MMC-main/`。

新的入口脚本已经迁移到：

- `scripts/generate_dataset.py`
- `scripts/train_refiner.py`
- `scripts/predict_with_nmf_refiner.py`
- `scripts/evaluate.py`

旧脚本与新入口的大致对应关系：

- `HCP-IMSC-main/数据处理部分/套系统随机snr1.py`
  对应 `scripts/generate_dataset.py`
- `GAN-MMC-main/GAN-MMC.py`
  被 `scripts/train_refiner.py` 与 `scripts/predict_with_nmf_refiner.py` 替代
- `HCP-IMSC-main/数据处理部分/获得最终结果.py`
  对应 `scripts/predict_with_nmf_refiner.py` 和 `scripts/evaluate.py`

当前没有删除旧代码，目的是方便对照原始流程和数值行为。
