import csv
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from src.data.dataset import NMFCompletionDataset
from src.data.normalization import MinMaxNormalizer
from src.models.residual_refiner import TinyMaskedResidualRefiner
from src.training.losses import correlation_loss, masked_mse_loss, observed_consistency_loss
from src.utils.io import ensure_dir, write_json


def _build_model(model_cfg: Dict[str, object]):
    return TinyMaskedResidualRefiner(
        hidden_channels=int(model_cfg["hidden_channels"]),
        use_valid_mask_feature=bool(model_cfg.get("use_valid_mask_feature", True)),
    )


def _fit_normalizer(data_dir: Path, split_dir: Path) -> MinMaxNormalizer:
    sample_ids = (split_dir / "train.txt").read_text(encoding="utf-8").splitlines()
    values = []
    for sample_id in sample_ids:
        payload = np.load(data_dir / f"sample_{sample_id}.npz", allow_pickle=True)
        h = payload["H_padded"].astype(np.float32)
        x_obs = payload["X_obs_padded"].astype(np.float32)
        valid = payload["valid_col_mask"].astype(np.float32)
        values.append(h[valid > 0])
        values.append(x_obs[valid > 0])
    all_values = np.concatenate(values)
    return MinMaxNormalizer().fit(all_values)


def _completion_collate_fn(batch: list[Dict[str, object]]) -> Dict[str, object]:
    tensor_keys = {"H", "X_obs", "M", "valid_col_mask", "row_pos", "col_pos"}
    scalar_keys = {"sample_id", "num_valid_cols"}
    collated: Dict[str, object] = {}

    for key in batch[0].keys():
        values = [item[key] for item in batch]
        if key in tensor_keys:
            collated[key] = torch.stack(values, dim=0)
        elif key in scalar_keys:
            collated[key] = values
        else:
            collated[key] = values

    return collated


def train_refiner(
    config: Dict[str, object],
    data_dir: str | Path,
    split_dir: str | Path,
    checkpoint_dir: str | Path,
) -> Tuple[Path, Path]:
    from torch.utils.data import DataLoader

    data_dir = Path(data_dir)
    split_dir = Path(split_dir)
    checkpoint_dir = ensure_dir(checkpoint_dir)
    normalizer = _fit_normalizer(data_dir, split_dir)
    normalizer_path = checkpoint_dir / "normalization.json"
    normalizer.save(normalizer_path)

    train_dataset = NMFCompletionDataset(data_dir, "train", split_dir=split_dir, normalizer=normalizer)
    val_dataset = NMFCompletionDataset(data_dir, "val", split_dir=split_dir, normalizer=normalizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=0,
        collate_fn=_completion_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=_completion_collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and bool(config["training"].get("use_cuda", True)) else "cpu")
    model = _build_model(config["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    best_val = float("inf")
    best_checkpoint = checkpoint_dir / "best_residual.pt"
    log_path = checkpoint_dir / "train_log_residual.csv"
    total_epochs = int(config["training"]["epochs"])

    print("开始训练 TinyMaskedResidualRefiner")
    print(f"训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(val_dataset)}")
    print(f"Batch size: {int(config['training']['batch_size'])}")
    print(f"Epochs: {total_epochs}")
    print(f"Device: {device}")

    with log_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        for epoch in range(1, total_epochs + 1):
            model.train()
            train_losses = []
            for batch in train_loader:
                H = batch["H"].to(device).float()
                X_obs = batch["X_obs"].to(device).float()
                M = batch["M"].to(device).float()
                valid = batch["valid_col_mask"].to(device).float()
                row_pos = batch["row_pos"].to(device).float()
                col_pos = batch["col_pos"].to(device).float()

                optimizer.zero_grad()
                pred = model(X_obs, M, valid_col_mask=valid, row_pos=row_pos, col_pos=col_pos)
                loss = masked_mse_loss(pred, H, M, valid)
                lambda_corr = float(config["training"].get("lambda_corr", 0.0))
                lambda_consistency = float(config["training"].get("lambda_consistency", 0.0))
                if lambda_corr > 0:
                    loss = loss + lambda_corr * correlation_loss(pred, H, M, valid)
                if lambda_consistency > 0:
                    loss = loss + lambda_consistency * observed_consistency_loss(pred, X_obs, M, valid)

                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))

            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    H = batch["H"].to(device).float()
                    X_obs = batch["X_obs"].to(device).float()
                    M = batch["M"].to(device).float()
                    valid = batch["valid_col_mask"].to(device).float()
                    row_pos = batch["row_pos"].to(device).float()
                    col_pos = batch["col_pos"].to(device).float()
                    pred = model(X_obs, M, valid_col_mask=valid, row_pos=row_pos, col_pos=col_pos)
                    val_losses.append(float(masked_mse_loss(pred, H, M, valid).cpu()))

            train_loss = float(np.mean(train_losses)) if train_losses else 0.0
            val_loss = float(np.mean(val_losses)) if val_losses else 0.0
            writer.writerow({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            print(f"Epoch {epoch}/{total_epochs} - train_loss: {train_loss:.6f} - val_loss: {val_loss:.6f}")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "model_type": "residual",
                        "config": config,
                    },
                    best_checkpoint,
                )
                print(f"  更新最佳模型: {best_checkpoint} (val_loss={best_val:.6f})")

    write_json(checkpoint_dir / "train_summary_residual.json", {"best_val_loss": best_val, "normalization_path": str(normalizer_path)})
    return best_checkpoint, normalizer_path
