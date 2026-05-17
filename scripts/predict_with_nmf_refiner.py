import argparse
import csv
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import NMFCompletionDataset
from src.data.normalization import MinMaxNormalizer
from src.evaluation.metrics import completion_metrics, location_metrics, power_metrics
from src.models.residual_refiner import TinyMaskedResidualRefiner
from src.nmf.second_stage import run_second_stage_localization
from src.utils.config import load_config
from src.utils.io import ensure_dir, write_csv, write_json
from src.utils.seed import set_seed


DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"
DEFAULT_CHECKPOINT = ROOT / "outputs" / "checkpoints" / "best_residual.pt"
DEFAULT_DATA_DIR = ROOT / "data" / "processed"
DEFAULT_SPLIT_DIR = ROOT / "data" / "splits"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "nmf_final"


def _clear_directory_files(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for file_path in directory.iterdir():
        if file_path.is_file():
            file_path.unlink()


def _write_comparison_csv(
    path: Path,
    target_matrix: np.ndarray,
    pred_matrix: np.ndarray,
    mask_matrix: np.ndarray,
) -> None:
    target_block = target_matrix.T
    pred_block = pred_matrix.T
    mask_block = mask_matrix.T
    blank = np.full((target_block.shape[0], 1), "", dtype=object)
    merged = np.concatenate(
        [
            target_block.astype(object),
            blank,
            pred_block.astype(object),
            blank.copy(),
            mask_block.astype(object),
        ],
        axis=1,
    )
    columns = [
        "true_r1", "true_r2", "true_r3", "true_r4",
        "",
        "pred_r1", "pred_r2", "pred_r3", "pred_r4",
        " ",
        "mask_r1", "mask_r2", "mask_r3", "mask_r4",
    ]
    pd.DataFrame(merged, columns=columns).to_csv(path, index=False, encoding="utf-8-sig")


def _load_model(checkpoint_path: Path):
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_cfg = checkpoint["config"]["model"]
    model = TinyMaskedResidualRefiner(hidden_channels=int(model_cfg["hidden_channels"]), use_valid_mask_feature=bool(model_cfg.get("use_valid_mask_feature", True)))
    model.load_state_dict(checkpoint["model_state"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight refiner and second-stage localization.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--split-dir", default=str(DEFAULT_SPLIT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    import torch

    config = load_config(args.config)
    set_seed(int(config["dataset"]["seed"]))
    output_dir = ensure_dir(args.output_dir)
    completion_dir = ensure_dir(output_dir / "completion")
    truth_dir = ensure_dir(output_dir / "ground_truth")
    prediction_dir = ensure_dir(output_dir / "predictions")
    comparison_dir = ensure_dir(output_dir / "comparison")

    _clear_directory_files(completion_dir)
    _clear_directory_files(truth_dir)
    _clear_directory_files(prediction_dir)
    _clear_directory_files(comparison_dir)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint 文件: {checkpoint_path}")
    model = _load_model(checkpoint_path)
    normalizer = MinMaxNormalizer.load(checkpoint_path.parent / "normalization.json")
    dataset = NMFCompletionDataset(
        data_dir=args.data_dir,
        split=args.split,
        split_dir=args.split_dir,
        normalizer=normalizer,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and bool(config["training"].get("use_cuda", True)) else "cpu")
    model = model.to(device).eval()
    rows = []
    summary = []
    all_location_errors = []
    all_power_errors = []
    scene_side_length = float(config["dataset"].get("scene_side_length", 80.0))

    print("开始测试与预测")
    print(f"Checkpoint: {checkpoint_path.resolve()}")
    print(f"测试样本数: {len(dataset)}")
    print(f"Device: {device}")

    with torch.no_grad():
        for item in dataset:
            sample_id = item["sample_id"]
            H = item["H"].unsqueeze(0).to(device).float()
            X_obs = item["X_obs"].unsqueeze(0).to(device).float()
            M = item["M"].unsqueeze(0).to(device).float()
            valid = item["valid_col_mask"].unsqueeze(0).to(device).float()
            row_pos = item["row_pos"].unsqueeze(0).to(device).float()
            col_pos = item["col_pos"].unsqueeze(0).to(device).float()

            pred = model(X_obs, M, valid_col_mask=valid, row_pos=row_pos, col_pos=col_pos)

            pred_np = normalizer.inverse_transform(pred.squeeze(0).squeeze(0).cpu().numpy())
            target_np = normalizer.inverse_transform(H.squeeze(0).squeeze(0).cpu().numpy())
            mask_np = M.squeeze(0).squeeze(0).cpu().numpy()
            valid_np = valid.squeeze(0).squeeze(0).cpu().numpy()

            num_valid_cols = item["num_valid_cols"]
            trimmed_pred = pred_np[:, :num_valid_cols]
            trimmed_target = target_np[:, :num_valid_cols]
            trimmed_mask = mask_np[:, :num_valid_cols]
            write_csv(completion_dir / f"output{sample_id}.csv", trimmed_pred)
            write_csv(truth_dir / f"H{sample_id}.csv", trimmed_target)
            _write_comparison_csv(
                comparison_dir / f"compare_{sample_id}.csv",
                trimmed_target,
                trimmed_pred,
                trimmed_mask,
            )

            second_stage = run_second_stage_localization(
                completed_columns=trimmed_pred,
                total_index=item["total_index"],
                sensor_ids=config["dataset"]["sensor_ids"],
                height=float(config["dataset"]["heights"][config["dataset"]["height_index"]]),
            )

            payload = {
                "sample_id": sample_id,
                "total_index": item["total_index"].tolist(),
                "true_locations": item["aligned_locations"].tolist(),
                "predicted_locations": second_stage["predicted_locations"].tolist(),
                "true_powers": item["aligned_powers"].tolist(),
                "predicted_powers": second_stage["predicted_powers"].tolist(),
            }
            write_json(prediction_dir / f"prediction_{sample_id}.json", payload)

            metrics = completion_metrics(pred_np, target_np, mask_np, valid_np)
            loc_metrics = location_metrics(
                np.asarray(second_stage["predicted_locations"]),
                np.asarray(item["aligned_locations"]),
                scene_side_length=scene_side_length,
            )
            pow_metrics = power_metrics(
                np.asarray(second_stage["predicted_powers"]),
                np.asarray(item["aligned_powers"]),
            )
            summary.append({"sample_id": sample_id, **metrics})
            all_location_errors.append(loc_metrics["location_error_normalized"])
            all_power_errors.append(pow_metrics["power_error_relative_mae"])
            rows.append(
                {
                    "sample_id": sample_id,
                    "completion_csv": str((completion_dir / f"output{sample_id}.csv").resolve()),
                    "ground_truth_csv": str((truth_dir / f"H{sample_id}.csv").resolve()),
                    "comparison_csv": str((comparison_dir / f"compare_{sample_id}.csv").resolve()),
                    "prediction_json": str((prediction_dir / f"prediction_{sample_id}.json").resolve()),
                }
            )

            true_locations = item["aligned_locations"].tolist()
            pred_locations = second_stage["predicted_locations"].tolist()
            true_powers = [round(float(x), 6) for x in item["aligned_powers"].tolist()]
            pred_powers = [round(float(x), 6) for x in second_stage["predicted_powers"].tolist()]

            print(f"场景 sample_id={sample_id}")
            print(f"  真实位置: {true_locations}")
            print(f"  预测位置: {pred_locations}")
            print(f"  真实功率: {true_powers}")
            print(f"  预测功率: {pred_powers}")
            print(
                "  补全误差: "
                f"MAE={metrics['completion_mae']:.6f}, "
                f"MSE={metrics['completion_mse']:.6f}, "
                f"RMSE={metrics['completion_rmse']:.6f}"
            )
            print(
                "  场景误差: "
                f"相对功率误差={pow_metrics['power_error_relative_mae']:.6f}, "
                f"距离误差={loc_metrics['location_error_normalized']:.6f}"
            )

    with (output_dir / "artifacts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "completion_csv", "ground_truth_csv", "comparison_csv", "prediction_json"])
        writer.writeheader()
        writer.writerows(rows)
    write_json(output_dir / "completion_summary.json", summary)
    final_power_error = float(sum(all_power_errors) / len(all_power_errors)) if all_power_errors else 0.0
    final_location_error = float(sum(all_location_errors) / len(all_location_errors)) if all_location_errors else 0.0
    print(f"Saved refined completion outputs to {completion_dir.resolve()}")
    print(f"Saved ground-truth H outputs to {truth_dir.resolve()}")
    print(f"Saved comparison csv outputs to {comparison_dir.resolve()}")
    print(f"Saved second-stage predictions to {prediction_dir.resolve()}")
    print(f"最终相对功率误差: {final_power_error:.6f}")
    print(f"最终距离误差: {final_location_error:.6f}")


if __name__ == "__main__":
    main()
