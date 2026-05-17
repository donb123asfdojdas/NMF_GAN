import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import completion_metrics, location_metrics, power_metrics
from src.utils.config import load_config
from src.utils.io import ensure_dir, read_csv, read_json, write_json


DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"
DEFAULT_DATA_DIR = ROOT / "data" / "processed"
DEFAULT_PREDICTION_DIR = ROOT / "outputs" / "nmf_final"
DEFAULT_SPLIT_DIR = ROOT / "data" / "splits"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "metrics"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate completion and downstream localization metrics.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--prediction-dir", default=str(DEFAULT_PREDICTION_DIR))
    parser.add_argument("--split-dir", default=str(DEFAULT_SPLIT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    config = load_config(args.config)
    data_dir = Path(args.data_dir)
    prediction_dir = Path(args.prediction_dir)
    output_dir = ensure_dir(args.output_dir)
    split_file = Path(args.split_dir) / f"{args.split}.txt"
    sample_ids = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    scene_side_length = float(config["dataset"].get("scene_side_length", 80.0))

    per_sample_rows = []

    print("开始评估")
    print(f"评估样本数: {len(sample_ids)}")
    print(f"预测目录: {prediction_dir.resolve()}")

    for sample_id in sample_ids:
        payload = np.load(data_dir / f"sample_{sample_id}.npz", allow_pickle=True)
        pred_completion = read_csv(prediction_dir / "completion" / f"output{sample_id}.csv").astype(np.float32)
        prediction = read_json(prediction_dir / "predictions" / f"prediction_{sample_id}.json")

        target = payload["H"].astype(np.float32)
        mask = payload["M"].astype(np.float32)
        valid_cols = payload["num_valid_cols"].item()
        valid_mask = payload["valid_col_mask"].astype(np.float32)[:, :valid_cols]

        completion = completion_metrics(pred_completion, target, mask, valid_mask)
        location = location_metrics(
            np.asarray(prediction["predicted_locations"]),
            payload["aligned_locations"],
            scene_side_length=scene_side_length,
        )
        power = power_metrics(np.asarray(prediction["predicted_powers"]), payload["aligned_powers"])
        row = {"sample_id": sample_id, **completion, **location, **power}
        per_sample_rows.append(row)

        true_locations = payload["aligned_locations"].tolist()
        pred_locations = prediction["predicted_locations"]
        true_powers = [round(float(x), 6) for x in payload["aligned_powers"].tolist()]
        pred_powers = [round(float(x), 6) for x in prediction["predicted_powers"]]

        print(f"场景 sample_id={sample_id}")
        print(f"  真实位置: {true_locations}")
        print(f"  预测位置: {pred_locations}")
        print(f"  真实功率: {true_powers}")
        print(f"  预测功率: {pred_powers}")
        print(
            "  指标: "
            f"completion_rmse={completion['completion_rmse']:.6f}, "
            f"location_error={location['location_error_normalized']:.6f}, "
            f"power_relative_mae={power['power_error_relative_mae']:.6f}"
        )

    def _mean_dict(keys, rows):
        if not rows:
            return {key: 0.0 for key in keys}
        return {key: float(np.mean([row[key] for row in per_sample_rows])) for key in keys}

    metrics = {}
    metrics.update(_mean_dict(["completion_mae", "completion_mse", "completion_rmse"], per_sample_rows))
    metrics.update(_mean_dict(["location_error_mean_distance", "location_error_normalized"], per_sample_rows))
    metrics.update(_mean_dict(["power_error_relative_mae", "power_error_mae"], per_sample_rows))
    metrics["num_samples"] = len(per_sample_rows)

    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_sample_rows[0].keys()) if per_sample_rows else ["sample_id"])
        writer.writeheader()
        writer.writerows(per_sample_rows)
    write_json(output_dir / "metrics.json", metrics)

    print("Evaluation complete.")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
