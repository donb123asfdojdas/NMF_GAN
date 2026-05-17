from typing import Dict

import numpy as np

from src.nmf.utils import horizontal_distance


def completion_metrics(pred: np.ndarray, target: np.ndarray, mask: np.ndarray, valid_col_mask: np.ndarray) -> Dict[str, float]:
    eff_mask = mask * valid_col_mask
    denom = max(float(eff_mask.sum()), 1.0)
    errors = (pred - target) * eff_mask
    mae = float(np.abs(errors).sum() / denom)
    mse = float((errors**2).sum() / denom)
    rmse = float(np.sqrt(mse))
    return {"completion_mae": mae, "completion_mse": mse, "completion_rmse": rmse}


def location_metrics(pred_locations: np.ndarray, true_locations: np.ndarray, scene_side_length: float = 80.0) -> Dict[str, float]:
    distances = [horizontal_distance(int(t), int(p)) for t, p in zip(true_locations, pred_locations)]
    mean_distance = float(np.mean(distances)) if distances else 0.0
    normalized_distance = float(mean_distance / scene_side_length) if distances else 0.0
    return {
        "location_error_mean_distance": mean_distance,
        "location_error_normalized": normalized_distance,
    }


def power_metrics(pred_powers: np.ndarray, true_powers: np.ndarray) -> Dict[str, float]:
    relative = np.abs(pred_powers - true_powers) / np.clip(np.abs(true_powers), 1e-8, None)
    absolute = np.abs(pred_powers - true_powers)
    return {
        "power_error_relative_mae": float(np.mean(relative)) if len(relative) else 0.0,
        "power_error_mae": float(np.mean(absolute)) if len(absolute) else 0.0,
    }
