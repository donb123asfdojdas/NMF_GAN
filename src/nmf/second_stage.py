from typing import Dict, Sequence

import numpy as np

from src.nmf.utils import build_second_stage_matrix


def get_nmf_objective(X: np.ndarray, W: np.ndarray, H: np.ndarray) -> float:
    return float(np.linalg.norm(X - H @ W, ord="fro") ** 2)


def get_regularized_objective(X: np.ndarray, W: np.ndarray, H: np.ndarray, rho: float) -> float:
    tmp = 0.0
    for row in range(H.shape[0]):
        tmp += np.linalg.norm(H[row, :], ord=1) ** 2 - np.linalg.norm(H[row, :]) ** 2
    return float(np.linalg.norm(X - H @ W, ord="fro") ** 2 + rho * tmp)


def update_prim_var_by_palm(X: np.ndarray, W_init: np.ndarray, H_init: np.ndarray, max_iter: int, rho: float) -> tuple[np.ndarray, np.ndarray]:
    H_prev = H_init.copy()
    W_cur = W_init.copy()
    all_ones = np.ones((H_init.shape[1], H_init.shape[1]))
    for _ in range(max_iter):
        numerator = X @ W_cur.T
        denominator = H_prev @ (W_cur @ W_cur.T) + rho * (H_prev @ all_ones - H_prev) + 1e-12
        H_prev = np.multiply(H_prev, numerator / denominator) + 1e-80
    return W_cur, H_prev


def solve_second_stage(X: np.ndarray, W: np.ndarray, H: np.ndarray, outer_iters: int = 61) -> tuple[np.ndarray, np.ndarray]:
    rho = 1e-10
    gamma = 1.6
    for _ in range(outer_iters):
        W, H = update_prim_var_by_palm(X, W, H, max_iter=200, rho=rho)
        _ = get_nmf_objective(X, W, H)
        _ = get_regularized_objective(X, W, H, rho)
        rho = min(rho * gamma, 1e20)
    return W, H


def run_second_stage_localization(
    completed_columns: np.ndarray,
    total_index: Sequence[int],
    sensor_ids: Sequence[int],
    height: float,
) -> Dict[str, np.ndarray]:
    W = build_second_stage_matrix(sensor_ids, height)
    num_signals = completed_columns.shape[1]
    H_init = np.ones((num_signals, 100), dtype=np.float64)
    _, q = solve_second_stage(completed_columns.T.astype(np.float64), W, H_init)

    predicted_locations = []
    predicted_powers = []
    for signal_idx in range(num_signals):
        best_idx = int(np.argmax(q[signal_idx, :]))
        predicted_locations.append(best_idx + 1)
        predicted_powers.append(float(q[signal_idx, best_idx]))

    return {
        "completed_trimmed_nmf_h": completed_columns.astype(np.float32),
        "total_index": np.asarray(total_index, dtype=np.int64),
        "predicted_locations": np.asarray(predicted_locations, dtype=np.int64),
        "predicted_powers": np.asarray(predicted_powers, dtype=np.float32),
    }
