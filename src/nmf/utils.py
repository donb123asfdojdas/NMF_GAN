import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from scipy.io import loadmat


def pos_from_grid_id(grid_id: int) -> tuple[float, float]:
    toty = grid_id // 10
    totx = grid_id % 10
    if grid_id % 10 == 0:
        toty -= 1
        totx = 10
    y = 8 * toty
    x = 8 * (totx - 1)
    return float(x), float(y)


def distance_squared(sensor_id: int, source_id: int, height: float) -> float:
    sensor_pos = pos_from_grid_id(sensor_id)
    source_pos = pos_from_grid_id(source_id)
    return (sensor_pos[0] - source_pos[0]) ** 2 + (sensor_pos[1] - source_pos[1]) ** 2 + height ** 2


def horizontal_distance(source_a: int, source_b: int) -> float:
    pos_a = pos_from_grid_id(source_a)
    pos_b = pos_from_grid_id(source_b)
    return math.sqrt((pos_a[0] - pos_b[0]) ** 2 + (pos_a[1] - pos_b[1]) ** 2)


def add_noise(signal: np.ndarray, snr: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.standard_normal(signal.shape[0]).reshape((-1, 1))
    noise_power = np.sum(signal) / (10 ** (snr / 10.0)) * (noise**2) / np.linalg.norm(noise) ** 2
    return signal + noise_power


def load_basis_signals(mat_path: str | Path) -> np.ndarray:
    mat = loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    return np.asarray(mat["fea"], dtype=np.float32)


def nmf_fixed_basis(X: np.ndarray, basis_signals: np.ndarray, rank: int = 8, num_iters: int = 200, seed: int = 15) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    W = basis_signals.T.astype(np.float64)
    H = rng.random((rank, X.shape[1]), dtype=np.float64)
    for _ in range(num_iters):
        numerator = W.T @ X
        denominator = (W.T @ W @ H) + 1e-12
        H *= numerator / denominator
    return W, H


def detect_active_components(h_column: np.ndarray, threshold: float) -> List[int]:
    return [idx for idx in range(len(h_column)) if h_column[idx] > threshold]


def align_vector_with_indices(values: Sequence[float], present_indices: Sequence[int], total_indices: Sequence[int]) -> np.ndarray:
    aligned = np.full((len(total_indices),), np.nan, dtype=np.float32)
    present_map = {int(idx): float(values[int(idx)]) for idx in present_indices}
    for pos, idx in enumerate(total_indices):
        if int(idx) in present_map:
            aligned[pos] = present_map[int(idx)]
    return aligned


def build_second_stage_matrix(sensor_ids: Sequence[int], height: float) -> np.ndarray:
    columns: List[np.ndarray] = []
    for sensor_id in sensor_ids:
        values = [1.0 / distance_squared(sensor_id, source_id, height) for source_id in range(1, 101)]
        columns.append(np.asarray(values, dtype=np.float64).reshape((100, 1)))
    return np.hstack(columns)
