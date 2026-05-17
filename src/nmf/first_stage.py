from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from src.data.preprocessing import pad_sample_matrices
from src.nmf.utils import (
    add_noise,
    align_vector_with_indices,
    detect_active_components,
    distance_squared,
    load_basis_signals,
    nmf_fixed_basis,
)


@dataclass
class FirstStageConfig:
    signal_library_mat: str
    num_samples: int
    sensor_ids: Sequence[int]
    heights: Sequence[float]
    height_index: int
    snr_values: Sequence[float]
    snr_index: int
    min_sources: int
    max_sources: int
    location_range: Sequence[int]
    power_range: Sequence[int]
    nmf_rank: int
    nmf_iters: int
    detection_anchor_sensor: int
    detection_anchor_source: int
    seed: int
    scene_side_length: float = 80.0


def validate_processed_sample(sample_path: str | Path, atol: float = 1e-6) -> None:
    payload = np.load(sample_path, allow_pickle=True)
    total_index = payload["total_index"].astype(np.int64)
    x_obs = payload["X_obs"].astype(np.float32)
    mask = payload["M"].astype(np.float32)
    h_target = payload["H"].astype(np.float32)
    nmf_h = payload["NMF_H"].astype(np.float32)
    nmf_hh = payload["NMF_HH"].astype(np.float32)

    coarse_selected = nmf_h[:, total_index]
    full_selected = nmf_hh[:, total_index]

    observed = 1.0 - mask
    if not np.allclose(x_obs * observed, coarse_selected * observed, atol=atol):
        raise ValueError(f"X_obs 对齐自检失败: {sample_path}")
    if not np.allclose(h_target, full_selected, atol=atol):
        raise ValueError(f"H 对齐自检失败: {sample_path}")


def _random_unique_ints(rng: np.random.Generator, low: int, high: int, size: int) -> list[int]:
    return [int(x) for x in rng.integers(low, high + 1, size=size)]


def _sample_case(
    sample_index: int,
    basis_signals: np.ndarray,
    cfg: FirstStageConfig,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray] | None:
    num_sources = int(rng.integers(cfg.min_sources, cfg.max_sources + 1))
    source_positions = _random_unique_ints(rng, int(cfg.location_range[0]), int(cfg.location_range[1]), num_sources)
    source_powers = _random_unique_ints(rng, int(cfg.power_range[0]), int(cfg.power_range[1]), num_sources)
    signal_type_indices = sorted(rng.choice(np.arange(8), size=num_sources, replace=False).astype(int).tolist())

    height = float(cfg.heights[cfg.height_index])
    snr = float(cfg.snr_values[cfg.snr_index])
    threshold = 50.0 / distance_squared(cfg.detection_anchor_sensor, cfg.detection_anchor_source, height)

    noisy_h_rows: List[np.ndarray] = []
    full_h_rows: List[np.ndarray] = []
    noisy_detected_sets: List[List[int]] = []
    receive_counts: List[int] = []

    for sensor_id in cfg.sensor_ids:
        partial_signal = 0.0
        full_signal = 0.0
        visible_count = 0
        for basis_idx, source_pos, source_power in zip(signal_type_indices, source_positions, source_powers):
            component = np.asarray(basis_signals[basis_idx]).reshape((1024, 1))
            attenuation = source_power / distance_squared(sensor_id, source_pos, height)
            full_signal = full_signal + attenuation * component
            if np.sqrt(distance_squared(sensor_id, source_pos, height)) < 78:
                partial_signal = partial_signal + attenuation * component
                visible_count += 1
        if isinstance(partial_signal, float) or isinstance(full_signal, float):
            return None
        receive_counts.append(visible_count)
        noisy_signal = add_noise(np.asarray(partial_signal).reshape((1024, 1)), snr, rng)
        full_signal = add_noise(np.asarray(full_signal).reshape((1024, 1)), snr, rng)
        _, noisy_h = nmf_fixed_basis(noisy_signal, basis_signals, rank=cfg.nmf_rank, num_iters=cfg.nmf_iters, seed=cfg.seed)
        _, full_h = nmf_fixed_basis(full_signal, basis_signals, rank=cfg.nmf_rank, num_iters=cfg.nmf_iters, seed=cfg.seed)
        noisy_h = noisy_h.flatten().astype(np.float32)
        full_h = full_h.flatten().astype(np.float32)
        detected = detect_active_components(noisy_h, threshold)
        full_detected = detect_active_components(full_h, threshold)
        noisy_h_rows.append(noisy_h)
        full_h_rows.append(full_h)
        noisy_detected_sets.append(detected)
        if len(full_detected) != num_sources:
            return None

    if all(count == num_sources for count in receive_counts):
        return None
    if any(count == 0 for count in receive_counts):
        return None

    total_index = sorted(set().union(*[set(indices) for indices in noisy_detected_sets]))
    if len(total_index) != num_sources:
        return None

    full_matrix = np.vstack(full_h_rows)
    coarse_matrix = np.vstack(noisy_h_rows)
    target_matrix = full_matrix[:, total_index].astype(np.float32)

    incomplete_rows: List[np.ndarray] = []
    missing_rows: List[np.ndarray] = []
    for row_idx, detected in enumerate(noisy_detected_sets):
        aligned = align_vector_with_indices(coarse_matrix[row_idx], detected, total_index)
        missing = np.isnan(aligned).astype(np.float32)
        incomplete_rows.append(np.nan_to_num(aligned, nan=0.0).astype(np.float32))
        missing_rows.append(missing)

    incomplete_matrix = np.vstack(incomplete_rows)
    missing_matrix = np.vstack(missing_rows)
    observed_indices = [np.where(1.0 - missing_matrix[row] > 0)[0].astype(np.int32) for row in range(4)]
    missing_indices = [np.where(missing_matrix[row] > 0)[0].astype(np.int32) for row in range(4)]

    aligned_locations = np.asarray(
        [source_positions[signal_type_indices.index(idx)] for idx in total_index],
        dtype=np.int64,
    )
    aligned_powers = np.asarray(
        [source_powers[signal_type_indices.index(idx)] for idx in total_index],
        dtype=np.float32,
    )

    H_padded, M_padded, incomplete_padded, valid_col_mask = pad_sample_matrices(
        target_matrix,
        missing_matrix,
        l_matrix=incomplete_matrix,
    )

    return {
        "sample_id": np.asarray(str(sample_index)),
        "H": target_matrix,
        "M": missing_matrix,
        "X_obs": incomplete_matrix,
        "H_padded": H_padded,
        "M_padded": M_padded,
        "X_obs_padded": incomplete_padded,
        "valid_col_mask": valid_col_mask,
        "num_valid_cols": np.asarray(target_matrix.shape[1]),
        "total_index": np.asarray(total_index, dtype=np.int64),
        "aligned_locations": aligned_locations,
        "aligned_powers": aligned_powers,
        "source_positions": np.asarray(source_positions, dtype=np.int64),
        "source_powers": np.asarray(source_powers, dtype=np.float32),
        "signal_type_indices": np.asarray(signal_type_indices, dtype=np.int64),
        "NMF_H": coarse_matrix.astype(np.float32),
        "NMF_HH": full_matrix.astype(np.float32),
        "signal_indices_per_row": np.asarray(observed_indices, dtype=object),
        "missing_indices_per_row": np.asarray(missing_indices, dtype=object),
    }


def generate_processed_samples(
    config: Dict[str, object],
    processed_dir: str | Path,
) -> List[str]:
    cfg = FirstStageConfig(**config["dataset"])
    basis_signals = load_basis_signals(cfg.signal_library_mat)
    rng = np.random.default_rng(cfg.seed)
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)

    valid_ids: List[str] = []
    sample_index = 0
    attempts = 0
    max_attempts = cfg.num_samples * 20
    while len(valid_ids) < cfg.num_samples and attempts < max_attempts:
        payload = _sample_case(sample_index, basis_signals, cfg, rng)
        attempts += 1
        sample_index += 1
        if payload is None:
            continue
        sample_id = str(len(valid_ids))
        payload["sample_id"] = np.asarray(sample_id)
        np.savez(processed_path / f"sample_{sample_id}.npz", **payload)
        valid_ids.append(sample_id)

    for sample_id in valid_ids[: min(5, len(valid_ids))]:
        validate_processed_sample(processed_path / f"sample_{sample_id}.npz")
    return valid_ids
