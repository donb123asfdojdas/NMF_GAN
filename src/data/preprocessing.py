from typing import Tuple

import numpy as np


TARGET_SHAPE = (4, 8)


def pad_columns(matrix: np.ndarray, target_cols: int = 8, mode: str = "edge") -> np.ndarray:
    rows, cols = matrix.shape
    if cols > target_cols:
        raise ValueError(f"Expected <= {target_cols} columns, got {cols}.")
    if cols == target_cols:
        return matrix.astype(np.float32, copy=True)

    padded = np.zeros((rows, target_cols), dtype=np.float32)
    padded[:, :cols] = matrix
    if cols == 0:
        return padded
    if mode == "edge":
        padded[:, cols:] = matrix[:, cols - 1 : cols]
    elif mode == "zero":
        pass
    else:
        raise ValueError(f"Unsupported padding mode: {mode}")
    return padded


def build_valid_col_mask(num_valid_cols: int, rows: int = 4, target_cols: int = 8) -> np.ndarray:
    mask = np.zeros((rows, target_cols), dtype=np.float32)
    mask[:, :num_valid_cols] = 1.0
    return mask


def pad_sample_matrices(
    h_matrix: np.ndarray,
    mask_matrix: np.ndarray,
    l_matrix: np.ndarray | None = None,
    target_cols: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    valid_cols = h_matrix.shape[1]
    h_padded = pad_columns(h_matrix, target_cols=target_cols, mode="edge")
    m_padded = pad_columns(mask_matrix, target_cols=target_cols, mode="zero")
    l_padded = None if l_matrix is None else pad_columns(l_matrix, target_cols=target_cols, mode="edge")
    valid_col_mask = build_valid_col_mask(valid_cols, rows=h_matrix.shape[0], target_cols=target_cols)
    return h_padded, m_padded, l_padded, valid_col_mask


def masked_positions(mask: np.ndarray, valid_col_mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.float32) * valid_col_mask


def apply_data_consistency_numpy(pred: np.ndarray, coarse: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return pred * mask + coarse * (1.0 - mask)


def build_position_encodings(rows: int = 4, cols: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    row_positions = np.linspace(0.0, 1.0, rows, dtype=np.float32).reshape(rows, 1)
    col_positions = np.linspace(0.0, 1.0, cols, dtype=np.float32).reshape(1, cols)
    return np.repeat(row_positions, cols, axis=1), np.repeat(col_positions, rows, axis=0)
