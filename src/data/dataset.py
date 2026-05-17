from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.normalization import MinMaxNormalizer
from src.data.preprocessing import build_position_encodings


def _load_ids_from_split(split_file: Path) -> List[str]:
    with split_file.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


class NMFCompletionDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        split_dir: str | Path | None = None,
        normalizer: Optional[MinMaxNormalizer] = None,
        target_cols: int = 8,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split_dir = Path(split_dir) if split_dir else self.data_dir.parent / "splits"
        self.normalizer = normalizer
        self.target_cols = target_cols
        self.sample_ids = _load_ids_from_split(self.split_dir / f"{split}.txt")
        row_pos, col_pos = build_position_encodings(rows=4, cols=target_cols)
        self.row_pos = row_pos
        self.col_pos = col_pos
        self._torch = torch

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample_id = self.sample_ids[index]
        payload = np.load(self.data_dir / f"sample_{sample_id}.npz", allow_pickle=True)
        h_matrix = payload["H_padded"].astype(np.float32)
        mask_matrix = payload["M_padded"].astype(np.float32)
        valid_col_mask = payload["valid_col_mask"].astype(np.float32)
        observed_matrix = payload["X_obs_padded"].astype(np.float32)

        if self.normalizer:
            h_matrix = self.normalizer.transform(h_matrix)
            observed_matrix = self.normalizer.transform(observed_matrix)

        sample = {
            "sample_id": sample_id,
            "H": self._torch.from_numpy(h_matrix).unsqueeze(0),
            "X_obs": self._torch.from_numpy(observed_matrix).unsqueeze(0),
            "M": self._torch.from_numpy(mask_matrix).unsqueeze(0),
            "valid_col_mask": self._torch.from_numpy(valid_col_mask).unsqueeze(0),
            "row_pos": self._torch.from_numpy(self.row_pos).unsqueeze(0),
            "col_pos": self._torch.from_numpy(self.col_pos).unsqueeze(0),
            "num_valid_cols": int(payload["num_valid_cols"]),
            "total_index": payload["total_index"].astype(np.int64),
            "aligned_locations": payload["aligned_locations"].astype(np.int64),
            "aligned_powers": payload["aligned_powers"].astype(np.float32),
            "NMF_H": payload["NMF_H"].astype(np.float32),
            "NMF_HH": payload["NMF_HH"].astype(np.float32),
            "metadata": {
                "source_positions": payload["source_positions"].astype(np.int64),
                "source_powers": payload["source_powers"].astype(np.float32),
                "signal_type_indices": payload["signal_type_indices"].astype(np.int64),
            },
        }
        return sample
