import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: str | Path, array: np.ndarray) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    pd.DataFrame(array).to_csv(target, header=False, index=False)


def read_csv(path: str | Path) -> np.ndarray:
    return pd.read_csv(path, header=None).values


def save_npz(path: str | Path, **arrays: Any) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    np.savez(target, **arrays)


def write_text_lines(path: str | Path, lines: Iterable[str]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(f"{line}\n")
