from copy import deepcopy
from pathlib import Path
from typing import Any, Dict


def _resolve_config_path(config_path: str | Path) -> Path:
    path = Path(config_path)
    if path.exists():
        return path

    project_root = Path(__file__).resolve().parents[2]
    candidate = project_root / path
    if candidate.exists():
        return candidate
    return path


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str | Path, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load configs/default.yaml.") from exc

    path = _resolve_config_path(config_path)
    project_root = Path(__file__).resolve().parents[2]
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if overrides:
        config = _deep_update(config, overrides)
    dataset_cfg = config.get("dataset", {})
    signal_library = dataset_cfg.get("signal_library_mat")
    if signal_library:
        signal_library_path = Path(signal_library)
        if not signal_library_path.is_absolute():
            dataset_cfg["signal_library_mat"] = str((project_root / signal_library_path).resolve())
    config["config_path"] = str(path.resolve())
    return config
