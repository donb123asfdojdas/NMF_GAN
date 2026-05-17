import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.train_refiner import train_refiner
from src.utils.config import load_config
from src.utils.seed import set_seed


DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"
DEFAULT_DATA_DIR = ROOT / "data" / "processed"
DEFAULT_SPLIT_DIR = ROOT / "data" / "splits"
DEFAULT_CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train lightweight masked residual refiner.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--split-dir", default=str(DEFAULT_SPLIT_DIR))
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config["dataset"]["seed"]))
    best_checkpoint, normalization_path = train_refiner(
        config=config,
        data_dir=args.data_dir,
        split_dir=args.split_dir,
        checkpoint_dir=args.checkpoint_dir,
    )
    print(f"Best checkpoint saved to {Path(best_checkpoint).resolve()}")
    print(f"Normalization parameters saved to {Path(normalization_path).resolve()}")


if __name__ == "__main__":
    main()
