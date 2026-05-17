import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.nmf.first_stage import generate_processed_samples
from src.utils.config import load_config
from src.utils.io import write_text_lines
from src.utils.seed import set_seed


DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed"


def build_splits(sample_ids: list[str], split_dir: Path, train_ratio: float, val_ratio: float) -> None:
    num_samples = len(sample_ids)
    train_end = int(num_samples * train_ratio)
    val_end = train_end + int(num_samples * val_ratio)
    write_text_lines(split_dir / "train.txt", sample_ids[:train_end])
    write_text_lines(split_dir / "val.txt", sample_ids[train_end:val_end])
    write_text_lines(split_dir / "test.txt", sample_ids[val_end:])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate processed dataset for NMF + lightweight refiner.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    config = load_config(args.config)
    if args.num_samples is not None:
        config["dataset"]["num_samples"] = args.num_samples

    processed_dir = Path(args.output_dir)
    split_dir = processed_dir.parent / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(config["dataset"]["seed"]))
    sample_ids = generate_processed_samples(config, processed_dir=processed_dir)
    build_splits(sample_ids, split_dir, float(config["splits"]["train_ratio"]), float(config["splits"]["val_ratio"]))

    print(f"Generated {len(sample_ids)} valid samples into {processed_dir.resolve()}")
    print(f"Split files saved to {split_dir.resolve()}")
    print("Next step: train the residual refiner directly on the generated dataset.")


if __name__ == "__main__":
    main()
