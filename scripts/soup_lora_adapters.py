import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import torch
from safetensors.torch import load_file, save_file


ADAPTER_WEIGHTS_NAME = "adapter_model.safetensors"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Average compatible LoRA adapter checkpoints into one adapter soup."
    )
    parser.add_argument(
        "--adapters",
        nargs="+",
        required=True,
        help="Adapter directories to average, for example checkpoint-300 checkpoint-400 checkpoint-500 checkpoint-600.",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="Optional non-negative weights, one per adapter. Defaults to equal weights.",
    )
    parser.add_argument("--output-dir", required=True, help="Output adapter directory.")
    parser.add_argument(
        "--copy-from",
        default=None,
        help="Directory to copy tokenizer/processor/config files from. Defaults to the first adapter.",
    )
    return parser.parse_args()


def normalize_weights(num_adapters: int, weights: Optional[List[float]]) -> List[float]:
    if weights is None:
        return [1.0 / num_adapters] * num_adapters
    if len(weights) != num_adapters:
        raise ValueError(f"Expected {num_adapters} weights, got {len(weights)}.")
    if any(weight < 0 for weight in weights):
        raise ValueError("Weights must be non-negative.")
    total = sum(weights)
    if total <= 0:
        raise ValueError("At least one weight must be positive.")
    return [weight / total for weight in weights]


def validate_adapter_dir(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Adapter directory not found: {path}")
    if not (path / ADAPTER_WEIGHTS_NAME).exists():
        raise FileNotFoundError(f"Missing {ADAPTER_WEIGHTS_NAME} in {path}")
    if not (path / "adapter_config.json").exists():
        raise FileNotFoundError(f"Missing adapter_config.json in {path}")


def copy_adapter_metadata(source_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in source_dir.iterdir():
        if path.name == ADAPTER_WEIGHTS_NAME:
            continue
        target = output_dir / path.name
        if path.is_file():
            shutil.copy2(path, target)
        elif path.is_dir() and path.name not in {"checkpoint", "checkpoints"}:
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(path, target)


def weighted_average(adapters: List[Path], weights: List[float]) -> Dict[str, torch.Tensor]:
    states = [load_file(adapter / ADAPTER_WEIGHTS_NAME, device="cpu") for adapter in adapters]
    reference_keys = set(states[0].keys())

    for adapter, state in zip(adapters[1:], states[1:]):
        keys = set(state.keys())
        if keys != reference_keys:
            missing = sorted(reference_keys - keys)[:10]
            extra = sorted(keys - reference_keys)[:10]
            raise ValueError(f"{adapter} has incompatible keys. missing={missing} extra={extra}")

    averaged: Dict[str, torch.Tensor] = {}
    for key in sorted(reference_keys):
        first = states[0][key]
        for adapter, state in zip(adapters[1:], states[1:]):
            tensor = state[key]
            if tensor.shape != first.shape:
                raise ValueError(f"{adapter}:{key} shape {tuple(tensor.shape)} != {tuple(first.shape)}")

        if torch.is_floating_point(first):
            acc = torch.zeros_like(first, dtype=torch.float32)
            for weight, state in zip(weights, states):
                acc += state[key].to(torch.float32) * weight
            averaged[key] = acc.to(first.dtype)
        else:
            if not all(torch.equal(first, state[key]) for state in states[1:]):
                raise ValueError(f"Non-floating tensor differs across adapters: {key}")
            averaged[key] = first

    return averaged


def main() -> None:
    args = parse_args()
    adapters = [Path(path) for path in args.adapters]
    for adapter in adapters:
        validate_adapter_dir(adapter)

    weights = normalize_weights(len(adapters), args.weights)
    copy_from = Path(args.copy_from) if args.copy_from else adapters[0]
    validate_adapter_dir(copy_from)

    output_dir = Path(args.output_dir)
    copy_adapter_metadata(copy_from, output_dir)
    averaged = weighted_average(adapters, weights)
    save_file(averaged, output_dir / ADAPTER_WEIGHTS_NAME, metadata={"format": "pt"})

    soup_config = {
        "adapters": [str(adapter) for adapter in adapters],
        "weights": weights,
        "copy_from": str(copy_from),
        "output_dir": str(output_dir),
    }
    (output_dir / "adapter_soup_config.json").write_text(
        json.dumps(soup_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Averaged {len(adapters)} adapters")
    for adapter, weight in zip(adapters, weights):
        print(f"- {adapter}: {weight:.6f}")
    print(f"Wrote adapter soup: {output_dir}")


if __name__ == "__main__":
    main()
