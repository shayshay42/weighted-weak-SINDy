from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .artifacts import (
    atomic_write_json, command_line, environment_snapshot, sha256_file,
    sha256_json, source_hash, utc_now,
)
from .config import load_config, save_config
from .data import noise_label


def merge_tuned_configs(
    base_config_path: str | Path,
    tuned_config_paths: list[str | Path],
    output_path: str | Path,
) -> Path:
    merged = load_config(base_config_path)
    merged.setdefault("noise_overrides", {})
    merged.setdefault("method_overrides", {})
    merged.setdefault("plot_order_by_noise", {})
    merged.setdefault("tuning_result", {}).setdefault("by_noise", {})
    input_records: list[dict[str, Any]] = []
    for path_value in tuned_config_paths:
        path = Path(path_value)
        tuned = load_config(path)
        by_noise = tuned.get("tuning_result", {}).get("by_noise", {})
        if len(by_noise) != 1:
            raise ValueError(f"expected exactly one tuned noise level in {path}, found {sorted(by_noise)}")
        label = next(iter(by_noise))
        merged["noise_overrides"][label] = tuned.get("noise_overrides", {}).get(label, {})
        merged["method_overrides"][label] = tuned["method_overrides"][label]
        merged["plot_order_by_noise"][label] = tuned["plot_order_by_noise"][label]
        merged["tuning_result"]["by_noise"][label] = by_noise[label]
        input_records.append({"path": str(path.resolve()), "sha256": sha256_file(path), "noise_label": label})
    expected = {noise_label(float(level)) for level in merged["data"]["noise_levels"]}
    actual = set(merged["method_overrides"])
    if not expected.issubset(actual):
        raise ValueError(f"missing tuned noise levels: {sorted(expected - actual)}")
    merged["tuning_result"]["merged_at"] = utc_now()
    merged["tuning_result"]["inputs"] = input_records
    output_path = Path(output_path)
    save_config(merged, output_path)
    atomic_write_json(output_path.with_suffix(".manifest.json"), {
        "schema": "lorenz63-frozen-config-manifest-v2", "status": "complete",
        "completed_at": utc_now(), "output": str(output_path.resolve()),
        "output_sha256": sha256_file(output_path), "inputs": input_records,
        "base_config_hash": sha256_json(load_config(base_config_path)),
        "source_hash": source_hash(), "command": command_line(),
        "environment": environment_snapshot(),
    })
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge independently tuned Lorenz63 v2 noise configs.")
    parser.add_argument("--base-config", default="configs/v2/lorenz63_paper.json")
    parser.add_argument("--tuned-configs", nargs="+", required=True)
    parser.add_argument("--output", default="configs/v2/lorenz63_frozen.json")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    print(merge_tuned_configs(args.base_config, args.tuned_configs, args.output))


if __name__ == "__main__":
    main()
