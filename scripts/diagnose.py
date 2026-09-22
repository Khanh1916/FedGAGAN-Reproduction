#!/usr/bin/env python3
"""Preflight checks for local and Google Colab FedGAGAN runs."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import shutil
import sys
from pathlib import Path


REQUIRED = {
    "joblib": "joblib",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "pandas": "pandas",
    "yaml": "PyYAML",
    "sklearn": "scikit-learn",
    "seaborn": "seaborn",
    "tensorflow": "tensorflow",
}


def status(ok: bool, message: str) -> None:
    print(f"[{'OK' if ok else 'FAIL'}] {message}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check FedGAGAN runtime and data configuration")
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()

    failures = 0
    print("FedGAGAN preflight")
    print(f"Python: {sys.version.split()[0]} | OS: {platform.platform()}")
    print(f"Working directory: {Path.cwd()}")

    if sys.version_info < (3, 10):
        status(False, "Python 3.10 or newer is required")
        failures += 1
    else:
        status(True, "Python version is supported")

    loaded = {}
    for module_name, package_name in REQUIRED.items():
        try:
            module = importlib.import_module(module_name)
            loaded[module_name] = module
            version = getattr(module, "__version__", "unknown")
            status(True, f"{package_name} {version}")
        except Exception as exc:
            status(False, f"{package_name} unavailable: {exc}")
            failures += 1

    tensorflow = loaded.get("tensorflow")
    if tensorflow is not None:
        gpu_devices = tensorflow.config.list_physical_devices("GPU")
        status(bool(gpu_devices), f"TensorFlow GPUs detected: {len(gpu_devices)}")
        if args.require_gpu and not gpu_devices:
            failures += 1

    config_path = Path(args.config)
    if not config_path.is_file():
        status(False, f"Configuration not found: {config_path}")
        failures += 1
    else:
        try:
            from fedgagan.config import load_config
            from fedgagan.data import load_frames

            config = load_config(config_path)
            status(True, f"Configuration valid: {config_path}")
            try:
                frame = load_frames(config.data)
                status(True, f"Matched dataset rows: {len(frame):,}")
                if config.data.label_column not in frame.columns:
                    status(False, f"Label column missing: {config.data.label_column}")
                    failures += 1
                else:
                    counts = frame[config.data.label_column].value_counts(dropna=False).head(20)
                    print("Top raw label counts:")
                    print(counts.to_string())
            except FileNotFoundError as exc:
                status(False, str(exc))
                failures += 1
        except Exception as exc:
            status(False, f"Configuration/data check failed: {exc}")
            failures += 1

    disk = shutil.disk_usage(Path.cwd())
    free_gb = disk.free / (1024**3)
    status(free_gb >= 2, f"Free disk space: {free_gb:.1f} GiB")
    if free_gb < 2:
        failures += 1

    result = {"status": "PASS" if failures == 0 else "FAIL", "failures": failures}
    print(json.dumps(result))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

