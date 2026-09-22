from __future__ import annotations

import argparse

from fedgagan.demo import create_demo_dataset
from fedgagan.pipeline import run_from_path


ALL_STAGES = ["preprocess", "search", "train", "generate", "evaluate"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fedgagan", description="Reproduce optimized federated WGAN-GP for IDS data"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run selected pipeline stages")
    run.add_argument("--config", required=True, help="Path to YAML configuration")
    run.add_argument(
        "--stages",
        default=",".join(ALL_STAGES),
        help="Comma-separated stages: preprocess,search,train,generate,evaluate",
    )
    demo = subparsers.add_parser("make-demo-data", help="Create the smoke-test CSV")
    demo.add_argument("--output", default="data/raw/demo.csv")
    demo.add_argument("--rows", type=int, default=2400)
    demo.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "make-demo-data":
        path = create_demo_dataset(args.output, args.rows, args.seed)
        print(f"Created {path}")
        return
    stages = [stage.strip() for stage in args.stages.split(",") if stage.strip()]
    run_from_path(args.config, stages)


if __name__ == "__main__":
    main()

