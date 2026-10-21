#!/usr/bin/env python3
"""Canonical launcher for finite-sample/CPDAG robustness v1."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.finite_sample_robustness.campaign import run_campaign
from experiments.finite_sample_robustness.design import DEFAULT_CONFIG, load_config
from scripts._output_safety import prepare_output_directory


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    output_dir = prepare_output_directory(
        arguments.output_dir,
        overwrite=arguments.overwrite,
        protected=(arguments.config,),
    )
    config = load_config(arguments.config)
    seed = int(config["default_seed"] if arguments.seed is None else arguments.seed)
    run_campaign(
        config_path=arguments.config,
        output_dir=output_dir,
        mode="pilot" if arguments.pilot else "full",
        seed=seed,
        workers=arguments.workers,
        resume=arguments.resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
