#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run reviewed Youshu K search, then NetEase K-first search, serially."""
from __future__ import annotations

import argparse
from pathlib import Path
import signal
import subprocess
import sys

import yaml

import run_experiments as runner


ROUNDS = ("youshu_round_03_k", "netease_round_01_k")


def reviewed_rounds(plan):
    rounds = []
    for rid, dataset in zip(ROUNDS, ("Youshu", "NetEase")):
        round_ = next((r for r in plan["rounds"] if r["id"] == rid), None)
        if round_ is None or round_["dataset"] != dataset or round_["selection_rule"] != runner.CROSS_RULE:
            raise ValueError(f"Missing or incompatible reviewed round: {rid}")
        rounds.append(round_)
    return rounds


def verify_complete(plan_path, round_id, gpu=None, device=None):
    # Re-check current full-precision results and artifacts, rather than trusting
    # an old summary file or a successful subprocess exit alone.
    plan = runner.load_plan(plan_path, gpu, device)
    round_ = next(r for r in reviewed_rounds(plan) if r["id"] == round_id)
    state = runner.load_state(plan["work_dir"] / "history.json")
    runner.resolve_dependencies(plan, round_, state)
    if any(not runner.check_success(runner.latest_execution(state, round_id, item), item, round_, plan)
           for item in round_["experiments"]):
        raise RuntimeError(f"Round is incomplete or artifacts failed verification: {round_id}")


def run(plan_path, gpu=None, device=None, dry_run=False):
    plan_path = Path(plan_path).resolve()
    plan = runner.load_plan(plan_path, gpu, device)
    reviewed_rounds(plan)
    if dry_run:
        print("DRY RUN: Youshu K search -> NetEase initial -> K -> lambda_1 -> lambda_0 -> omega -> tau -> lambda_2",
              flush=True)
        for rid in ROUNDS:
            code = runner.run(plan_path, rid, dry_run=True, gpu=gpu, device=device)
            if code:
                return code
        return 0
    # Distinct from the per-round controller lock, held for this whole sequence.
    with runner.file_lock(plan["work_dir"] / "k_search_pipeline.lock"):
        for rid in ROUNDS:
            print(f"Starting reviewed round: {rid}", flush=True)
            code = runner.run(plan_path, rid, gpu=gpu, device=device)
            if code:
                print(f"Stopped at {rid}; later rounds were not started. Resume with the same command.", flush=True)
                return code
            verify_complete(plan_path, rid, gpu, device)
        print("Youshu K search and NetEase K-first search completed and verified.", flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=runner.MODEL_DIR / "搜参配置.yaml")
    parser.add_argument("--gpu", "-g")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        return run(args.plan, args.gpu, args.device, args.dry_run)
    except KeyboardInterrupt:
        print("Interrupted. Resume with the same command; unfinished trials restart from epoch 1.", file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, StopIteration,
            subprocess.TimeoutExpired, yaml.YAMLError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    sys.exit(main())
