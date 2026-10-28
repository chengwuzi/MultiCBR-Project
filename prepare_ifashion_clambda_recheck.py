"""Prepare round seven after explicitly selecting the validation-best fusion group."""

import argparse
import copy
from pathlib import Path
import sys

import yaml

from run_ifashion_experiments import PROJECT_ROOT, deep_merge, load_plan, validate_config


FUSION_PLAN = PROJECT_ROOT / "tuning_plans/ifashion_fusion_local_06.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "tuning_plans/ifashion_clambda_recheck_07.yaml"


def build_plan(fusion_experiment, gpu=None):
    source = load_plan(FUSION_PLAN, gpu=gpu)
    if fusion_experiment == "baseline":
        with (PROJECT_ROOT / source["base_config"]).open(encoding="utf-8") as stream:
            conf = yaml.safe_load(stream)["iFashion"]
        conf = deep_merge(conf, source.get("common_overrides", {}))
        label = "第四轮基线 ui_to_bi_010（验证分数 0.5638427481）"
    else:
        selected = next((e for e in source["experiments"] if e["id"] == fusion_experiment), None)
        if selected is None:
            raise ValueError(f"Unknown fusion experiment: {fusion_experiment}")
        conf = copy.deepcopy(selected["config"])
        label = f"第六轮 {fusion_experiment}"
    validate_config(conf, source["seed"])
    weights = conf["fusion_weights"]["modal_weight"]
    experiments = []
    for eid, value in (("clambda_025", 0.25), ("clambda_030", 0.30), ("clambda_040", 0.40)):
        candidate = copy.deepcopy(conf)
        candidate["c_lambdas"] = [value]
        validate_config(candidate, source["seed"])
        experiments.append({
            "id": eid,
            "note": f"固定选定融合权重 {weights}；c_lambda={value}，与同融合权重下已有 0.35 对照。",
            "overrides": {"c_lambdas": [value]},
        })
    return {
        "version": 1,
        "round_id": "stage1_clambda_recheck_07",
        "note": f"第七轮融合后对比权重回查：人工按验证分数选定 {label}，融合权重 {weights}；"
                "搜索 c_lambda=0.25/0.30/0.40，固定温度=0.2、40 轮、seed=2026，关闭融合前对比。",
        "base_config": source["base_config"],
        "seed": source["seed"],
        "gpu": source["gpu"],
        "max_attempts": 3,
        "result_file": "实验结果存档/Ifashion调参.txt",
        # Freeze every effective input, including nested diffusion and layer settings.
        "common_overrides": conf,
        "experiments": experiments,
    }


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fusion-experiment", required=True,
                        choices=("baseline", "ub_005", "ub_0075", "ub_0125"),
                        help="explicitly choose the validation winner, including the existing baseline")
    parser.add_argument("--gpu", "-g", default=None, help="GPU for the generated plan")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="new plan path; never overwrite")
    parser.add_argument("--dry-run", action="store_true", help="print the plan without writing or training")
    args = parser.parse_args(argv)
    try:
        plan = build_plan(args.fusion_experiment, gpu=args.gpu)
        serialized = yaml.safe_dump(plan, allow_unicode=True, sort_keys=False)
        if args.dry_run:
            print(serialized, end="")
        else:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(serialized)
            print(f"Prepared: {args.output.resolve()}")
            print("Plan prepared only; run the experiment controller separately on the server.")
        return 0
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
