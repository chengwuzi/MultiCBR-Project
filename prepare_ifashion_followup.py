"""Prepare the next iFashion search from an explicitly selected previous configuration."""

import argparse
import copy
from pathlib import Path
import sys

import yaml

from run_ifashion_experiments import PROJECT_ROOT, deep_merge, load_plan, validate_config


STAGE_FILE = PROJECT_ROOT / "tuning_plans/ifashion_followup_stages_20261028.yaml"


def build_plan(stage, source_plan, experiment, gpu=None):
    with STAGE_FILE.open(encoding="utf-8") as stream:
        definitions = yaml.safe_load(stream)
    if definitions["version"] != 1 or stage not in definitions["stages"]:
        raise ValueError(f"Unknown follow-up stage: {stage}")
    definition = definitions["stages"][stage]
    source = load_plan(source_plan, gpu=gpu)
    if source["round_id"] != definition["source_round"]:
        raise ValueError(f"{stage} requires source round {definition['source_round']}")
    if experiment == "baseline":
        with (PROJECT_ROOT / source["base_config"]).open(encoding="utf-8") as stream:
            conf = yaml.safe_load(stream)["iFashion"]
        conf = deep_merge(conf, source.get("common_overrides", {}))
    else:
        selected = next((e for e in source["experiments"] if e["id"] == experiment), None)
        if selected is None:
            raise ValueError(f"Unknown source experiment: {experiment}")
        conf = copy.deepcopy(selected["config"])
    validate_config(conf, source["seed"])
    if stage in ("ctemp", "anchor_cl") and conf["anchor_cl"]["enabled"]:
        raise ValueError("Temperature and anchor searches require a pre-fusion-disabled baseline")
    if stage == "ctemp" and conf["c_lambdas"][0] <= 0:
        raise ValueError("A positive post-fusion contrastive weight is required for temperature search")
    if stage == "anchor_cl":
        anchor = conf["anchor_cl"]
        required = {"temp": .2, "user_cl": True, "bundle_cl": True,
                    "ub_ui_user_weight": 1., "ub_bi_user_weight": 1.,
                    "ub_ui_bundle_weight": 1., "ub_bi_bundle_weight": 1.}
        if any(anchor.get(key) != value for key, value in required.items()):
            raise ValueError("Anchor search requires temp=0.2, both entity sides, and four unit weights")
    if stage == "bi_noise" and conf["aug_type"] != "Noise":
        raise ValueError("BI noise search requires aug_type=Noise")
    experiments = copy.deepcopy(definition["experiments"])
    for candidate in experiments:
        validate_config(deep_merge(conf, candidate["overrides"]), source["seed"])
    return {
        "version": 1,
        "round_id": definition["round_id"],
        "note": definition["note"] + f" 配置来源：{source['round_id']}/{experiment}；按验证综合分数人工选定。",
        "base_config": source["base_config"],
        "seed": source["seed"],
        "gpu": source["gpu"],
        "max_attempts": 3,
        "result_file": "实验结果存档/Ifashion调参.txt",
        "common_overrides": conf,
        "experiments": experiments,
    }, PROJECT_ROOT / definition["output"]


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("ctemp", "anchor_cl", "bi_noise"))
    parser.add_argument("--source-plan", required=True, type=Path,
                        help="previous round's effective plan, including its frozen common configuration")
    parser.add_argument("--experiment", required=True,
                        help="validation-best previous experiment ID, or baseline to keep the previous input")
    parser.add_argument("--gpu", "-g", default=None)
    parser.add_argument("--output", type=Path, default=None, help="new plan path; never overwrite")
    parser.add_argument("--dry-run", action="store_true", help="print without writing or training")
    args = parser.parse_args(argv)
    try:
        plan, default_output = build_plan(args.stage, args.source_plan, args.experiment, gpu=args.gpu)
        serialized = yaml.safe_dump(plan, allow_unicode=True, sort_keys=False)
        if args.dry_run:
            print(serialized, end="")
        else:
            output = args.output or default_output
            with output.open("x", encoding="utf-8") as stream:
                stream.write(serialized)
            print(f"Prepared: {output.resolve()}")
            print("Plan prepared only; run the experiment controller separately on the server.")
        return 0
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
