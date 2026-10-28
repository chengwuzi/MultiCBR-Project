"""Run all 14 iFashion candidates, carrying validation winners through rounds 6-10."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sys

import yaml

from experiment_io import atomic_write_json, validation_score
import prepare_ifashion_clambda_recheck as recheck
import prepare_ifashion_followup as followup
import run_ifashion_experiments as runner


PROJECT_ROOT = Path(__file__).resolve().parent
BASELINE_EXECUTION = "20261025T082203_9dec4c53"
PIPELINE_DIR = PROJECT_ROOT / "log/ifashion_tuning/pipeline_06_10"


def initial_baseline(plan):
    with runner.resolve_path(plan["base_config"]).open(encoding="utf-8") as stream:
        conf = runner.deep_merge(yaml.safe_load(stream)["iFashion"], plan.get("common_overrides", {}))
    # Verify the historical score against its archived effective configuration.
    archive = Path(plan["result_file"]).read_text(encoding="utf-8")
    block = next((b for b in re.split(r"^EXECUTION_ID: ", archive, flags=re.M)[1:]
                  if b.splitlines()[0] == BASELINE_EXECUTION), None)
    if block is None:
        raise ValueError("The historical baseline execution is missing from the result archive")
    recorded = yaml.safe_load(block.split("完整实际配置:\n", 1)[1].split("\n尝试 ", 1)[0])
    if any(recorded.get(k) != value for k, value in conf.items()):
        raise ValueError("The frozen baseline differs from the historical result configuration")
    match = re.search(r"^验证选模分数: ([0-9.]+)$", block, re.M)
    if match is None or float(match[1]) <= 0:
        raise ValueError("Invalid historical baseline validation score")
    return {"config": conf, "selection_score": float(match[1]),
            "execution_id": BASELINE_EXECUTION, "round_id": "stage3_fusion_04",
            "experiment_id": "ui_to_bi_010", "source": "historical_result_archive"}


def preview(gpu):
    first = runner.load_plan(recheck.FUSION_PLAN, gpu=gpu)
    baseline = initial_baseline(first)
    second = recheck.build_plan("baseline", gpu=gpu)
    with followup.STAGE_FILE.open(encoding="utf-8") as stream:
        stages = yaml.safe_load(stream)["stages"]
    groups = [(first["round_id"], first["experiments"]), (second["round_id"], second["experiments"])]
    groups.extend((stages[s]["round_id"], stages[s]["experiments"])
                  for s in ("ctemp", "anchor_cl", "bi_noise"))
    total = sum(len(experiments) for _, experiments in groups)
    if [len(experiments) for _, experiments in groups] != [3, 3, 2, 3, 3]:
        raise ValueError("The pipeline requires exactly 3+3+2+3+3 candidates")
    for _, experiments in groups:
        for experiment in experiments:
            runner.validate_config(runner.deep_merge(baseline["config"], experiment["overrides"]), first["seed"])
    print(f"完整调参总览：5 批，共 {total} 组 | GPU: {first['gpu']} | seed=2026 | 每组 40 轮", flush=True)
    for number, (round_id, experiments) in enumerate(groups, 6):
        print(f"第 {number} 轮 | {round_id} | {len(experiments)} 组", flush=True)
        for experiment in experiments:
            note = experiment["note"]
            if number == 7:
                note = (f"c_lambda={experiment['overrides']['c_lambdas'][0]}；"
                        "融合权重继承第六轮实际验证优胜配置。")
            print(f"  {experiment['id']}: {note}", flush=True)
    print("后续批次自动继承实际验证优胜配置；没有提升则保留输入基线。", flush=True)
    return first, baseline


def signature(first):
    digest = hashlib.sha256()
    paths = ["run_ifashion_pipeline.py", "run_ifashion_experiments.py", "experiment_io.py",
             "prepare_ifashion_clambda_recheck.py", "prepare_ifashion_followup.py",
             "train.py", "models/model.py", "utility.py"]
    paths.extend(str(p) for p in (recheck.FUSION_PLAN, followup.STAGE_FILE,
                                  runner.resolve_path(first["base_config"])))
    for rel in paths:
        path = PROJECT_ROOT / rel
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    digest.update(first["gpu"].encode())
    return digest.hexdigest()


def select_winner(plan, baseline):
    state_path = PROJECT_ROOT / "log/ifashion_tuning" / plan["round_id"] / "state.json"
    state = runner.load_state(state_path)
    best = copy.deepcopy(baseline)
    choice = "baseline"
    for experiment in plan["experiments"]:
        record = runner.latest_execution(state, experiment)
        if record is None or record["status"] != "SUCCESS":
            raise RuntimeError(f"Incomplete round: {plan['round_id']}/{experiment['id']}")
        attempt = record["attempts"][-1]
        result_path = Path(attempt["config_snapshot"]).with_name(f"attempt_{attempt['number']}_result.json")
        result = runner.validate_result(result_path, experiment["config"], record["result"]["config"]["info"])
        score = validation_score(result["metrics"])
        if score > best["selection_score"]:
            choice = experiment["id"]
            best = {"config": copy.deepcopy(experiment["config"]), "selection_score": score,
                    "execution_id": record["execution_id"], "round_id": plan["round_id"],
                    "experiment_id": choice, "metrics": result["metrics"],
                    "best_epoch": result["best_epoch"], "artifacts": result["artifacts"],
                    "source": "completed_pipeline_experiment"}
    return choice, best


def freeze_plan(path, payload):
    if path.exists():
        if yaml.safe_load(path.read_text(encoding="utf-8")) != payload:
            raise RuntimeError(f"An existing pipeline plan has different inputs: {path}")
    else:
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(payload, stream, allow_unicode=True, sort_keys=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)


def run(gpu="0", dry_run=False):
    first, baseline = preview(gpu)
    if dry_run:
        print("DRY RUN：仅预览全部 14 组，不写文件、不启动训练。", flush=True)
        return 0
    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
    with runner.round_lock(PIPELINE_DIR / "controller.lock"):
        state_path = PIPELINE_DIR / "state.json"
        fingerprint = signature(first)
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if (state.get("version") != 1 or state.get("signature") != fingerprint
                    or state.get("initial_baseline") != baseline):
                raise RuntimeError("Pipeline inputs, code or GPU changed; resume with the original inputs")
        else:
            state = {"version": 1, "signature": fingerprint, "gpu": first["gpu"],
                     "stages": [], "initial_baseline": baseline}
        state.update(status="RUNNING", updated_at=runner.timestamp())
        state.pop("failed_round", None)
        atomic_write_json(state_path, state)
        source_path = recheck.FUSION_PLAN
        choice = None
        try:
            for index in range(5):
                if index == 0:
                    payload = yaml.safe_load(recheck.FUSION_PLAN.read_text(encoding="utf-8"))
                elif index == 1:
                    payload = recheck.build_plan(choice, gpu=gpu)
                else:
                    payload, _ = followup.build_plan(("ctemp", "anchor_cl", "bi_noise")[index - 2],
                                                     source_path, choice, gpu=gpu)
                payload["note"] = payload["note"].replace("人工按验证分数选定", "自动按验证综合分数选定").replace(
                    "按验证综合分数人工选定", "按验证综合分数自动选定")
                source_path = PIPELINE_DIR / f"round_{index + 6}_plan.yaml"
                freeze_plan(source_path, payload)
                plan = runner.load_plan(source_path, gpu=gpu)
                print(f"\n总进度：第 {index + 1}/5 批，第 {index + 6} 轮；此前已完成 "
                      f"{sum([3, 3, 2, 3, 3][:index])}/14 组", flush=True)
                code = runner.run(source_path, gpu=gpu)
                if code:
                    state.update(status="FAILED", failed_round=plan["round_id"], updated_at=runner.timestamp())
                    atomic_write_json(state_path, state)
                    print("本批仍有失败项，停止后续批次。修复后使用同一命令续跑。", flush=True)
                    return code
                choice, winner = select_winner(plan, baseline)
                entry = {"round_id": plan["round_id"], "plan": str(source_path),
                         "selected_experiment": choice, "winner": winner}
                if len(state["stages"]) > index:
                    if state["stages"][index] != entry:
                        raise RuntimeError("Previously selected results changed; downstream plans were not overwritten")
                else:
                    state["stages"].append(entry)
                baseline = winner
                state.update(best=baseline, updated_at=runner.timestamp())
                atomic_write_json(state_path, state)
                print(f"本批选优：{choice} | 当前最佳验证分数：{baseline['selection_score']:.10f}", flush=True)
            freeze_plan(PIPELINE_DIR / "best_config.yaml", {"iFashion": baseline["config"]})
            atomic_write_json(PIPELINE_DIR / "best_result.json", baseline)
            state.update(status="COMPLETE", updated_at=runner.timestamp())
            atomic_write_json(state_path, state)
            print(f"\n全部 14/14 组已完成。最佳配置：{PIPELINE_DIR / 'best_config.yaml'}", flush=True)
            return 0
        except KeyboardInterrupt:
            state.update(status="INTERRUPTED", updated_at=runner.timestamp())
            atomic_write_json(state_path, state)
            raise
        except Exception:
            state.update(status="ERROR", updated_at=runner.timestamp())
            atomic_write_json(state_path, state)
            raise


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", "-g", default="0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        return run(args.gpu, args.dry_run)
    except KeyboardInterrupt:
        print("已停止；使用同一命令续跑，未完成实验从第 0 轮重新训练。", file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError, TypeError, IndexError, RuntimeError, yaml.YAMLError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
