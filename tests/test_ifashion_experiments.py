"""Controller integration tests use a fake trainer, never a recommendation model."""

from contextlib import redirect_stdout
import copy
import io
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import yaml

import run_ifashion_experiments as runner
import prepare_ifashion_clambda_recheck as recheck
import prepare_ifashion_followup as followup
from experiment_io import atomic_write_json, save_best_result


ROOT = Path(__file__).resolve().parents[1]
FAKE_TRAINER = r'''
import argparse, json, os
from pathlib import Path
import yaml
p=argparse.ArgumentParser()
for key in ['--config','--result-json','-d','-g','-i']: p.add_argument(key)
a=p.parse_args()
conf=yaml.safe_load(Path(a.config).read_text(encoding='utf-8'))['iFashion']
assert os.environ['PYTHONHASHSEED']==str(conf['seed'])
assert os.environ['CUDA_VISIBLE_DEVICES']==a.g
assert conf['anchor_cl']['enabled'] is False
value=str(conf['c_lambdas'][0])
behavior=json.loads(Path('behavior.json').read_text())
counter=Path('counter_'+value+'.txt')
n=int(counter.read_text())+1 if counter.exists() else 1
counter.write_text(str(n))
action=behavior.get(value,'success')
print('FAKE TRAINER ONLY',value,n,flush=True)
if action=='fail' or (action=='twice' and n<=2): raise SystemExit(9)
if action=='missing': raise SystemExit(0)
conf.update(info=a.i,dataset='iFashion',model='AnchorViewBundleNet',gpu=a.g)
metrics={split:{metric:{str(k):.1 for k in conf['topk']} for metric in ['recall','ndcg']} for split in ['val','test']}
model=Path(a.result_json+'.model'); model.write_text('fake checkpoint')
config=Path(a.result_json+'.conf'); config.write_text('{}')
result={'completed':True,'config':conf,'best_epoch':4,'best_epoch_1based':5,'selection_score':.4,
        'metrics':metrics,'artifacts':{'checkpoint_model':str(model),'checkpoint_config':str(config)}}
if action=='nan': result['metrics']['test']['ndcg']['20']=float('nan')
if action=='wrong': result['config']['seed']=0
if action=='missing_checkpoint': model.unlink()
Path(a.result_json).write_text(json.dumps(result),encoding='utf-8')
'''


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.root_patch = mock.patch.object(runner, "PROJECT_ROOT", self.root)
        self.root_patch.start()
        (self.root / "train.py").write_text(FAKE_TRAINER, encoding="utf-8")
        (self.root / "models").mkdir()
        for name in ("models/model.py", "utility.py", "experiment_io.py"):
            (self.root / name).write_text("# no model executed\n")
        self.base = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
        data = self.root / "data" / "iFashion"
        data.mkdir(parents=True)
        for name in ("user_item.txt", "bundle_item.txt", "user_bundle_train.txt",
                     "user_bundle_tune.txt", "user_bundle_test.txt"):
            (data / name).write_text("0\t0\n")
        (data / "iFashion_data_size.txt").write_text("3\t100\t4\n")
        for entity in ("user", "bundle"):
            (self.root / f"{entity}.pt").write_bytes(b"unused by fake trainer")
            self.base["iFashion"]["latent_rebuild"][f"latent_diffusion_{entity}_embedding_path"] = f"{entity}.pt"
        self.base["iFashion"]["data_path"] = "data"
        (self.root / "base.yaml").write_text(yaml.safe_dump(self.base), encoding="utf-8")
        self.plan = {
            "version": 1, "round_id": "test_round", "note": "round note",
            "base_config": "base.yaml", "seed": 2026, "gpu": "0", "max_attempts": 3,
            "result_file": "results/result.txt",
            "common_overrides": {"anchor_cl": {"enabled": False}, "c_temps": [.2]},
            "experiments": [
                {"id": "a", "note": "experiment a", "overrides": {"c_lambdas": [.05]}},
                {"id": "b", "note": "experiment b", "overrides": {"c_lambdas": [.1]}},
            ],
        }
        self.path = self.root / "plan.yaml"
        self.save_plan()
        self.behavior({})

    def tearDown(self):
        self.root_patch.stop()
        self.temporary.cleanup()

    def save_plan(self):
        self.path.write_text(yaml.safe_dump(self.plan), encoding="utf-8")

    def behavior(self, value):
        (self.root / "behavior.json").write_text(json.dumps(value), encoding="utf-8")

    def run_plan(self, **kwargs):
        output = io.StringIO()
        with redirect_stdout(output):
            try:
                return runner.run(self.path, **kwargs)
            finally:
                self.console = output.getvalue()

    def state(self):
        return json.loads((self.root / "log/ifashion_tuning/test_round/state.json").read_text(encoding="utf-8"))

    def test_success_resume_and_report_repair(self):
        original = (self.root / "base.yaml").read_bytes()
        self.assertEqual(self.run_plan(), 0)
        records = self.state()["executions"]
        self.assertEqual([r["status"] for r in records], ["SUCCESS", "SUCCESS"])
        self.assertTrue(all(len(r["attempts"]) == 1 for r in records))
        self.assertEqual(self.run_plan(), 0)
        self.assertEqual(len(self.state()["executions"]), 2)
        report = self.root / "results/result.txt"
        text = report.read_text(encoding="utf-8")
        self.assertEqual(text.count("END_EXECUTION:"), 2)
        self.assertIn("round note", text)
        self.assertIn("experiment a", text)
        report.unlink()
        self.run_plan()
        self.assertEqual(report.read_text(encoding="utf-8"), text)
        self.assertEqual((self.root / "base.yaml").read_bytes(), original)

    def test_cli_runs_fake_plan_and_resumes_with_utf8_progress_and_reports(self):
        # Exercise the actual CLI in an isolated project with the fake trainer.
        for name in ("run_ifashion_experiments.py", "experiment_io.py"):
            (self.root / name).write_bytes((ROOT / name).read_bytes())
        command = [sys.executable, str(self.root / "run_ifashion_experiments.py"), "--plan", str(self.path)]
        self.behavior({"0.05": "fail"})
        first = subprocess.run(command + ["--gpu", "1"], cwd=self.root, capture_output=True, encoding="utf-8", timeout=30)
        self.assertEqual(first.returncode, 1, first.stderr)
        self.assertIn("GPU: 1", first.stdout)
        self.assertIn("已结束 2/2 | 已完成 1 | 三次机会耗尽 1", first.stdout)
        self.assertIn("a | c_lambda=0.05 | 失败：三次机会耗尽（3/3）", first.stdout)
        self.behavior({})
        resumed = subprocess.run(command + ["-g", "1"], cwd=self.root, capture_output=True, encoding="utf-8", timeout=30)
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        self.assertIn("b | c_lambda=0.1 | 已完成（历史成功，本次跳过）", resumed.stdout)
        self.assertIn("已结束 1/1 | 已完成 1 | 三次机会耗尽 0", resumed.stdout)
        self.assertEqual([r["status"] for r in self.state()["executions"]], ["FAILED", "SUCCESS", "SUCCESS"])
        self.assertTrue(all(r["gpu"] == "1" and r["result"]["config"]["gpu"] == "1"
                            for r in self.state()["executions"] if r["status"] == "SUCCESS"))
        report = (self.root / "results/result.txt").read_text(encoding="utf-8")
        self.assertEqual(report.count("END_EXECUTION:"), 3)
        self.assertEqual(report.count("实验总耗时（含重试）:"), 3)
        self.assertEqual(report.count("GPU: 1"), 3)

    def test_gpu_override_applies_before_fingerprinting_without_changing_plan(self):
        original = self.path.read_bytes()
        default = runner.load_plan(self.path)
        overridden = runner.load_plan(self.path, gpu="1")
        self.assertEqual(default["gpu"], "0")
        self.assertEqual(overridden["gpu"], "1")
        self.assertEqual(default["experiments"][0]["config"], overridden["experiments"][0]["config"])
        self.assertNotEqual(default["experiments"][0]["fingerprint"], overridden["experiments"][0]["fingerprint"])
        self.assertEqual(self.run_plan(gpu="1", dry_run=True), 0)
        self.assertIn("GPU: 1", self.console)
        self.assertFalse((self.root / "log").exists())
        self.assertEqual(self.path.read_bytes(), original)

    def test_retry_success_on_third_attempt(self):
        self.behavior({"0.05": "twice"})
        self.assertEqual(self.run_plan(), 0)
        record = self.state()["executions"][0]
        self.assertEqual([a["status"] for a in record["attempts"]], ["FAILED", "FAILED", "SUCCESS"])
        self.assertEqual(len({a["config_snapshot"] for a in record["attempts"]}), 3)

    def test_report_records_execution_and_attempt_times_including_retries(self):
        self.behavior({"0.05": "twice", "0.1": "fail"})
        self.assertEqual(self.run_plan(), 1)
        report = (self.root / "results/result.txt").read_text(encoding="utf-8")
        records = self.state()["executions"]
        self.assertEqual([r["status"] for r in records], ["SUCCESS", "FAILED"])
        for record in records:
            self.assertGreaterEqual(record["duration_seconds"], 0)
            self.assertGreaterEqual(record["duration_seconds"] + .003,
                                    sum(a["duration_seconds"] for a in record["attempts"]))
            self.assertIn(f"开始时间(UTC+08:00): {record['started_at']}", report)
            self.assertIn(f"结束时间(UTC+08:00): {record['ended_at']}", report)
            self.assertIn(f"实验总耗时（含重试）: {runner.elapsed_text(record)}", report)
            for attempt in record["attempts"]:
                self.assertGreaterEqual(attempt["duration_seconds"], 0)
                self.assertIn(f"  开始时间(UTC+08:00): {attempt['started_at']}", report)
                self.assertIn(f"  结束时间(UTC+08:00): {attempt['ended_at']}", report)
                self.assertIn(f"  本次尝试耗时: {runner.elapsed_text(attempt)}", report)

    def test_console_lists_all_experiments_and_updates_retry_and_terminal_states(self):
        self.behavior({"0.05": "twice", "0.1": "fail"})
        self.assertEqual(self.run_plan(), 1)
        snapshots = [part for part in self.console.split("=" * 72) if "实验进度 |" in part]
        self.assertIn("计划 2 组，本次运行 2 组，跳过 0 组", snapshots[0])
        self.assertIn("1. a | c_lambda=0.05 | 等待运行", snapshots[0])
        self.assertIn("2. b | c_lambda=0.1 | 等待运行", snapshots[0])
        self.assertIn("轮次备注: round note", snapshots[0])
        self.assertIn("a | c_lambda=0.05 | 正在运行（第 1/3 次尝试）", self.console)
        self.assertIn("第 1/3 次尝试失败，等待第 2/3 次尝试", self.console)
        self.assertIn("第 2/3 次尝试失败，等待第 3/3 次尝试", self.console)
        self.assertIn("a | c_lambda=0.05 | 已完成（第 3/3 次尝试成功）", snapshots[-1])
        self.assertIn("b | c_lambda=0.1 | 失败：三次机会耗尽（3/3）", snapshots[-1])
        self.assertIn("已结束 2/2 | 已完成 1 | 三次机会耗尽 1 | 正在运行 0 | 等待 0", snapshots[-1])
        self.assertEqual([len(r["attempts"]) for r in self.state()["executions"]], [3, 3])

        self.behavior({})
        self.assertEqual(self.run_plan(), 0)
        self.assertIn("计划 2 组，本次运行 1 组，跳过 1 组", self.console)
        self.assertIn("a | c_lambda=0.05 | 已完成（历史成功，本次跳过）", self.console)
        self.assertIn("b | c_lambda=0.1 | 等待运行（上次失败，本次重新尝试）", self.console)
        self.assertIn("已结束 1/1 | 已完成 1 | 三次机会耗尽 0", self.console)
        self.assertEqual([r["experiment_id"] for r in self.state()["executions"]], ["a", "b", "b"])

    def test_exhaustion_retries_on_next_launch_and_explicit_failed_rerun(self):
        self.behavior({"0.05": "fail"})
        self.assertEqual(self.run_plan(), 1)
        first = copy.deepcopy(self.state()["executions"][0])
        self.assertEqual(first["status"], "FAILED")
        self.assertEqual(len(first["attempts"]), 3)
        self.assertEqual([r["experiment_id"] for r in self.state()["executions"]], ["a", "b"])
        self.assertEqual(self.run_plan(), 1)
        records = self.state()["executions"]
        self.assertEqual([r["experiment_id"] for r in records], ["a", "b", "a"])
        self.assertEqual(records[-1]["status"], "FAILED")
        self.assertEqual(len(records[-1]["attempts"]), 3)
        self.assertEqual(records[0], first)
        self.behavior({"0.05": "twice"})
        (self.root / "counter_0.05.txt").unlink()
        self.assertEqual(self.run_plan(mode="failed"), 0)
        records = self.state()["executions"]
        self.assertEqual(len(records), 4)
        self.assertEqual(records[0], first)
        self.assertEqual(records[-1]["experiment_id"], "a")
        self.assertEqual(records[-1]["status"], "SUCCESS")
        self.assertEqual(len(records[-1]["attempts"]), 3)

    def test_selected_successful_group_and_all_rerun(self):
        self.run_plan()
        self.run_plan(mode="selected", only=["b"])
        self.assertIn("b | c_lambda=0.1 | 等待运行（主动重跑）", self.console)
        self.assertEqual([r["experiment_id"] for r in self.state()["executions"]], ["a", "b", "b"])
        self.run_plan(mode="all")
        self.assertEqual(len(self.state()["executions"]), 5)
        with self.assertRaises(ValueError):
            self.run_plan(mode="selected", only=["unknown"])

    def test_missing_nonfinite_and_mismatched_results_retry(self):
        for action in ("missing", "nan", "wrong", "missing_checkpoint"):
            with self.subTest(action=action):
                self.behavior({"0.05": action})
                self.assertEqual(self.run_plan(mode="selected", only=["a"]), 1)
                record = self.state()["executions"][-1]
                self.assertEqual(record["status"], "FAILED")
                self.assertEqual(len(record["attempts"]), 3)

    def test_configuration_change_does_not_skip_old_success(self):
        self.run_plan()
        self.plan["experiments"][0]["overrides"]["c_lambdas"] = [.2]
        self.save_plan()
        self.run_plan()
        records = self.state()["executions"]
        self.assertEqual(len(records), 3)
        self.assertNotEqual(records[0]["fingerprint"], records[2]["fingerprint"])

    def test_dry_run_is_read_only_and_merge_preserves_nested_values(self):
        self.run_plan(dry_run=True)
        self.assertFalse((self.root / "log").exists())
        self.assertFalse((self.root / "results").exists())
        loaded = runner.load_plan(self.path)
        anchor = loaded["experiments"][0]["config"]["anchor_cl"]
        self.assertFalse(anchor["enabled"])
        self.assertEqual(anchor["anchor_cl_lambda"], self.base["iFashion"]["anchor_cl"]["anchor_cl_lambda"])
        self.assertTrue(anchor["user_cl"])

    def test_bad_plans_rejected_before_launch(self):
        variants = []
        bad = copy.deepcopy(self.plan); bad["experiments"][0]["overrides"]["c_lambdas"] = [.1, .2]; variants.append(bad)
        bad = copy.deepcopy(self.plan); bad["common_overrides"]["anchor_cl"]["enabeld"] = False; variants.append(bad)
        bad = copy.deepcopy(self.plan); bad["experiments"][1]["id"] = "a"; variants.append(bad)
        bad = copy.deepcopy(self.plan); bad["experiments"][0]["overrides"]["seed"] = 42; variants.append(bad)
        bad = copy.deepcopy(self.plan); bad["round_id"] = "../outside"; variants.append(bad)
        for bad in variants:
            self.path.write_text(yaml.safe_dump(bad), encoding="utf-8")
            with self.assertRaises(ValueError):
                self.run_plan()
        self.assertFalse((self.root / "log").exists())

    def test_interrupt_stops_queue_and_resume_restarts_group(self):
        original_wait = subprocess.Popen.wait
        triggered = []
        def interrupt_once(process, *args, **kwargs):
            if str(self.root / "train.py") in process.args and not triggered:
                triggered.append(process.pid)
                raise KeyboardInterrupt
            return original_wait(process, *args, **kwargs)
        with mock.patch.object(subprocess.Popen, "wait", interrupt_once):
            with self.assertRaises(KeyboardInterrupt):
                self.run_plan()
        records = self.state()["executions"]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["status"], "INTERRUPTED")
        self.assertGreaterEqual(records[0]["duration_seconds"], 0)
        self.assertGreaterEqual(records[0]["attempts"][-1]["duration_seconds"], 0)
        self.assertIn("a | c_lambda=0.05 | 已中断（下次启动重新运行）", self.console)
        self.assertIn("b | c_lambda=0.1 | 等待运行", self.console)
        self.assertFalse(runner.process_alive(triggered[0]))
        self.assertEqual(self.run_plan(), 0)
        self.assertEqual([r["status"] for r in self.state()["executions"]], ["INTERRUPTED", "SUCCESS", "SUCCESS"])

    def test_interrupt_between_failure_and_retry_preserves_attempt_timing(self):
        self.behavior({"0.05": "fail"})
        original_update = runner.ConsoleProgress.update
        def interrupt_on_retry(progress, experiment, status, label):
            original_update(progress, experiment, status, label)
            if status == "RETRY":
                raise KeyboardInterrupt
        with mock.patch.object(runner.ConsoleProgress, "update", interrupt_on_retry):
            with self.assertRaises(KeyboardInterrupt):
                self.run_plan()
        records = self.state()["executions"]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["status"], "INTERRUPTED")
        self.assertEqual(len(records[0]["attempts"]), 1)
        attempt = records[0]["attempts"][0]
        self.assertEqual(attempt["status"], "FAILED")
        self.assertIn("ended_at", attempt)
        self.assertGreaterEqual(attempt["duration_seconds"], 0)
        report = (self.root / "results/result.txt").read_text(encoding="utf-8")
        self.assertIn("状态: INTERRUPTED", report)
        self.assertNotIn("耗时: 未知", report)
        self.assertNotIn("结束时间(UTC+08:00): 未记录", report)

    def test_round_lock_and_stale_running_record(self):
        lock = self.root / "lock"
        with runner.round_lock(lock):
            with self.assertRaises(RuntimeError):
                with runner.round_lock(lock):
                    pass
        self.run_plan()
        path = self.root / "log/ifashion_tuning/test_round/state.json"
        state = self.state()
        state["executions"][0].update(status="RUNNING")
        state["executions"][0]["attempts"] = []
        (self.root / "results/result.txt").unlink()
        atomic_write_json(path, state)
        self.run_plan()
        self.assertEqual(self.state()["executions"][0]["status"], "INTERRUPTED")
        self.assertIsNone(self.state()["executions"][0]["duration_seconds"])
        report = (self.root / "results/result.txt").read_text(encoding="utf-8")
        self.assertIn("实验总耗时（含重试）: 未知", report)
        self.assertIn("结束时间为恢复检测时间", report)

    def test_live_orphan_worker_blocks_another_launch(self):
        self.run_plan()
        state = self.state()
        record = state["executions"][0]
        record["status"] = "RUNNING"
        record["attempts"][-1].update(status="RUNNING", pid=os.getpid())
        atomic_write_json(self.root / "log/ifashion_tuning/test_round/state.json", state)
        with self.assertRaisesRegex(RuntimeError, "still alive"):
            self.run_plan()

    def test_recheck_carries_selected_fusion_and_all_other_inputs(self):
        self.plan["experiments"][0].update(id="ub_0075", overrides={
            "fusion_weights": {"modal_weight": [.075, .325, .6]}})
        self.save_plan()
        expected = runner.load_plan(self.path)["experiments"][0]["config"]
        with mock.patch.object(recheck, "FUSION_PLAN", self.path):
            plan = recheck.build_plan("ub_0075", gpu="1")
        self.assertEqual(plan["common_overrides"], expected)
        self.assertEqual(plan["gpu"], "1")
        generated = self.root / "recheck.yaml"
        generated.write_text(yaml.safe_dump(plan), encoding="utf-8")
        candidates = runner.load_plan(generated)["experiments"]
        self.assertEqual([e["config"]["c_lambdas"][0] for e in candidates], [.25, .30, .40])
        for experiment in candidates:
            actual = dict(experiment["config"], c_lambdas=expected["c_lambdas"])
            self.assertEqual(actual, expected)

    def test_recheck_baseline_includes_source_common_overrides(self):
        self.plan["common_overrides"].update(batch_size_train=1024, epochs=50)
        self.save_plan()
        expected = runner.deep_merge(self.base["iFashion"], self.plan["common_overrides"])
        with mock.patch.object(recheck, "PROJECT_ROOT", self.root), \
                mock.patch.object(recheck, "FUSION_PLAN", self.path):
            generated = recheck.build_plan("baseline")
        self.assertEqual(generated["common_overrides"], expected)

    def test_generators_report_malformed_yaml_without_creating_output(self):
        self.path.write_text("version: [1\n", encoding="utf-8")
        destination = self.root / "invalid_output.yaml"
        with mock.patch.object(recheck, "FUSION_PLAN", self.path), \
                mock.patch.object(sys, "stderr", io.StringIO()) as errors:
            self.assertEqual(recheck.main(["--fusion-experiment", "baseline",
                                           "--output", str(destination)]), 2)
            self.assertEqual(followup.main(["--stage", "ctemp", "--source-plan", str(self.path),
                                            "--experiment", "baseline", "--output", str(destination)]), 2)
            self.assertEqual(errors.getvalue().count("ERROR:"), 2)
        self.assertFalse(destination.exists())

    def test_recheck_requires_explicit_choice_and_preserves_existing_plan(self):
        destination = self.root / "generated.yaml"
        with mock.patch.object(recheck, "PROJECT_ROOT", self.root), \
                mock.patch.object(recheck, "FUSION_PLAN", self.path), \
                mock.patch.object(subprocess, "Popen") as launch, \
                redirect_stdout(io.StringIO()), mock.patch.object(sys, "stderr", io.StringIO()):
            with self.assertRaises(ValueError):
                recheck.build_plan("unknown")
            with self.assertRaises(SystemExit) as missing_choice:
                recheck.main(["--output", str(destination)])
            self.assertEqual(missing_choice.exception.code, 2)
            args = ["--fusion-experiment", "baseline", "--output", str(destination)]
            self.assertEqual(recheck.main(args + ["--dry-run"]), 0)
            self.assertFalse(destination.exists())
            self.assertEqual(recheck.main(args), 0)
            original = destination.read_bytes()
            generated = runner.load_plan(destination)
            self.assertEqual(generated["experiments"][0]["config"]["fusion_weights"],
                             self.base["iFashion"]["fusion_weights"])
            self.assertEqual(recheck.main(args), 2)
            self.assertEqual(destination.read_bytes(), original)
            launch.assert_not_called()

    def test_followup_chain_preserves_selected_inputs_and_baseline_fallback(self):
        self.plan["round_id"] = "stage1_clambda_recheck_07"
        self.plan["common_overrides"]["fusion_weights"] = {"modal_weight": [.075, .325, .6]}
        self.save_plan()
        with mock.patch.object(followup, "PROJECT_ROOT", self.root):
            temperature, _ = followup.build_plan("ctemp", self.path, "b", gpu="1")
            self.assertEqual(temperature["gpu"], "1")
            source = self.root / "temperature.yaml"
            source.write_text(yaml.safe_dump(temperature), encoding="utf-8")
            loaded = runner.load_plan(source)
            self.assertEqual([e["config"]["c_temps"][0] for e in loaded["experiments"]], [.1, .3])
            for e in loaded["experiments"]:
                self.assertEqual(e["config"]["c_lambdas"], [.1])
                self.assertEqual(e["config"]["fusion_weights"]["modal_weight"], [.075, .325, .6])
            anchor, _ = followup.build_plan("anchor_cl", source, "ctemp_030")
            source = self.root / "anchor.yaml"
            source.write_text(yaml.safe_dump(anchor), encoding="utf-8")
            loaded = runner.load_plan(source)
            self.assertEqual([e["config"]["anchor_cl"]["anchor_cl_lambda"]
                              for e in loaded["experiments"]], [.001, .003, .01])
            for e in loaded["experiments"]:
                self.assertTrue(e["config"]["anchor_cl"]["enabled"])
                self.assertEqual(e["config"]["anchor_cl"]["temp"], .2)
                self.assertEqual(e["config"]["c_temps"], [.3])
            noise, _ = followup.build_plan("bi_noise", source, "anchor_0003")
            fallback, _ = followup.build_plan("bi_noise", source, "baseline")
            self.assertTrue(noise["common_overrides"]["anchor_cl"]["enabled"])
            self.assertEqual(noise["common_overrides"]["anchor_cl"]["anchor_cl_lambda"], .003)
            self.assertFalse(fallback["common_overrides"]["anchor_cl"]["enabled"])
            for plan in (noise, fallback):
                self.assertEqual(plan["common_overrides"]["c_temps"], [.3])
                self.assertEqual(plan["common_overrides"]["c_lambdas"], [.1])
                path = self.root / "noise.yaml"
                path.write_text(yaml.safe_dump(plan), encoding="utf-8")
                self.assertEqual([e["config"]["BI_ratios"][0]
                                  for e in runner.load_plan(path)["experiments"]], [0., .1, .3])

    def test_followup_rejects_wrong_round_unknown_group_and_inert_noise(self):
        with mock.patch.object(followup, "PROJECT_ROOT", self.root):
            with self.assertRaisesRegex(ValueError, "requires source round"):
                followup.build_plan("ctemp", self.path, "baseline")
            self.plan["round_id"] = "stage1_clambda_recheck_07"
            self.save_plan()
            with self.assertRaisesRegex(ValueError, "Unknown source experiment"):
                followup.build_plan("ctemp", self.path, "unknown")
            self.plan["round_id"] = "stage2_anchor_09"
            self.plan["common_overrides"]["aug_type"] = "ED"
            self.save_plan()
            with self.assertRaisesRegex(ValueError, "aug_type=Noise"):
                followup.build_plan("bi_noise", self.path, "baseline")

    def test_followup_cli_preview_generation_and_no_overwrite(self):
        self.plan["round_id"] = "stage1_clambda_recheck_07"
        self.save_plan()
        destination = self.root / "generated_temperature.yaml"
        args = ["--stage", "ctemp", "--source-plan", str(self.path),
                "--experiment", "baseline", "--output", str(destination), "--gpu", "1"]
        with mock.patch.object(followup, "PROJECT_ROOT", self.root), \
                mock.patch.object(subprocess, "Popen") as launch, \
                redirect_stdout(io.StringIO()), mock.patch.object(sys, "stderr", io.StringIO()):
            self.assertEqual(followup.main(args + ["--dry-run"]), 0)
            self.assertFalse(destination.exists())
            self.assertEqual(followup.main(args), 0)
            original = destination.read_bytes()
            self.assertEqual(runner.load_plan(destination)["gpu"], "1")
            self.assertEqual(followup.main(args), 2)
            self.assertEqual(destination.read_bytes(), original)
            launch.assert_not_called()

    def test_all_five_rounds_execute_with_fake_workers_and_resume(self):
        # Test orchestration with all 14 candidates; this never imports a model.
        (self.root / "train.py").write_text(
            FAKE_TRAINER.replace("assert conf['anchor_cl']['enabled'] is False", ""), encoding="utf-8")
        for name in ("run_ifashion_experiments.py", "experiment_io.py"):
            (self.root / name).write_bytes((ROOT / name).read_bytes())
        self.plan.update(round_id="stage3_fusion_local_06", common_overrides={})
        self.plan["experiments"] = [
            {"id": eid, "note": eid, "overrides": {"fusion_weights": {"modal_weight": weights}}}
            for eid, weights in (("ub_005", [.05, .35, .6]), ("ub_0075", [.075, .325, .6]),
                                 ("ub_0125", [.125, .275, .6]))
        ]
        self.save_plan()
        source = self.path
        total = 0
        with mock.patch.object(recheck, "PROJECT_ROOT", self.root), \
                mock.patch.object(recheck, "FUSION_PLAN", self.path), \
                mock.patch.object(followup, "PROJECT_ROOT", self.root):
            for index in range(5):
                if index == 1:
                    plan = recheck.build_plan("ub_0075", gpu="1")
                elif index >= 2:
                    stage, selected = (("ctemp", "clambda_030"), ("anchor_cl", "ctemp_030"),
                                       ("bi_noise", "anchor_0003"))[index - 2]
                    plan, _ = followup.build_plan(stage, source, selected, gpu="1")
                if index:
                    source = self.root / f"round_{index + 6}.yaml"
                    source.write_text(yaml.safe_dump(plan), encoding="utf-8")
                parsed = runner.load_plan(source, gpu="1")
                command = [sys.executable, str(self.root / "run_ifashion_experiments.py"),
                           "--plan", str(source), "--gpu", "1"]
                completed = subprocess.run(command, cwd=self.root, capture_output=True,
                                           encoding="utf-8", timeout=30)
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                state_path = self.root / "log/ifashion_tuning" / parsed["round_id"] / "state.json"
                state = json.loads(state_path.read_text(encoding="utf-8"))
                self.assertEqual(len(state["executions"]), len(parsed["experiments"]))
                for record, experiment in zip(state["executions"], parsed["experiments"]):
                    self.assertEqual(record["status"], "SUCCESS")
                    self.assertEqual(record["config"], experiment["config"])
                before_resume = state_path.read_bytes()
                resumed = subprocess.run(command, cwd=self.root, capture_output=True,
                                         encoding="utf-8", timeout=30)
                self.assertEqual(resumed.returncode, 0, resumed.stderr)
                self.assertEqual(state_path.read_bytes(), before_resume)
                total += len(parsed["experiments"])
        self.assertEqual(total, 14)
        last = state["executions"]
        self.assertTrue(all(r["config"]["anchor_cl"]["enabled"] for r in last))
        self.assertTrue(all(r["config"]["c_temps"] == [.3] for r in last))

    def install_automatic_pipeline(self):
        for name in ("run_ifashion_pipeline.py", "run_ifashion_experiments.py", "experiment_io.py",
                     "prepare_ifashion_clambda_recheck.py", "prepare_ifashion_followup.py"):
            (self.root / name).write_bytes((ROOT / name).read_bytes())
        plans = self.root / "tuning_plans"
        plans.mkdir()
        for name in ("ifashion_followup_stages_20261028.yaml", "ifashion_fusion_local_06.yaml"):
            (plans / name).write_bytes((ROOT / "tuning_plans" / name).read_bytes())
        snapshot = plans / "ifashion_local_baseline_20261028.yaml"
        snapshot.write_text(yaml.safe_dump(self.base), encoding="utf-8")
        first_path = plans / "ifashion_fusion_local_06.yaml"
        first = yaml.safe_load(first_path.read_text(encoding="utf-8"))
        first["result_file"] = "results/result.txt"
        first_path.write_text(yaml.safe_dump(first), encoding="utf-8")
        reference = self.root / "results/result.txt"
        reference.parent.mkdir()
        reference.write_text("EXECUTION_ID: 20261025T082203_9dec4c53\n完整实际配置:\n" +
                             yaml.safe_dump(self.base["iFashion"]) +
                             "\n尝试 1: SUCCESS\n验证选模分数: 0.4000000000\n", encoding="utf-8")
        scoring = '''
score=.1
score+=.01*(conf['fusion_weights']['modal_weight'][0]==.075)
score+=.01*(conf['c_lambdas'][0]==.3)
score+=.01*(conf['c_temps'][0]==.3)
score+=.01*(conf['anchor_cl']['enabled'] and conf['anchor_cl']['anchor_cl_lambda']==.003)
score+=.01*(conf['BI_ratios'][0]==.1)
metrics={s:{m:{str(k):(score if s=='val' else 1-score) for k in conf['topk']}
            for m in ['recall','ndcg']} for s in ['val','test']}
'''
        trainer = FAKE_TRAINER.replace("assert conf['anchor_cl']['enabled'] is False", "")
        trainer = trainer.replace("model=Path(a.result_json", scoring + "\nmodel=Path(a.result_json")
        trainer = trainer.replace("'selection_score':.4", "'selection_score':4*score")
        (self.root / "train.py").write_text(trainer, encoding="utf-8")
        return [sys.executable, str(self.root / "run_ifashion_pipeline.py"), "--gpu", "1"]

    def test_automatic_pipeline_selects_validation_winners_runs_all_and_resumes(self):
        command = self.install_automatic_pipeline()
        result = subprocess.run(command, cwd=self.root, capture_output=True, encoding="utf-8", timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        pipeline = self.root / "log/ifashion_tuning/pipeline_06_10"
        state = json.loads((pipeline / "state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["status"], "COMPLETE")
        self.assertEqual([s["selected_experiment"] for s in state["stages"]],
                         ["ub_0075", "clambda_030", "ctemp_030", "anchor_0003", "bi_noise_010"])
        self.assertEqual(sum(len(json.loads(p.read_text(encoding="utf-8"))["executions"])
                             for p in (self.root / "log/ifashion_tuning").glob("*/state.json")
                             if p.parent != pipeline), 14)
        best = yaml.safe_load((pipeline / "best_config.yaml").read_text(encoding="utf-8"))["iFashion"]
        self.assertEqual(best["fusion_weights"]["modal_weight"], [.075, .325, .6])
        self.assertEqual(best["c_lambdas"], [.3])
        self.assertEqual(best["c_temps"], [.3])
        self.assertEqual(best["BI_ratios"], [.1])
        self.assertEqual(best["anchor_cl"]["anchor_cl_lambda"], .003)
        counts = {p: p.read_bytes() for p in self.root.glob('counter_*.txt')}
        resumed = subprocess.run(command, cwd=self.root, capture_output=True, encoding="utf-8", timeout=60)
        self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
        self.assertEqual({p: p.read_bytes() for p in counts}, counts)
        changed_gpu = subprocess.run(command[:-1] + ["0"], cwd=self.root, capture_output=True,
                                     encoding="utf-8", timeout=30)
        self.assertEqual(changed_gpu.returncode, 2)
        self.assertIn("GPU changed", changed_gpu.stderr)

    def test_automatic_pipeline_dry_run_lists_all_without_writing(self):
        command = self.install_automatic_pipeline()
        result = subprocess.run(command + ["--dry-run"], cwd=self.root, capture_output=True,
                                encoding="utf-8", timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("共 14 组", result.stdout)
        self.assertIn("bi_noise_030", result.stdout)
        self.assertFalse((self.root / "log").exists())
        self.assertFalse(list(self.root.glob('counter_*.txt')))

    def test_automatic_pipeline_stops_on_failure_and_resumes_remaining_groups(self):
        command = self.install_automatic_pipeline()
        self.behavior({"0.35": "fail"})
        failed = subprocess.run(command, cwd=self.root, capture_output=True, encoding="utf-8", timeout=60)
        self.assertEqual(failed.returncode, 1, failed.stdout + failed.stderr)
        pipeline = self.root / "log/ifashion_tuning/pipeline_06_10"
        self.assertEqual(json.loads((pipeline / "state.json").read_text())["status"], "FAILED")
        self.assertFalse((pipeline / "round_7_plan.yaml").exists())
        self.behavior({})
        resumed = subprocess.run(command, cwd=self.root, capture_output=True, encoding="utf-8", timeout=60)
        self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
        self.assertEqual(json.loads((pipeline / "state.json").read_text())["status"], "COMPLETE")

    def test_automatic_pipeline_ties_keep_baseline_and_duplicate_launch_is_blocked(self):
        command = self.install_automatic_pipeline()
        trainer = self.root / "train.py"
        trainer.write_text(re.sub(r"^score\+=.*\n", "", trainer.read_text(encoding="utf-8"), flags=re.M),
                           encoding="utf-8")
        pipeline = self.root / "log/ifashion_tuning/pipeline_06_10"
        pipeline.mkdir(parents=True)
        with runner.round_lock(pipeline / "controller.lock"):
            blocked = subprocess.run(command, cwd=self.root, capture_output=True, encoding="utf-8", timeout=30)
        self.assertEqual(blocked.returncode, 2)
        self.assertFalse(list(self.root.glob('counter_*.txt')))
        result = subprocess.run(command, cwd=self.root, capture_output=True, encoding="utf-8", timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        state = json.loads((pipeline / "state.json").read_text(encoding="utf-8"))
        self.assertEqual([s["selected_experiment"] for s in state["stages"]], ["baseline"] * 5)
        self.assertEqual(state["best"]["execution_id"], "20261025T082203_9dec4c53")
        exported = yaml.safe_load((pipeline / "best_config.yaml").read_text(encoding="utf-8"))
        self.assertEqual(exported["iFashion"], self.base["iFashion"])

    def test_automatic_pipeline_gpu_migration_preserves_success_and_upgrades_old_state(self):
        command = self.install_automatic_pipeline()
        script = self.root / "run_ifashion_pipeline.py"
        current = script.read_bytes()
        script.write_bytes(current + b"\n# previous pipeline revision\n")
        # A prior committed pipeline revision models the existing server state.
        for args in (["git", "init", "-q"], ["git", "add", "run_ifashion_pipeline.py"],
                     ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                      "commit", "-q", "-m", "prior pipeline"]):
            result = subprocess.run(args, cwd=self.root, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        self.behavior({"0.3": "fail"})
        first = subprocess.run(command, cwd=self.root, capture_output=True, encoding="utf-8", timeout=60)
        self.assertEqual(first.returncode, 1, first.stdout + first.stderr)
        round7 = self.root / "log/ifashion_tuning/stage1_clambda_recheck_07/state.json"
        old_records = json.loads(round7.read_text(encoding="utf-8"))["executions"]
        self.assertEqual([r["status"] for r in old_records], ["SUCCESS", "FAILED", "SUCCESS"])
        script.write_bytes(current)
        self.behavior({})
        migrated = subprocess.run(command[:-1] + ["2", "--allow-gpu-change"], cwd=self.root,
                                  capture_output=True, encoding="utf-8", timeout=60)
        self.assertEqual(migrated.returncode, 0, migrated.stdout + migrated.stderr)
        records = json.loads(round7.read_text(encoding="utf-8"))["executions"]
        self.assertEqual(records[:3], old_records)
        self.assertEqual(len(records), 4)
        self.assertEqual(records[-1]["experiment_id"], "clambda_030")
        self.assertEqual(records[-1]["gpu"], "2")
        first_round = self.root / "log/ifashion_tuning/stage3_fusion_local_06/state.json"
        self.assertEqual(len(json.loads(first_round.read_text(encoding="utf-8"))["executions"]), 3)
        pipeline = self.root / "log/ifashion_tuning/pipeline_06_10/state.json"
        state = json.loads(pipeline.read_text(encoding="utf-8"))
        self.assertEqual(state["status"], "COMPLETE")
        self.assertEqual(state["gpu_history"], ["1", "2"])
        self.assertEqual(state["gpu_changes"][0]["from"], "1")
        resumed = subprocess.run(command[:-1] + ["2"], cwd=self.root, capture_output=True,
                                 encoding="utf-8", timeout=60)
        self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
        (self.root / "models/model.py").write_text("# changed training source\n", encoding="utf-8")
        rejected = subprocess.run(command[:-1] + ["2", "--allow-gpu-change"], cwd=self.root,
                                  capture_output=True, encoding="utf-8", timeout=30)
        self.assertEqual(rejected.returncode, 2)
        self.assertIn("inputs, code", rejected.stderr)


class ResultTests(unittest.TestCase):
    def test_stop_process_tolerates_unix_exit_races(self):
        for timeout in (False, True):
            with self.subTest(force_stop=timeout), mock.patch.object(runner, "os") as platform, \
                    mock.patch.object(runner, "signal") as signals:
                # Windows exposes neither killpg nor SIGKILL; simulate both Unix APIs.
                platform.name = "posix"
                signals.SIGTERM, signals.SIGKILL = 15, 9
                platform.killpg.side_effect = ProcessLookupError
                process = mock.Mock(pid=123)
                process.poll.return_value = None
                process.wait.side_effect = [subprocess.TimeoutExpired("fake", 10), 0] if timeout else [0]
                runner.stop_process(process)
                self.assertEqual(platform.killpg.call_count, 2 if timeout else 1)
                self.assertEqual(process.wait.call_count, 2 if timeout else 1)

    def test_elapsed_time_uses_monotonic_clock_and_reads_legacy_timestamps(self):
        record = {"started_at": "2026-10-24T12:00:00+08:00"}
        with mock.patch.object(runner, "time") as clock, mock.patch.object(
                runner, "timestamp", return_value="2026-10-24T11:59:00+08:00"):
            clock.monotonic.return_value = 10000.125
            runner.finish_timing(record, 6000)
        self.assertEqual(record["duration_seconds"], 4000.125)
        self.assertEqual(runner.elapsed_text(record), "01:06:40.125（共 4000.125 秒）")
        legacy = {"started_at": "2026-10-24T23:30:00+08:00", "ended_at": "2026-10-25T00:31:01+08:00"}
        self.assertEqual(runner.elapsed_text(legacy), "01:01:01.000（共 3661.000 秒）（根据开始和结束时间估算）")
        self.assertEqual(runner.elapsed_text(dict(legacy, timing_note="Recovered after hard stop")), "未知")

    def test_export_requires_best_and_excludes_device(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.json"
            metrics = {s: {m: {20: .1, 40: .2} for m in ("recall", "ndcg")} for s in ("val", "test")}
            save_best_result(path, {"device": object(), "seed": 2026}, metrics, 4, {})
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertNotIn("device", payload["config"])
            self.assertEqual(payload["best_epoch_1based"], 5)
            self.assertAlmostEqual(payload["selection_score"], .6)
            with self.assertRaises(ValueError):
                save_best_result(path, {}, {s: {m: {20: 0, 40: 0} for m in ("recall", "ndcg")} for s in ("val", "test")}, 0, {})

    def test_atomic_write_failure_preserves_existing_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            atomic_write_json(path, {"value": 1})
            with self.assertRaises(ValueError):
                atomic_write_json(path, {"value": float("nan")})
            self.assertEqual(json.loads(path.read_text()), {"value": 1})

    def test_committed_coarse_plan_has_exactly_five_candidates(self):
        with mock.patch.object(runner, "PROJECT_ROOT", ROOT):
            plan = runner.load_plan(ROOT / "tuning_plans/ifashion_clambda_coarse_01.yaml")
        self.assertEqual([e["config"]["c_lambdas"][0] for e in plan["experiments"]], [.05, .1, .2, .5, 1.0])
        for e in plan["experiments"]:
            self.assertFalse(e["config"]["anchor_cl"]["enabled"])
            self.assertEqual(e["config"]["c_temps"], [.2])
            self.assertEqual(e["config"]["seed"], 2026)


if __name__ == "__main__":
    unittest.main()
