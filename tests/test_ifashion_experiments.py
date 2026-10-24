"""Controller integration tests use a fake trainer, never a recommendation model."""

from contextlib import redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import yaml

import run_ifashion_experiments as runner
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
