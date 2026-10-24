"""Controller integration tests use a fake trainer, never a recommendation model."""

from contextlib import redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import subprocess
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
        with redirect_stdout(io.StringIO()):
            return runner.run(self.path, **kwargs)

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

    def test_retry_success_on_third_attempt(self):
        self.behavior({"0.05": "twice"})
        self.assertEqual(self.run_plan(), 0)
        record = self.state()["executions"][0]
        self.assertEqual([a["status"] for a in record["attempts"]], ["FAILED", "FAILED", "SUCCESS"])
        self.assertEqual(len({a["config_snapshot"] for a in record["attempts"]}), 3)

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
        self.assertEqual([r["experiment_id"] for r in self.state()["executions"]], ["a", "b", "b"])
        self.run_plan(mode="all")
        self.assertEqual(len(self.state()["executions"]), 5)
        with self.assertRaises(ValueError):
            self.run_plan(mode="selected", only=["unknown"])

    def test_missing_nonfinite_and_mismatched_results_retry(self):
        for action in ("missing", "nan", "wrong"):
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
        self.assertFalse(runner.process_alive(triggered[0]))
        self.assertEqual(self.run_plan(), 0)
        self.assertEqual([r["status"] for r in self.state()["executions"]], ["INTERRUPTED", "SUCCESS", "SUCCESS"])

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
        atomic_write_json(path, state)
        self.run_plan()
        self.assertEqual(self.state()["executions"][0]["status"], "INTERRUPTED")

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
