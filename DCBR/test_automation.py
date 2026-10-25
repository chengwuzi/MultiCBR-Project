"""Search orchestration tests. All artifacts/results stay in temporary folders."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

import test_reproduction as reference

HERE = Path(__file__).resolve().parent
runner = reference.load_module("dcbr_automation_test", HERE / "run_experiments.py")


class AutomationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reference.ReproductionTests.setUpClass()

    def make_plan(self, root, two=False):
        fixture = reference.ReproductionTests()
        fixture.base = reference.ReproductionTests.base
        config_path = fixture.make_tiny_data(root)
        # Use the new dataset name, preserving zero-UB users in integration.
        directory = root / "data/iFashion"
        directory.rename(root / "data/Youshu")
        directory = root / "data/Youshu"
        (directory / "iFashion_data_size.txt").rename(directory / "Youshu_data_size.txt")
        (directory / "Youshu_data_size.txt").write_text("6\t100\t8\n")
        conf = yaml.safe_load(config_path.read_text())["iFashion"]
        config_path.write_text(yaml.safe_dump({"Youshu": conf}), encoding="utf-8")
        experiments = [{"id": "initial", "note": "tiny baseline", "overrides": {}}]
        if two:
            experiments.append({"id": "second", "note": "tiny variation", "overrides": {"tau": .2}})
        document = {"version": 1, "base_config": str(config_path), "work_dir": str(root / "work"),
                    "result_file": str(root / "search.txt"),
                    "defaults": {"gpu": "0", "device": "cpu", "max_attempts": 2, "retry_delay_seconds": 0,
                                 "console_interval_seconds": 1, "stream_output": False},
                    "rounds": [{"id": "round_01", "dataset": "Youshu", "note": "test round",
                                "overrides": {}, "experiments": experiments},
                               {"id": "later_round", "dataset": "Youshu", "note": "must not run",
                                "experiments": [{"id": "later", "note": "later experiment", "overrides": {}}]}]}
        path = root / "plan.yaml"
        path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        return path

    def test_youshu_initial_configuration_is_mealrec_l_with_user_budget(self):
        overall = yaml.safe_load((HERE / "raw/configs/overall.yaml").read_text())
        author = yaml.safe_load((HERE / "raw/configs/models/DCBR.yaml").read_text())["MealRec+L"]
        expected = dict(overall, **author, seed=0)
        expected["topk"] = [10, 20, 40, 80]
        expected["epochs"] = 90  # User-specified Youshu budget, not the author's MealRec+L budget.
        actual = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))["Youshu"]
        self.assertEqual(actual, expected)

    def test_plan_dry_run_schema_protocol_and_fingerprints(self):
        with tempfile.TemporaryDirectory(prefix="dcbr-plan-") as temp:
            root = Path(temp)
            path = self.make_plan(root, two=True)
            original = path.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.run(path, "round_01", dry_run=True), 0)
            self.assertEqual(path.read_bytes(), original)
            self.assertFalse((root / "work").exists())
            plan = runner.load_plan(path)
            self.assertEqual(len(plan["rounds"]), 2)
            first = plan["rounds"][0]["experiments"][0]["fingerprint"]
            self.assertNotEqual(first, runner.load_plan(path, gpu="1")["rounds"][0]["experiments"][0]["fingerprint"])
            data = root / "data/Youshu/user_bundle_test.txt"
            data.write_bytes(data.read_bytes() + b"\n")
            self.assertNotEqual(first, runner.load_plan(path)["rounds"][0]["experiments"][0]["fingerprint"])
            for edit in (lambda p: p["rounds"][0]["experiments"][0].update(overrides={"typo": 1}),
                         lambda p: p["rounds"][0]["experiments"][0].update(overrides={"seed": 9}),
                         lambda p: p["rounds"][0].update(overrides={"topk": [20]}),
                         lambda p: p["rounds"].append(p["rounds"][0])):
                document = yaml.safe_load(original)
                edit(document)
                path.write_text(yaml.safe_dump(document), encoding="utf-8")
                with self.assertRaises(ValueError):
                    runner.load_plan(path)
            path.write_text("version: 1\nversion: 1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate YAML"):
                runner.load_plan(path)

    def test_full_serial_run_skip_rerun_invalidation_and_no_raw_changes(self):
        with tempfile.TemporaryDirectory(prefix="dcbr-search-") as temp:
            root = Path(temp)
            path = self.make_plan(root, two=True)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(runner.run(path, "round_01"), 0)
            state_path = root / "work/history.json"
            state = runner.load_state(state_path)
            self.assertEqual(len(state["executions"]), 2)
            self.assertTrue(all(r["status"] == "SUCCESS" for r in state["executions"]))
            self.assertTrue(all(len(r["attempts"]) == 1 for r in state["executions"]))
            self.assertLessEqual(state["executions"][0]["ended_at"], state["executions"][1]["started_at"])
            self.assertIn("最近输出", stdout.getvalue())
            document = runner.read_yaml(path)
            self.assertTrue(document["rounds"][0]["experiments"][0]["success"])
            self.assertEqual(document["rounds"][1]["experiments"][0]["status"], "PENDING")
            trained = state["executions"][0]["training_record"]
            self.assertEqual(trained["config"]["num_users"], 6)
            self.assertEqual(trained["experiment_case"], "second")
            self.assertEqual(trained["result"]["selection_rule"], runner.RULE)
            plan = runner.load_plan(path)
            round_ = plan["rounds"][0]
            item = round_["experiments"][0]
            attempt = state["executions"][0]["attempts"][0]
            for mutate in (lambda r: r.update(completed_epochs=1),
                           lambda r: r.update(source_hash="wrong"),
                           lambda r: r["result"]["selection_metrics"].update({"recall@20": 0}),
                           lambda r: r["result"]["metrics"]["test"]["ndcg"].update({"20": float("inf")}),
                           lambda r: r["config"].update(info="wrong")):
                corrupted = copy.deepcopy(trained)
                mutate(corrupted)
                bad_result = root / "tampered.json"
                bad_result.write_text(json.dumps(corrupted), encoding="utf-8")
                with self.assertRaises(ValueError):
                    runner.validate_result(bad_result, item["config"], "Youshu", plan["sources"],
                                           round_["data_files"], attempt["info"], "0", "cpu")
            report = (root / "search.txt").read_text(encoding="utf-8")
            self.assertEqual(report.count("END_EXECUTION:"), 2)
            self.assertIn("第二种情况", report)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.run(path, "round_01"), 0)
            self.assertEqual(len(runner.load_state(state_path)["executions"]), 2)
            self.assertEqual((root / "search.txt").read_text(encoding="utf-8"), report)
            # Deliberate rerun preserves previous artifacts and execution IDs.
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.run(path, "round_01", mode="selected", only=["initial"]), 0)
            self.assertEqual(len(runner.load_state(state_path)["executions"]), 3)
            # A damaged checkpoint is not trusted as a historical success.
            state = runner.load_state(state_path)
            checkpoint = Path(state["executions"][-1]["training_record"]["artifacts"]["checkpoint_model"])
            with checkpoint.open("ab") as stream:
                stream.write(b"corruption")
            plan = runner.load_plan(path)
            selected = runner.select_experiments(plan, plan["rounds"][0], state, "resume", [])
            self.assertEqual([e["id"] for e in selected], ["initial"])
            runner.sync_plan(plan, state)
            self.assertEqual(runner.read_yaml(path)["rounds"][0]["experiments"][0]["status"], "INVALIDATED")
            # Source changes invalidate even identical IDs and parameters.
            with patch.object(runner, "source_fingerprints", return_value={"train.py": "new", "run_experiments.py": "new"}):
                changed = runner.load_plan(path)
            self.assertEqual(len(runner.select_experiments(changed, changed["rounds"][0], state, "resume", [])), 2)

    def test_retry_failure_and_failure_mode(self):
        with tempfile.TemporaryDirectory(prefix="dcbr-retry-") as temp:
            root = Path(temp)
            path = self.make_plan(root)
            calls = []
            real = runner.run_worker
            def fail_once(*args, **kwargs):
                calls.append(1)
                if len(calls) == 1:
                    return 7
                return real(*args, **kwargs)
            with patch.object(runner, "run_worker", side_effect=fail_once), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.run(path, "round_01"), 0)
            state = runner.load_state(root / "work/history.json")
            self.assertEqual([a["status"] for a in state["executions"][0]["attempts"]], ["FAILED", "SUCCESS"])
            with patch.object(runner, "run_worker", return_value=8), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.run(path, "round_01", mode="all"), 1)
            state = runner.load_state(root / "work/history.json")
            self.assertEqual(state["executions"][-1]["status"], "FAILED")
            self.assertEqual(len(state["executions"][-1]["attempts"]), 2)
            plan = runner.load_plan(path)
            self.assertEqual(len(runner.select_experiments(plan, plan["rounds"][0], state, "failed", [])), 1)
            with self.assertRaises(ValueError):
                runner.select_experiments(plan, plan["rounds"][0], state, "selected", ["missing"])

    def test_interrupt_recovery_lock_and_report_idempotency(self):
        with tempfile.TemporaryDirectory(prefix="dcbr-recover-") as temp:
            root = Path(temp)
            path = self.make_plan(root)
            with patch.object(runner, "run_worker", side_effect=KeyboardInterrupt), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(KeyboardInterrupt):
                    runner.run(path, "round_01")
            history = root / "work/history.json"
            state = runner.load_state(history)
            record = state["executions"][0]
            self.assertEqual(record["status"], "INTERRUPTED")
            self.assertEqual(record["attempts"][0]["status"], "INTERRUPTED")
            report = (root / "search.txt").read_bytes()
            runner.append_report(root / "search.txt", record)
            self.assertEqual((root / "search.txt").read_bytes(), report)
            # A living orphan blocks recovery without killing someone else's process.
            record["status"] = "RUNNING"
            record["attempts"][-1].update(status="RUNNING", pid=os.getpid())
            with self.assertRaisesRegex(RuntimeError, "still be alive"):
                runner.recover(state, history)
            with patch.object(runner, "process_alive", return_value=False):
                runner.recover(state, history)
            self.assertIsNone(record["duration_seconds"])
            self.assertIn("恢复检测时间", record["timing_note"])
            # Lock acquisition across processes must fail while held.
            lock = root / "exclusive.lock"
            script = "import sys; sys.path.insert(0,sys.argv[1]); import run_experiments as r;\nwith r.file_lock(sys.argv[2]): pass"
            with runner.file_lock(lock):
                result = subprocess.run([sys.executable, "-c", script, str(HERE), str(lock)], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
            # A human change is preserved rather than overwritten by status sync.
            plan = runner.load_plan(path)
            changed = runner.read_yaml(path)
            changed["rounds"][0]["note"] = "edited while running"
            path.write_text(yaml.safe_dump(changed), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "edited during"):
                runner.sync_plan(plan, state)

    def test_interrupt_stops_the_actual_owned_worker_and_keeps_output(self):
        with tempfile.TemporaryDirectory(prefix="dcbr-child-stop-") as temp:
            root = Path(temp)
            attempt = {"stdout": str(root / "stdout.txt"), "gate": str(root / "go.gate")}
            state = runner.empty_state()
            record = {"config": {"seed": 0}, "gpu": "0"}
            settings = {"stream_output": False, "console_interval_seconds": 1}
            command = [sys.executable, "-u", "-c", "import time; print('worker started', flush=True); time.sleep(60)"]
            def interrupt(detail):
                raise KeyboardInterrupt
            with self.assertRaises(KeyboardInterrupt):
                runner.run_worker(command, attempt, record, state, root / "history.json", settings, interrupt)
            self.assertFalse(runner.process_alive(attempt["pid"]))
            self.assertIn("worker started", (root / "stdout.txt").read_text())

    def make_dependent_plan(self, root):
        path = self.make_plan(root)
        document = runner.read_yaml(path)
        reviewed = runner.read_yaml(HERE / "搜参配置.yaml")["rounds"][0]
        document["rounds"] = [copy.deepcopy(reviewed)]
        document["rounds"][0]["id"] = "round_01"
        # Keep real tiny model/data, with just two test epochs; production is 90.
        document["rounds"][0]["overrides"].update(epochs=2)
        path.write_text(yaml.safe_dump(document, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return path

    def test_reviewed_first_round_has_exactly_nine_experiments(self):
        plan = runner.load_plan(HERE / "搜参配置.yaml")
        self.assertEqual(len(plan["rounds"]), 1)
        round_ = plan["rounds"][0]
        items = round_["experiments"]
        self.assertEqual(len(items), 9)
        self.assertTrue(all(item["config"]["epochs"] == 90 for item in items))
        self.assertEqual([e["config"]["lambda_1"] for e in items[:5]], [.01, .002, .005, .02, .05])
        self.assertTrue(all(e["config"]["lambda_0"] == 1000 for e in items[:5]))
        self.assertEqual([e["overrides"]["lambda_0"] for e in items[5:]], [100, 500, 2000, 10000])
        expected_parents = [e["id"] for e in items[:5]]
        self.assertTrue(all(e["inherit_best"]["candidates"] == expected_parents for e in items[5:]))
        runner.resolve_dependencies(plan, round_, runner.empty_state())
        self.assertTrue(all(e["config"] is None for e in items[5:]))

    def test_dependency_selection_uses_validation_and_provenance(self):
        with tempfile.TemporaryDirectory(prefix="dcbr-dependency-") as temp:
            path = self.make_dependent_plan(Path(temp))
            plan = runner.load_plan(path)
            round_ = plan["rounds"][0]
            items = round_["experiments"]
            values = [(.1, .9), (.2, .1), (.2, .2), (.2, .2), (.19, .99)]
            state = runner.empty_state()
            for index, (item, (recall, ndcg)) in enumerate(zip(items[:5], values)):
                state["executions"].append({"round_id": "round_01", "experiment_id": item["id"],
                    "execution_id": f"parent_{index}", "fingerprint": item["fingerprint"], "status": "SUCCESS",
                    "training_record": {"result": {"metrics": {"val": {"recall": {"20": recall},
                                                                               "ndcg": {"20": ndcg}}}}}})
            valid = lambda record, *args: record is not None and record["status"] == "SUCCESS"
            with patch.object(runner, "check_success", side_effect=valid):
                runner.resolve_dependencies(plan, round_, state)
                self.assertTrue(all(e["config"]["lambda_1"] == .005 for e in items[5:]))
                self.assertTrue(all(e["dependency_selection"]["winner_experiment_id"] == "lambda1_0005" for e in items[5:]))
                first = items[5]["fingerprint"]
                # Even a non-winning parent's rerun is part of the selection provenance.
                state["executions"][0]["execution_id"] = "parent_0_rerun"
                runner.resolve_dependencies(plan, round_, state)
                self.assertNotEqual(first, items[5]["fingerprint"])
                # One missing/failed parent blocks all dependent configurations.
                state["executions"][0]["status"] = "FAILED"
                runner.resolve_dependencies(plan, round_, state)
                self.assertTrue(all(e["config"] is None and e["blocked_by"] == ["initial"] for e in items[5:]))
            document = runner.read_yaml(path)
            # Forward/self references, duplicates and test-based rules are rejected.
            for inherit in ({"candidates": ["lambda0_0100"], "rule": runner.CROSS_RULE},
                            {"candidates": ["initial", "initial"], "rule": runner.CROSS_RULE},
                            {"candidates": ["initial"], "rule": "test_recall20"}):
                bad = copy.deepcopy(document)
                bad["rounds"][0]["experiments"][5]["inherit_best"] = inherit
                path.write_text(yaml.safe_dump(bad), encoding="utf-8")
                with self.assertRaises(ValueError):
                    runner.load_plan(path)

    def test_nine_experiment_chain_interrupt_and_resume(self):
        with tempfile.TemporaryDirectory(prefix="dcbr-nine-") as temp:
            root = Path(temp)
            path = self.make_dependent_plan(root)
            document = runner.read_yaml(path)
            parent_ids = [e["id"] for e in document["rounds"][0]["experiments"][:5]]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.run(path, "round_01", mode="selected", only=parent_ids), 0)
            history = root / "work/history.json"
            parents = runner.load_state(history)["executions"]
            self.assertEqual(len(parents), 5)
            winner = max(parents, key=lambda r: runner.validation_key(r["training_record"]))
            selected_lambda1 = winner["config"]["lambda_1"]
            parent_execution_ids = [r["execution_id"] for r in parents]
            actual = runner.run_worker
            def interrupt_second_child(command, attempt, record, *args, **kwargs):
                if record["experiment_id"] == "lambda0_0500":
                    raise KeyboardInterrupt
                return actual(command, attempt, record, *args, **kwargs)
            with patch.object(runner, "run_worker", side_effect=interrupt_second_child), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(KeyboardInterrupt):
                    runner.run(path, "round_01")
            state = runner.load_state(history)
            self.assertEqual([r["status"] for r in state["executions"]][-2:], ["SUCCESS", "INTERRUPTED"])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.run(path, "round_01"), 0)
            state = runner.load_state(history)
            self.assertEqual(len(state["executions"]), 10)  # Nine successes plus one interrupted attempt/execution.
            self.assertEqual([r["execution_id"] for r in state["executions"][:5]], parent_execution_ids)
            children = [r for r in state["executions"] if r["status"] == "SUCCESS" and r["experiment_id"].startswith("lambda0_")]
            self.assertEqual(len(children), 4)
            for child in children:
                self.assertEqual(child["config"]["lambda_1"], selected_lambda1)
                self.assertEqual(child["dependency_selection"]["winner_execution_id"], winner["execution_id"])
                self.assertEqual(child["training_record"]["config"]["lambda_1"], selected_lambda1)
            report = (root / "search.txt").read_text(encoding="utf-8")
            self.assertEqual(report.count("END_EXECUTION:"), 10)
            self.assertIn("前置实验选择依据", report)
            summary = json.loads((root / "work/Youshu/round_01/round_summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["valid_successes"], 9)
            final = max([r for r in state["executions"] if r["status"] == "SUCCESS"],
                        key=lambda r: runner.validation_key(r["training_record"]))
            self.assertEqual(summary["best"]["execution_id"], final["execution_id"])
            # A completed resume appends neither executions nor duplicate summaries.
            before = (root / "search.txt").read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.run(path, "round_01"), 0)
            self.assertEqual((root / "search.txt").read_bytes(), before)
            self.assertEqual(len(runner.load_state(history)["executions"]), 10)

    def test_failed_parent_never_launches_dependents(self):
        with tempfile.TemporaryDirectory(prefix="dcbr-blocked-") as temp:
            root = Path(temp)
            path = self.make_dependent_plan(root)
            with patch.object(runner, "preflight"), patch.object(runner, "run_worker", return_value=8), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.run(path, "round_01"), 1)
            state = runner.load_state(root / "work/history.json")
            self.assertEqual(len(state["executions"]), 5)
            self.assertTrue(all(r["status"] == "FAILED" for r in state["executions"]))
            items = runner.read_yaml(path)["rounds"][0]["experiments"]
            self.assertTrue(all(e["status"] == "BLOCKED" and not e["success"] for e in items[5:]))
            summary = json.loads((root / "work/Youshu/round_01/round_summary.json").read_text())
            self.assertFalse(summary["complete"])
            self.assertIsNone(summary["best"])

    def test_worker_interrupt_exit_stops_without_retry(self):
        with tempfile.TemporaryDirectory(prefix="dcbr-exit130-") as temp:
            root = Path(temp)
            path = self.make_plan(root, two=True)
            with patch.object(runner, "preflight"), patch.object(runner, "run_worker", return_value=130) as worker:
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(KeyboardInterrupt):
                    runner.run(path, "round_01")
            worker.assert_called_once()
            records = runner.load_state(root / "work/history.json")["executions"]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["status"], "INTERRUPTED")
            self.assertEqual(records[0]["attempts"][0]["status"], "INTERRUPTED")
            self.assertEqual(records[0]["attempts"][0]["return_code"], 130)

    def test_pending_dependency_checks_known_overrides_before_training(self):
        with tempfile.TemporaryDirectory(prefix="dcbr-pending-validation-") as temp:
            root = Path(temp)
            path = self.make_dependent_plan(root)
            document = runner.read_yaml(path)
            document["rounds"][0]["experiments"][5]["overrides"]["lambda_0"] = -1
            path.write_text(yaml.safe_dump(document), encoding="utf-8")
            plan = runner.load_plan(path)
            round_ = plan["rounds"][0]
            runner.resolve_dependencies(plan, round_, runner.empty_state())
            child = round_["experiments"][5]
            self.assertIsNone(child["config"])
            with self.assertRaisesRegex(ValueError, "Invalid experiment"):
                runner.preflight(plan, round_, [child])
            self.assertIsNone(child["config"])
            self.assertIsNone(child["fingerprint"])


if __name__ == "__main__":
    unittest.main()
