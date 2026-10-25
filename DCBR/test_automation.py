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

    def test_youshu_configuration_is_mealrec_l_with_selected_baseline(self):
        overall = yaml.safe_load((HERE / "raw/configs/overall.yaml").read_text())
        author = yaml.safe_load((HERE / "raw/configs/models/DCBR.yaml").read_text())["MealRec+L"]
        expected = dict(overall, **author, seed=0)
        expected["topk"] = [10, 20, 40, 80]
        expected["epochs"] = 90  # User-specified Youshu budget, not the author's MealRec+L budget.
        expected.update(lambda_1=0.02, lambda_0=2000.0, omega=0.3, tau=0.25)  # User-selected round-02 configuration.
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
            self.assertNotIn("status", document["rounds"][1]["experiments"][0])  # Unselected rounds remain untouched.
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
        self.assertEqual([r["id"] for r in plan["rounds"][:3]],
                         ["youshu_round_01", "youshu_round_02", "netease_round_01"])
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

    def test_reviewed_second_round_has_thirteen_trials_and_one_pinned_reference(self):
        plan = runner.load_plan(HERE / "搜参配置.yaml")
        round_ = plan["rounds"][1]
        items = round_["experiments"]
        self.assertEqual(round_["id"], "youshu_round_02")
        self.assertEqual(round_["selection_rule"], runner.TEST_RULE)
        self.assertEqual(len(items), 14)
        self.assertEqual(items[0]["reference"]["execution_id"], "20261025T093608_5009e405")
        historical = next(i for i in plan['document']['rounds'][0]['experiments'] if i['id'] == 'lambda0_2000')
        comparable = dict(items[0]['template_config'], data_path=historical['resolved_config']['data_path'])
        self.assertEqual(comparable, historical['resolved_config'])
        historical_sources = {'train.py': 'efc0c8a83909e2ba8aadc6c06fa3a55b5f81cc2244593eef54266ddf83237379',
                              'run_experiments.py': '15fb8ceffa34ccf990e733825e3b9501b29e3e1a6c14dab3620ba7c3e841421c'}
        self.assertEqual(runner.experiment_fingerprint(historical['resolved_config'], 'Youshu', historical_sources,
                         round_['data_files'], {'gpu': '0', 'device': 'cuda'}, historical['dependency_selection']),
                         items[0]['reference']['fingerprint'])
        self.assertEqual([i["config"]["omega"] for i in items[1:6]], [.1, .3, .5, .7, 1.])
        self.assertEqual([i["overrides"]["tau"] for i in items[6:10]], [.1, .15, .25, .3])
        self.assertEqual([(i["overrides"]["gamma_1"], i["overrides"]["gamma_2"]) for i in items[10:]],
                         [(.1, .4), (.2, .3), (.25, .25), (.4, .1)])
        self.assertTrue(all(i["config"]["epochs"] == 90 and i["config"]["seed"] == 0
                            and i["config"]["lambda_1"] == .02 and i["config"]["lambda_0"] == 2000 for i in items))
        self.assertTrue(all(i["inherit_best"]["candidates"] == [p["id"] for p in items[:6]] for i in items[6:10]))
        self.assertTrue(all(i["inherit_best"]["candidates"] == [p["id"] for p in items[:10]] for i in items[10:]))
        selected = runner.select_experiments(plan, round_, runner.empty_state(), "all", [])
        self.assertEqual(len(selected), 13)
        self.assertNotIn(items[0], selected)
        self.assertEqual([s["id"] for s in round_["stages"]], ["omega", "tau", "gamma"])

    def test_test_recall_rule_ignores_ndcg_validation_and_display_rounding(self):
        def training(recall, ndcg, validation):
            return {"result": {"metrics": {"test": {"recall": {"20": recall}, "ndcg": {"20": ndcg}},
                                            "val": {"recall": {"20": validation}, "ndcg": {"20": validation}}}}}
        control = training(.255661201, .01, .01)
        tie = training(.255661201, .99, .99)
        better = training(.255661202, .001, .001)
        self.assertEqual(runner.configuration_key(control, runner.TEST_RULE), runner.configuration_key(tie, runner.TEST_RULE))
        self.assertIs(max([control, tie], key=lambda r: runner.configuration_key(r, runner.TEST_RULE)), control)
        self.assertGreater(runner.configuration_key(better, runner.TEST_RULE), runner.configuration_key(control, runner.TEST_RULE))
        self.assertLess(runner.configuration_key(better, runner.CROSS_RULE), runner.configuration_key(tie, runner.CROSS_RULE))
        for value in (float('nan'), float('inf'), -.1, 1.1, True):
            with self.assertRaises(ValueError):
                runner.configuration_key(training(value, .1, .1), runner.TEST_RULE)

    def test_sequential_test_choices_propagate_and_keep_existing_controls(self):
        with tempfile.TemporaryDirectory(prefix='dcbr-test-choice-') as temp:
            document = runner.read_yaml(HERE / '搜参配置.yaml')
            # This test isolates ordering from artifact checks, covered by the
            # real reference/integration tests below.
            document['rounds'][1]['experiments'][0].pop('reference')
            path = Path(temp) / 'plan.yaml'
            path.write_text(yaml.safe_dump(document, allow_unicode=True), encoding='utf-8')
            plan = runner.load_plan(path)
            round_ = plan['rounds'][1]
            items = round_['experiments']
            state = runner.empty_state()
            def add(item, recall, ndcg):
                state['executions'].append(dict(round_id=round_['id'], experiment_id=item['id'],
                    execution_id='execution_' + item['id'], fingerprint=item['fingerprint'], status='SUCCESS',
                    training_record={'result': {'metrics': {
                        'test': {'recall': {'20': recall}, 'ndcg': {'20': ndcg}},
                        'val': {'recall': {'20': ndcg}, 'ndcg': {'20': ndcg}}}}}))
            valid = lambda record, *args: record is not None and record['status'] == 'SUCCESS'
            for item, recall, ndcg in zip(items[:6], [.5, .6, .6, .4, .3, .2], [.9, .01, .99, .9, .9, .9]):
                add(item, recall, ndcg)
            with patch.object(runner, 'check_success', side_effect=valid):
                runner.resolve_dependencies(plan, round_, state)
                self.assertTrue(all(i['config']['omega'] == .1 for i in items[6:10]))
                self.assertTrue(all(i['dependency_selection']['winner_experiment_id'] == 'omega_01' for i in items[6:10]))
                # All four tau trials tie the omega winner in Recall, despite
                # higher NDCG: the existing tau=.2 control must be retained.
                for item in items[6:10]:
                    add(item, .6, .99)
                runner.resolve_dependencies(plan, round_, state)
                self.assertTrue(all(i['config']['omega'] == .1 and i['config']['tau'] == .2 for i in items[10:]))
                old = items[10]['fingerprint']
                # Now tau=.15 wins Recall independently of its lower NDCG.
                state['executions'][7]['training_record']['result']['metrics']['test']['recall']['20'] = .7
                state['executions'][7]['training_record']['result']['metrics']['test']['ndcg']['20'] = .001
                runner.resolve_dependencies(plan, round_, state)
                self.assertTrue(all(i['config']['tau'] == .15 for i in items[10:]))
                self.assertNotEqual(old, items[10]['fingerprint'])

    def make_second_round_plan(self, root):
        path = self.make_plan(root)
        document = runner.read_yaml(path)
        reviewed = copy.deepcopy(runner.read_yaml(HERE / "搜参配置.yaml")["rounds"][1])
        reviewed["id"] = "round_02"
        reviewed["overrides"]["epochs"] = 2
        document["rounds"] = document["rounds"][:1]
        document["rounds"][0]["overrides"] = copy.deepcopy(reviewed["overrides"])
        path.write_text(yaml.safe_dump(document, allow_unicode=True, sort_keys=False), encoding='utf-8')
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.run(path, 'round_01'), 0)
        history = runner.load_state(root / 'work/history.json')
        original = copy.deepcopy(history['executions'][0])
        reviewed['experiments'][0]['reference'] = {key: original[key] for key in
                                                  ('round_id', 'experiment_id', 'execution_id', 'fingerprint')}
        document = runner.read_yaml(path)
        document['rounds'].append(reviewed)
        path.write_text(yaml.safe_dump(document, allow_unicode=True, sort_keys=False), encoding='utf-8')
        return path, original

    def test_historical_reference_checks_and_rejects_training_or_missing_artifacts(self):
        with tempfile.TemporaryDirectory(prefix='dcbr-reference-') as temp:
            root = Path(temp)
            path, original = self.make_second_round_plan(root)
            plan = runner.load_plan(path)
            round_ = plan['rounds'][1]
            item = round_['experiments'][0]
            state = runner.load_state(root / 'work/history.json')
            runner.resolve_dependencies(plan, round_, state)
            self.assertFalse(item['blocked_by'])
            self.assertTrue(runner.check_success(original, item, round_, plan))
            # Updating only orchestration code must not require another model run.
            plan['sources']['run_experiments.py'] = 'f' * 64
            self.assertTrue(runner.check_reference(original, item, round_, plan))
            for mutate in (lambda p: p['sources'].update({'train.py': 'e' * 64}),
                           lambda p: p['settings'].update(gpu='1'),
                           lambda p: p['settings'].update(device='cuda')):
                bad = copy.deepcopy(plan)
                mutate(bad)
                self.assertFalse(runner.check_reference(original, item, round_, bad))
            incompatible = copy.deepcopy(item)
            incompatible['template_config']['tau'] = .6
            self.assertFalse(runner.check_reference(original, incompatible, round_, plan))
            with self.assertRaises(ValueError):
                runner.select_experiments(plan, round_, state, 'selected', [item['id']])
            # Corrupting a real artifact blocks the entire new round before any training.
            checkpoint = Path(original['training_record']['artifacts']['checkpoint_model'])
            checkpoint.write_bytes(checkpoint.read_bytes() + b'broken')
            self.assertFalse(runner.check_reference(original, item, round_, plan))
            with patch.object(runner, 'run_worker') as worker, contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, 'Historical references missing or invalid'):
                    runner.run(path, 'round_02')
            worker.assert_not_called()
            self.assertEqual(len(runner.load_state(root / 'work/history.json')['executions']), 1)

    def test_missing_reference_dry_run_has_no_writes_and_actual_run_stops(self):
        with tempfile.TemporaryDirectory(prefix='dcbr-missing-reference-') as temp:
            root = Path(temp)
            path, _ = self.make_second_round_plan(root)
            history = root / 'work/history.json'
            history.write_text(json.dumps(runner.empty_state()), encoding='utf-8')
            before = path.read_bytes()
            report = (root / 'search.txt').read_bytes()
            output = io.StringIO()
            with patch.object(runner, 'preflight'), contextlib.redirect_stdout(output):
                self.assertEqual(runner.run(path, 'round_02', dry_run=True), 0)
            self.assertIn('BLOCKED', output.getvalue())
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual((root / 'search.txt').read_bytes(), report)
            with patch.object(runner, 'run_worker') as worker, contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, 'Historical references missing or invalid'):
                    runner.run(path, 'round_02')
            worker.assert_not_called()

    def test_thirteen_trial_chain_reuses_control_inherits_and_resumes(self):
        with tempfile.TemporaryDirectory(prefix='dcbr-thirteen-') as temp:
            root = Path(temp)
            path, original = self.make_second_round_plan(root)
            first_round = copy.deepcopy(runner.read_yaml(path)['rounds'][0])
            real = runner.run_worker
            def interrupt_tau(command, attempt, record, *args, **kwargs):
                if record['experiment_id'] == 'tau_015':
                    raise KeyboardInterrupt
                return real(command, attempt, record, *args, **kwargs)
            with patch.object(runner, 'run_worker', side_effect=interrupt_tau), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(KeyboardInterrupt):
                    runner.run(path, 'round_02')
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.run(path, 'round_02'), 0)
            state = runner.load_state(root / 'work/history.json')
            self.assertEqual(state['executions'][0], original)
            self.assertEqual(runner.read_yaml(path)['rounds'][0], first_round)
            records = [r for r in state['executions'] if r['round_id'] == 'round_02']
            successes = [r for r in records if r['status'] == 'SUCCESS']
            self.assertEqual(len(successes), 13)
            self.assertEqual(len(records), 14)  # One interruption, no baseline training.
            self.assertFalse(any(r['experiment_id'] == 'baseline_omega0' for r in records))
            self.assertTrue(all(r['config']['seed'] == 0 and r['training_record']['config']['seed'] == 0 for r in successes))
            omega = [original] + [r for r in successes if r['experiment_id'].startswith('omega_')]
            winner_omega = max(omega, key=lambda r: runner.configuration_key(r['training_record'], runner.TEST_RULE))
            tau = [r for r in successes if r['experiment_id'].startswith('tau_')]
            winner_tau = max(omega + tau, key=lambda r: runner.configuration_key(r['training_record'], runner.TEST_RULE))
            for record in tau:
                self.assertEqual(record['config']['omega'], winner_omega['config']['omega'])
                self.assertEqual(record['dependency_selection']['winner_execution_id'], winner_omega['execution_id'])
            for record in [r for r in successes if r['experiment_id'].startswith('gamma_')]:
                self.assertEqual(record['config']['omega'], winner_tau['config']['omega'])
                self.assertEqual(record['config']['tau'], winner_tau['config']['tau'])
                self.assertEqual(record['config']['gamma_1'] + record['config']['gamma_2'], .5)
                self.assertEqual(record['dependency_selection']['winner_execution_id'], winner_tau['execution_id'])
            summary = json.loads((root / 'work/Youshu/round_02/round_summary.json').read_text(encoding='utf-8'))
            self.assertTrue(summary['complete'])
            self.assertEqual(summary['training_planned'], 13)
            self.assertEqual(summary['reference_count'], 1)
            self.assertEqual(summary['valid_successes'], 14)
            self.assertEqual(summary['configuration_selection_rule'], runner.TEST_RULE)
            self.assertEqual(summary['stages'][0]['best']['execution_id'], winner_omega['execution_id'])
            self.assertEqual(summary['stages'][1]['best']['execution_id'], winner_tau['execution_id'])
            final = max([original] + successes, key=lambda r: runner.configuration_key(r['training_record'], runner.TEST_RULE))
            self.assertEqual(summary['best']['execution_id'], final['execution_id'])
            self.assertEqual(runner.read_yaml(path)['rounds'][1]['experiments'][0]['status'], 'REFERENCED')
            report = (root / 'search.txt').read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.run(path, 'round_02'), 0)
            self.assertEqual((root / 'search.txt').read_bytes(), report)
            self.assertEqual(len(runner.load_state(root / 'work/history.json')['executions']), 15)

    def test_failed_omega_blocks_both_following_stages(self):
        with tempfile.TemporaryDirectory(prefix='dcbr-omega-failed-') as temp:
            root = Path(temp)
            path, original = self.make_second_round_plan(root)
            with patch.object(runner, 'preflight'), patch.object(runner, 'run_worker', return_value=8), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.run(path, 'round_02'), 1)
            state = runner.load_state(root / 'work/history.json')
            self.assertEqual(state['executions'][0], original)
            self.assertEqual(len(state['executions']), 6)
            self.assertTrue(all(r['experiment_id'].startswith('omega_') and r['status'] == 'FAILED' for r in state['executions'][1:]))
            entries = runner.read_yaml(path)['rounds'][1]['experiments']
            self.assertTrue(all(e['status'] == 'BLOCKED' and e['resolved_config'] is None for e in entries[6:]))
            summary = json.loads((root / 'work/Youshu/round_02/round_summary.json').read_text(encoding='utf-8'))
            self.assertFalse(summary['complete'])
            self.assertIsNone(summary['best'])
            self.assertTrue(all(s['best'] is None for s in summary['stages']))

    def test_netease_plan_preserves_budget_controls_and_parameter_sources(self):
        plan = runner.load_plan(HERE / '搜参配置.yaml')
        round_ = next(r for r in plan['rounds'] if r['id'] == 'netease_round_01')
        items = round_['experiments']
        base = runner.read_yaml(HERE / 'config.yaml')['NetEase']
        self.assertEqual(round_['dataset'], 'NetEase')
        self.assertEqual(round_['selection_rule'], runner.CROSS_RULE)
        self.assertEqual(len(items), 19)
        self.assertFalse(any(i['reference'] for i in items))
        self.assertEqual(items[0]['template_config'], dict(base, data_path=str(runner.resolve_path(base['data_path']))))
        expected = [('lambda_1', [.005, .01, .03, .05]),
                    ('lambda_0', [100, 500, 2000, 10000]),
                    ('omega', [.2, .3, .5, .6]), ('tau', [.1, .15, .25, .3]),
                    ('lambda_2', [1e-6, 1e-7])]
        start = 1
        for stage, (key, values) in zip(round_['stages'], expected):
            end = start + len(values)
            with self.subTest(parameter=key):
                self.assertEqual(stage['id'], key)
                self.assertEqual(stage['candidates'], [i['id'] for i in items[:end]])
                self.assertEqual([i['overrides'] for i in items[start:end]], [{key: v} for v in values])
                for item in items[start:end]:
                    self.assertEqual(item['inherit_best'],
                                     {'candidates': [i['id'] for i in items[:start]], 'rule': runner.CROSS_RULE})
                if key != 'lambda_2':
                    self.assertEqual(sum(v < base[key] for v in values), 2)
                    self.assertEqual(sum(v > base[key] for v in values), 2)
            start = end
        self.assertTrue(all(i['template_config']['seed'] == 4096 and i['template_config']['epochs'] == 40
                            and i['template_config']['test_interval'] == 5 for i in items))
        runner.resolve_dependencies(plan, round_, runner.empty_state())
        self.assertIsNotNone(items[0]['config'])
        self.assertTrue(all(i['config'] is None and i['fingerprint'] is None for i in items[1:]))

    def test_netease_five_stage_validation_selection_and_failure_propagation(self):
        plan = runner.load_plan(HERE / '搜参配置.yaml')
        round_ = next(r for r in plan['rounds'] if r['id'] == 'netease_round_01')
        items = round_['experiments']
        # Recall dominates NDCG, NDCG breaks Recall ties, exact ties retain the
        # earlier control. Deliberately opposite test scores must have no effect.
        metrics = [(.1, .1), (.11, .01), (.11, .02), (.11, .02), (.10, .99),
                   (.11, .02), (.10, .9), (.10, .8), (.10, .7),
                   (.12, .02), (.11, .9), (.11, .8), (.11, .7),
                   (.12, .03), (.12, .03), (.11, .9), (.11, .8),
                   (.12, .03), (.11, .99)]
        state = runner.empty_state()
        valid = lambda record, *args: record is not None and record['status'] == 'SUCCESS'
        expected_parent = {1: 0, 5: 2, 9: 2, 13: 9, 17: 13}
        with patch.object(runner, 'check_success', side_effect=valid):
            for index, (item, (recall, ndcg)) in enumerate(zip(items, metrics)):
                runner.resolve_dependencies(plan, round_, state)
                if index in expected_parent:
                    parent = state['executions'][expected_parent[index]]
                    self.assertEqual(item['dependency_selection']['winner_execution_id'], parent['execution_id'])
                    self.assertEqual(item['config'], dict(parent['config'], **item['overrides']))
                self.assertIsNotNone(item['config'])
                state['executions'].append({'round_id': round_['id'], 'experiment_id': item['id'],
                    'execution_id': f'netease_{index}', 'fingerprint': item['fingerprint'], 'status': 'SUCCESS',
                    'config': copy.deepcopy(item['config']),
                    'training_record': {'result': {'metrics': {
                        'val': {'recall': {'20': recall}, 'ndcg': {'20': ndcg}},
                        'test': {'recall': {'20': 1 - recall}, 'ndcg': {'20': 1 - ndcg}}}}}})
            winner = max(state['executions'], key=lambda r: runner.validation_key(r['training_record']))
            self.assertEqual(winner['experiment_id'], 'tau_010')
            self.assertEqual({k: winner['config'][k] for k in ('lambda_1', 'lambda_0', 'omega', 'tau', 'lambda_2')},
                             dict(lambda_1=.01, lambda_0=1000., omega=.2, tau=.1, lambda_2=1e-5))
            state['executions'][0]['status'] = 'FAILED'
            runner.resolve_dependencies(plan, round_, state)
            self.assertTrue(all(i['config'] is None and 'initial' in i['blocked_by'] for i in items[1:]))

    def test_netease_nineteen_trial_training_summary_and_resume(self):
        with tempfile.TemporaryDirectory(prefix='dcbr-netease-') as temp:
            root = Path(temp)
            path = self.make_plan(root)
            document = runner.read_yaml(path)
            base_path = Path(document['base_config'])
            tiny = runner.read_yaml(base_path)['Youshu']
            data = root / 'data/Youshu'
            data.rename(root / 'data/NetEase')
            (root / 'data/NetEase/Youshu_data_size.txt').rename(root / 'data/NetEase/NetEase_data_size.txt')
            base = runner.read_yaml(HERE / 'config.yaml')['NetEase']
            # Only reduce compute/data for the fixture; retain production
            # candidates, seed, stage ordering and all five dependency groups.
            fixture = {k: tiny[k] for k in ('data_path', 'batch_size_train', 'batch_size_test',
                        'epochs', 'test_interval', 'steps', 'dims', 'embedding_size')}
            base.update(fixture)
            base_path.write_text(yaml.safe_dump({'NetEase': base}), encoding='utf-8')
            reviewed = copy.deepcopy(next(r for r in runner.read_yaml(HERE / '搜参配置.yaml')['rounds']
                                          if r['id'] == 'netease_round_01'))
            reviewed['overrides'].update(fixture)
            document['rounds'] = [reviewed]
            path.write_text(yaml.safe_dump(document, allow_unicode=True, sort_keys=False), encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.run(path, 'netease_round_01'), 0)
            state = runner.load_state(root / 'work/history.json')
            records = state['executions']
            self.assertEqual(len(records), 19)
            self.assertTrue(all(r['status'] == 'SUCCESS' and len(r['attempts']) == 1 for r in records))
            self.assertTrue(all(r['config']['seed'] == 4096 and r['config']['epochs'] == 2 for r in records))
            for record, item in zip(records[1:], reviewed['experiments'][1:]):
                candidates = [r for r in records if r['experiment_id'] in item['inherit_best']['candidates']]
                winner = max(candidates, key=lambda r: runner.validation_key(r['training_record']))
                self.assertEqual(record['dependency_selection']['winner_execution_id'], winner['execution_id'])
                self.assertEqual(record['config'], dict(winner['config'], **item['overrides']))
                self.assertEqual(record['training_record']['experiment_case'], 'second')
            summary = json.loads((root / 'work/NetEase/netease_round_01/round_summary.json').read_text(encoding='utf-8'))
            self.assertTrue(summary['complete'])
            self.assertEqual(summary['valid_successes'], 19)
            self.assertEqual(summary['configuration_selection_rule'], runner.CROSS_RULE)
            self.assertEqual([s['id'] for s in summary['stages']], ['lambda_1', 'lambda_0', 'omega', 'tau', 'lambda_2'])
            for stage in summary['stages']:
                candidates = [r for r in records if r['experiment_id'] in
                              next(s['candidates'] for s in reviewed['stages'] if s['id'] == stage['id'])]
                winner = max(candidates, key=lambda r: runner.validation_key(r['training_record']))
                self.assertTrue(stage['complete'])
                self.assertEqual(stage['best']['execution_id'], winner['execution_id'])
            self.assertEqual(summary['best']['execution_id'], summary['stages'][-1]['best']['execution_id'])
            report = (root / 'search.txt').read_bytes()
            self.assertEqual(report.count(b'END_EXECUTION:'), 19)
            # Resuming a completed round must not rerun or append duplicate results.
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.run(path, 'netease_round_01'), 0)
            self.assertEqual(len(runner.load_state(root / 'work/history.json')['executions']), 19)
            self.assertEqual((root / 'search.txt').read_bytes(), report)

    def test_second_round_rejects_rule_mismatch_and_bad_references(self):
        original = runner.read_yaml(HERE / '搜参配置.yaml')
        with tempfile.TemporaryDirectory(prefix='dcbr-round-schema-') as temp:
            path = Path(temp) / 'plan.yaml'
            for edit in (lambda r: r.update(selection_rule='unknown'),
                         lambda r: r['experiments'][6]['inherit_best'].update(rule=runner.CROSS_RULE),
                         lambda r: r['experiments'][0]['reference'].update(round_id='youshu_round_02'),
                         lambda r: r['experiments'][0]['reference'].update(round_id='future'),
                         lambda r: r['experiments'][0]['reference'].update(fingerprint='bad'),
                         lambda r: r['stages'][0].update(candidates=['unknown'])):
                document = copy.deepcopy(original)
                edit(document['rounds'][1])
                path.write_text(yaml.safe_dump(document, allow_unicode=True), encoding='utf-8')
                with self.assertRaises(ValueError):
                    runner.load_plan(path)


if __name__ == "__main__":
    unittest.main()
