"""CPU fixtures only; no real dataset training or formal search-state changes."""
import contextlib
import copy
import importlib.util
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

import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import recobr_config as config
import run_experiments as runner

spec = importlib.util.spec_from_file_location("recobr_worker_test", HERE / "train.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def fixture(root):
    directory = root / "data" / "Meal-H"
    directory.mkdir(parents=True)
    (directory / "Meal-H_data_size.txt").write_text("6\t100\t8\n")
    relations = {
        "user_bundle_train": [(u, (u + j) % 100) for u in range(5) for j in range(3)],
        "user_bundle_tune": [(u, b) for u in range(6) for b in range(10, 50)],
        "user_bundle_test": [(u, b) for u in range(6) for b in range(50, 100)],
        "user_item": [(u, (u + j) % 8) for u in range(6) for j in range(3)],
        "bundle_item": [(b, (b + j) % 8) for b in range(100) for j in range(3)],
    }
    for name, pairs in relations.items():
        (directory / (name + ".txt")).write_text("".join(f"{a}\t{b}\n" for a, b in pairs))
    generator = torch.Generator().manual_seed(17)
    torch.save({u: torch.randn(8, generator=generator) for u in range(5)}, root / "users.pt")
    torch.save({b: torch.randn(8, generator=generator) for b in range(7)}, root / "bundles.pt")
    conf = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))["Meal-H"]
    conf.update(data_path=str(root / "data"), batch_size_train=4, batch_size_test=3,
                embedding_sizes=[8], epochs=2, test_interval=1)
    conf["latent_rebuild"].update(rebuild_k=1, latent_diffusion_hidden_dim=8,
                                latent_diffusion_time_dim=4, latent_diffusion_denoiser_layers=1,
                                latent_diffusion_num_steps=3, latent_diffusion_batch_size=3,
                                latent_diffusion_infer_batch_size=3,
                                latent_diffusion_user_embedding_path=str(root / "users.pt"),
                                latent_diffusion_bundle_embedding_path=str(root / "bundles.pt"))
    path = root / "base.yaml"
    path.write_text(yaml.safe_dump({"Meal-H": conf}), encoding="utf-8")
    document = {"version": 1, "base_config": str(path), "work_dir": str(root / "work"),
                "result_file": str(root / "results.txt"),
                "defaults": {"gpu": "0", "device": "cpu", "max_attempts": 2, "retry_delay_seconds": 0,
                             "console_interval_seconds": 1, "stream_output": False},
                "rounds": [{"id": "tiny", "dataset": "Meal-H", "note": "CPU test only",
                            "overrides": copy.deepcopy(conf),
                            "experiments": [{"id": "initial", "note": "baseline", "overrides": {}}]},
                           {"id": "later", "dataset": "Meal-H", "note": "not selected",
                            "experiments": [{"id": "initial", "note": "later"}]}]}
    plan = root / "plan.yaml"
    plan.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return plan, path, conf


class SearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_dataset_plans_have_independent_writable_paths(self):
        writable = []
        for dataset, round_id in (("Meal-H", "meal_h_initial"), ("Meal-L", "meal_l_initial")):
            path = HERE / dataset / "搜参配置.yaml"
            plan = runner.load_plan(path)
            self.assertTrue(all(r["dataset"] == dataset for r in plan["rounds"]))
            self.assertIn(round_id, [r["id"] for r in plan["rounds"]])
            writable.append({path.resolve(), Path(str(path.resolve()) + ".lock"),
                             plan["work_dir"], plan["work_dir"] / "history.json",
                             plan["work_dir"] / "controller.lock", plan["result_file"],
                             Path(str(plan["result_file"]) + ".lock")})
            self.assertEqual(plan["result_file"].parent, path.parent)
            self.assertTrue(plan["work_dir"].is_relative_to(path.parent))
        self.assertFalse(writable[0] & writable[1])

    def test_schema_nested_merge_stage_guards_and_fixed_seed(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, path, base = fixture(Path(temp))
            normalized = config.normalize_config(base, "Meal-H")
            merged = config.merge_config(base, {"latent_rebuild": {"rebuild_k": 3}})
            self.assertEqual(merged["latent_rebuild"]["latent_diffusion_lr"], base["latent_rebuild"]["latent_diffusion_lr"])
            self.assertEqual(base["latent_rebuild"]["rebuild_k"], 1)
            for change in ({"lrs": [.001, .002]}, {"anchor_cl": {"enabled": True}},
                           {"latent_rebuild": {"enabled": False}}, {"typo": 1},
                           {"latent_rebuild": {"latent_diffusion_dropout": 1}}):
                with self.assertRaises(ValueError):
                    config.normalize_config(config.merge_config(base, change), "Meal-H")
            document = runner.read_yaml(plan)
            for change in ({"seed": 7}, {"anchor_cl": {"temp": .4}},
                           {"latent_rebuild": {"latent_diffusion_user_embedding_path": str(Path(temp)/"users.pt")}}):
                bad = copy.deepcopy(document)
                bad["rounds"][0]["experiments"][0]["overrides"] = change
                plan.write_text(yaml.safe_dump(bad), encoding="utf-8")
                with self.assertRaises(ValueError):
                    runner.load_plan(plan)
            self.assertEqual(normalized["seed"], 2026)

    def test_read_only_dry_run_and_input_fingerprints(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan, _, _ = fixture(root)
            original = plan.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.run(plan, "tiny", dry_run=True), 0)
            self.assertEqual(plan.read_bytes(), original)
            self.assertFalse((root / "work").exists())
            old = runner.load_plan(plan)["rounds"][0]["experiments"][0]["fingerprint"]
            self.assertNotEqual(old, runner.load_plan(plan, gpu="1")["rounds"][0]["experiments"][0]["fingerprint"])
            with (root / "users.pt").open("ab") as stream:
                stream.write(b"changed")
            new = runner.load_plan(plan)["rounds"][0]["experiments"][0]["fingerprint"]
            self.assertNotEqual(old, new)

    def test_worker_matches_root_training_and_reloads_saved_graph(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            _, path, base = fixture(root)
            args = ["--config", str(path), "-d", "Meal-H", "--device", "cpu", "--quiet",
                    "--output-dir", str(root / "new")]
            self.assertEqual(worker.main(args), 0)
            trained = json.loads((root / "new/result.json").read_text(encoding="utf-8"))
            self.assertEqual(worker.load_configuration(root / "new/config.yaml", "Meal-H"),
                             config.normalize_config(base, "Meal-H"))
            reference = worker.reference_training()
            conf = config.runtime_config(config.normalize_config(base, "Meal-H"), "Meal-H", "0", "cpu", "")
            conf["quiet"] = True
            dataset = worker.make_dataset(reference, conf)
            reference.set_seed(conf["seed"])
            old_cwd = os.getcwd()
            os.chdir(root)
            try:
                reference.run_recobr_training(copy.deepcopy(conf), dataset, "cpu")
            finally:
                os.chdir(old_cwd)
            checkpoints = list((root / "checkpoints/Meal-H/ReCoBR/model").iterdir())
            original = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
            actual = torch.load(root / "new/best.pt", map_location="cpu", weights_only=False)
            self.assertEqual(set(original), set(actual["model_state"]))
            for key in original:
                torch.testing.assert_close(original[key], actual["model_state"][key], rtol=0, atol=0)
            metrics = worker.evaluate_checkpoint(reference, root / "new/best.pt", conf, dataset,
                                                  lambda text: None, trained["data_files"])
            self.assertEqual(json.loads(json.dumps(metrics)), trained["result"]["metrics"])
            # K=1, and the inactive user remains in the complete evaluation space.
            self.assertEqual(actual["ub_graph"].nnz, 5)
            self.assertEqual(trained["config"]["num_users"], 6)
            command = [sys.executable, "-B", str(HERE / "train.py"), "--config", str(root / "new/config.yaml"),
                       "-d", "Meal-H", "--device", "cpu", "--quiet", "--evaluate", str(root / "new/best.pt"),
                       "--output-dir", str(root / "evaluation")]
            evaluated = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=60)
            self.assertEqual(evaluated.returncode, 0, evaluated.stdout + evaluated.stderr)
            evaluation = json.loads((root / "evaluation/result.json").read_text(encoding="utf-8"))
            self.assertEqual(evaluation["status"], "EVALUATED")
            self.assertEqual(evaluation["evaluation_metrics"], trained["result"]["metrics"])

    def test_cross_round_history_and_explicit_text_references(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, _, base = fixture(root)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.run(path, "tiny"), 0)
            original = runner.load_state(root / "work/history.json")["executions"][0]
            ref = {"round_id": "tiny", "experiment_id": "initial", "execution_id": original["execution_id"],
                   "fingerprint": original["fingerprint"]}
            document = runner.read_yaml(path)
            document["rounds"].append({"id": "reuse", "dataset": "Meal-H", "note": "reuse pinned baseline",
                                       "overrides": base, "experiments": [
                                           {"id": "baseline", "note": "pinned history", "reference": ref}]})
            path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
            with patch.object(runner, "run_worker", side_effect=AssertionError("References must not train")), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.run(path, "reuse"), 0)
            self.assertEqual(len(runner.load_state(root / "work/history.json")["executions"]), 1)
            document = runner.read_yaml(path)
            ref["result_file"] = str(root / "results.txt")
            document["rounds"].append({"id": "text_reuse", "dataset": "Meal-H", "note": "explicit TXT reuse",
                                       "overrides": base, "experiments": [
                                           {"id": "baseline", "note": "pinned TXT", "reference": ref}]})
            path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
            with patch.object(runner, "run_worker", side_effect=AssertionError("References must not train")), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.run(path, "text_reuse"), 0)
            plan = runner.load_plan(path)
            round_ = plan["rounds"][-1]
            runner.resolve_dependencies(plan, round_, runner.empty_state())
            record = round_["experiments"][0]["_reference_record"]
            self.assertFalse(record["reference_evidence"]["artifacts_verified"])
            self.assertEqual(runner.validation_key(record["training_record"]),
                             runner.validation_key(original["training_record"]))

    def test_real_serial_dependencies_resume_rerun_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, _, _ = fixture(root)
            document = runner.read_yaml(path)
            experiments = document["rounds"][0]["experiments"]
            experiments.extend([
                {"id": "k3", "note": "K test", "overrides": {"latent_rebuild": {"rebuild_k": 3}},
                 "inherit_best": {"candidates": ["initial"], "rule": runner.CROSS_RULE}},
                {"id": "temperature", "note": "inherit full K winner", "overrides": {"c_temps": [.3]},
                 "inherit_best": {"candidates": ["initial", "k3"], "rule": runner.CROSS_RULE}}])
            document["rounds"][0]["stages"] = [{"id": "k", "candidates": ["initial", "k3"]}]
            path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.run(path, "tiny"), 0)
            state_path = root / "work/history.json"
            state = runner.load_state(state_path)
            self.assertEqual(len(state["executions"]), 3)
            self.assertTrue(all(r["status"] == "SUCCESS" for r in state["executions"]))
            first, second, third = state["executions"]
            self.assertLessEqual(first["ended_at"], second["started_at"])
            self.assertEqual(second["config"]["latent_rebuild"]["rebuild_k"], 3)
            winner = max((first, second), key=lambda r: runner.validation_key(r["training_record"]))
            self.assertEqual(third["config"]["latent_rebuild"], winner["config"]["latent_rebuild"])
            self.assertEqual(third["config"]["c_temps"], [.3])
            self.assertTrue(all(r["config"]["seed"] == 2026 for r in state["executions"]))
            self.assertNotIn("status", runner.read_yaml(path)["rounds"][1]["experiments"][0])
            before = (root / "results.txt").read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.run(path, "tiny"), 0)
            self.assertEqual(len(runner.load_state(state_path)["executions"]), 3)
            self.assertEqual(before, (root / "results.txt").read_bytes())
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.run(path, "tiny", mode="selected", only=["temperature"]), 0)
            self.assertEqual(len(runner.load_state(state_path)["executions"]), 4)
            plan = runner.load_plan(path)
            round_ = plan["rounds"][0]
            item = round_["experiments"][0]
            trained = first["training_record"]
            for mutate in (lambda r: r.update(completed_epochs=1),
                           lambda r: r["sources"].update({"model.py": "changed"}),
                           lambda r: r["result"]["selection_metrics"].update({"recall@20": 0})):
                damaged = copy.deepcopy(trained)
                mutate(damaged)
                result = root / "damaged.json"
                result.write_text(json.dumps(damaged), encoding="utf-8")
                with self.assertRaises(ValueError):
                    runner.validate_result(result, item["config"], "Meal-H", plan["sources"],
                                           round_["data_files"], first["attempts"][0]["info"], "0", "cpu")
            Path(trained["artifacts"]["checkpoint_model"]).write_bytes(b"damaged")
            plan = runner.load_plan(path)
            selected = runner.select_experiments(plan, plan["rounds"][0], runner.load_state(state_path), "resume", [])
            self.assertIn("initial", [e["id"] for e in selected])
            self.assertTrue(plan["rounds"][0]["experiments"][1]["blocked_by"])

    def test_retry_then_success_failure_and_interruption_recovery(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, _, _ = fixture(root)
            real = runner.run_worker
            calls = []
            def fail_once(*args, **kwargs):
                calls.append(1)
                return 7 if len(calls) == 1 else real(*args, **kwargs)
            with patch.object(runner, "run_worker", side_effect=fail_once), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.run(path, "tiny"), 0)
            state = runner.load_state(root / "work/history.json")
            self.assertEqual([a["status"] for a in state["executions"][0]["attempts"]], ["FAILED", "SUCCESS"])
            with patch.object(runner, "run_worker", return_value=9), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.run(path, "tiny", mode="all"), 1)
            with patch.object(runner, "run_worker", side_effect=KeyboardInterrupt), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(KeyboardInterrupt):
                    runner.run(path, "tiny", mode="failed")
            state_path = root / "work/history.json"
            state = runner.load_state(state_path)
            last = state["executions"][-1]
            self.assertEqual(last["status"], "INTERRUPTED")
            self.assertEqual(len(last["attempts"]), 1)
            last.update(status="RUNNING", host=socket.gethostname())
            last["attempts"][-1].update(status="RUNNING", pid=None)
            runner.recover(state, state_path)
            self.assertEqual(last["status"], "INTERRUPTED")
            self.assertIsNone(last["duration_seconds"])

    def test_validation_ranking_ignores_test_and_strict_checkpoint_rule(self):
        def record(r, n, test):
            return {"result": {"metrics": {"val": {"recall": {"20": r}, "ndcg": {"20": n}},
                                             "test": {"recall": {"20": test}}}}}
        records = [record(.2, .1, .99), record(.21, .08, .01), record(.21, .09, 0)]
        self.assertIs(max(records, key=runner.validation_key), records[2])
        self.assertIs(max([records[2], copy.deepcopy(records[2])], key=runner.validation_key), records[2])
        with self.assertRaises(ValueError):
            runner.configuration_key(records[0], "test_recall20")
        reference = worker.reference_training()
        best = {"recall": {20: .2}, "ndcg": {20: .1}}
        self.assertFalse(reference.is_better_validation({"recall": {20: .21}, "ndcg": {20: .1}}, best))
        self.assertFalse(reference.is_better_validation({"recall": {20: .19}, "ndcg": {20: .2}}, best))
        self.assertTrue(reference.is_better_validation({"recall": {20: .21}, "ndcg": {20: .11}}, best))


if __name__ == "__main__":
    unittest.main()
