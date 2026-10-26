"""Checks for the agreed first-round plan; no formal training or failure injection."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml

import test_search as shared

runner = shared.runner
config = shared.config
worker = shared.worker
HERE = shared.HERE


class RoundPlanTests(unittest.TestCase):
    def plans(self):
        for dataset in ("Meal-H", "Meal-L"):
            yield runner.load_plan(HERE / dataset / "搜参配置.yaml")

    def test_agreed_grids_switches_and_fixed_protocol(self):
        expected = {
            "Meal-H": (.05, [.01, .02, .04, .1, .2, .4], [1, 3, 7, 13, 19, 35]),
            "Meal-L": (.04, [.005, .01, .02, .05, .1, .2], [1, 3, 5, 7, 11, 19]),
        }
        for plan in self.plans():
            round_ = plan["rounds"][0]
            items = round_["experiments"]
            initial, lambdas, ks = expected[round_["dataset"]]
            self.assertEqual(len(items), 26)
            self.assertEqual(items[0]["config"]["c_lambdas"], [initial])
            self.assertEqual([e["overrides"]["c_lambdas"][0] for e in items[1:7]], lambdas)
            self.assertEqual([e["overrides"]["latent_rebuild"]["rebuild_k"] for e in items[7:13]], ks)
            for i, item in enumerate(items):
                conf = item["template_config"]
                self.assertEqual(conf["latent_rebuild"]["enabled"], i >= 7)
                self.assertFalse(conf["anchor_cl"]["enabled"])
                self.assertEqual(conf["anchor_cl"]["anchor_cl_lambda"], 0)
                self.assertEqual((conf["seed"], conf["epochs"]), (2026, 150))
                if i >= 20:
                    self.assertEqual(item["fusion_transfer"]["amount"], .05)
                    self.assertEqual(item["inherit_best"]["candidates"], [e["id"] for e in items[7:20]])
            self.assertEqual(round_["final_candidates"], [e["id"] for e in items[7:]])

    def test_all_stages_inherit_and_refine_same_winner_and_exclude_off_final(self):
        for plan in self.plans():
            round_ = plan["rounds"][0]
            items = round_["experiments"]
            state = runner.empty_state()
            # Disabled trials deliberately beat every enabled trial. They must
            # select lambda, but must never become the final reported winner.
            with patch.object(runner, "check_success", side_effect=lambda record, *args: record is not None):
                for index, item in enumerate(items):
                    runner.resolve_dependencies(plan, round_, state)
                    conf = item["config"]
                    self.assertIsNotNone(conf)
                    if 1 <= index <= 6:
                        self.assertFalse(conf["latent_rebuild"]["enabled"])
                    if index >= 7:
                        self.assertTrue(conf["latent_rebuild"]["enabled"])
                        self.assertEqual(conf["c_lambdas"], items[2]["config"]["c_lambdas"])
                    if index >= 13:
                        self.assertEqual(conf["latent_rebuild"]["rebuild_k"], items[9]["config"]["latent_rebuild"]["rebuild_k"])
                    if index >= 20:
                        adjustment = item["dependency_selection"]["fusion_adjustment"]
                        self.assertEqual(item["dependency_selection"]["winner_experiment_id"], items[15]["id"])
                        self.assertEqual(adjustment["base_modal_weight"], items[15]["config"]["fusion_weights"]["modal_weight"])
                        weights = conf["fusion_weights"]["modal_weight"]
                        self.assertAlmostEqual(sum(weights), 1)
                        self.assertGreater(min(weights), 0)
                    score = .90 if index == 2 else .80 if index < 7 else .40
                    if index == 9:
                        score = .45
                    if index == 15:
                        score = .50
                    if index == 20:
                        score = .60  # Fine trials must not shift their shared reference to this result.
                    metrics = {"val": {"recall": {"20": score}, "ndcg": {"20": score / 2}},
                               "test": {"recall": {"20": 1 - score}}}
                    state["executions"].append({"execution_id": f"synthetic_{index}", "round_id": round_["id"],
                                                "experiment_id": item["id"], "fingerprint": item["fingerprint"],
                                                "status": "SUCCESS", "training_record": {
                                                    "result": {"metrics": metrics},
                                                    "artifacts": {"checkpoint_model": "synthetic.pt"}}})
                with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
                    plan["work_dir"] = Path(temp) / "work"
                    plan["result_file"] = Path(temp) / "results.txt"
                    summary = runner.write_round_summary(plan, round_, state)
                self.assertTrue(summary["complete"])
                self.assertEqual(summary["best"]["experiment_id"], items[20]["id"])
                self.assertTrue(summary["best"]["config"]["latent_rebuild"]["enabled"])
                self.assertEqual(summary["stages"][0]["best"]["experiment_id"], items[2]["id"])
                self.assertEqual(summary["stages"][1]["best"]["experiment_id"], items[9]["id"])

    def test_disabled_diffusion_training_and_checkpoint_evaluation(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            _, path, conf = shared.fixture(root)
            conf["latent_rebuild"]["enabled"] = False
            path.write_text(yaml.safe_dump({"Meal-H": conf}), encoding="utf-8")
            reference = worker.reference_training()
            with patch.object(worker, "reference_training", return_value=reference), \
                    patch.object(reference, "train_latent_diffusion_epoch", side_effect=AssertionError("Diffusion must remain off")), \
                    patch.object(reference, "rebuild_ub_graph_with_latent_diffusion", side_effect=AssertionError("No graph filtering")):
                self.assertEqual(worker.main(["--config", str(path), "-d", "Meal-H", "--device", "cpu", "--quiet",
                                              "--output-dir", str(root / "run")]), 0)
            record = json.loads((root / "run/result.json").read_text(encoding="utf-8"))
            saved = torch.load(root / "run/best.pt", map_location="cpu", weights_only=False)
            self.assertIsNone(saved["diffusion_state"])
            self.assertIsNone(saved["diffusion_optimizer_state"])
            self.assertEqual(saved["ub_graph"].nnz, 15)
            actual = record["config"]
            dataset = worker.make_dataset(reference, actual)
            evaluated = worker.evaluate_checkpoint(reference, root / "run/best.pt", actual, dataset,
                                                   lambda text: None, record["data_files"])
            self.assertEqual(json.loads(json.dumps(evaluated)), record["result"]["metrics"])
            runner.validate_result(root / "run/result.json", config.normalize_config(conf, "Meal-H"),
                                   "Meal-H", record["sources"], record["data_files"], "", "0", "cpu")


if __name__ == "__main__":
    unittest.main()
