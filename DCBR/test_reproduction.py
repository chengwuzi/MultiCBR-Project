"""Numerical port checks and tiny end-to-end experiments; no full iFashion run.

Run: python -m unittest discover -s DCBR -p test_reproduction.py -v
Only tests load raw code as a reference. Production train.py is independent.
"""
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import scipy.sparse as sp
import torch
import yaml


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.dont_write_bytecode = True  # Reference imports must not create files under raw/.


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


port = load_module("dcbr_port_test", HERE / "train.py")
raw = load_module("dcbr_original_test", HERE / "raw/models/DCBR.py")


class ReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        with (HERE / "config.yaml").open(encoding="utf-8") as stream:
            cls.base = yaml.safe_load(stream)["iFashion"]
        cls.graphs = [sp.csr_matrix(np.asarray(values, dtype=np.float32)) for values in [
            [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]],
            [[1, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 1, 1]],
            [[1, 0, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 1, 1]],
        ]]
        cls.batch = [torch.tensor([[0], [1], [2], [0]]),
                     torch.tensor([[0, 2], [1, 3], [2, 0], [1, 3]])]

    def conf(self):
        conf = copy.deepcopy(self.base)
        conf.update(device=torch.device("cpu"), num_users=3, num_bundles=4, num_items=5,
                    embedding_size=8, dims=[12], steps=5)
        return conf

    def assert_tensor_equal(self, left, right):
        self.assertTrue(torch.isfinite(left).all())
        self.assertTrue(torch.isfinite(right).all())
        torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_author_configuration_preserved(self):
        author = yaml.safe_load((HERE / "raw/configs/overall.yaml").read_text(encoding="utf-8"))
        author.update(yaml.safe_load((HERE / "raw/configs/models/DCBR.yaml").read_text(encoding="utf-8"))["iFashion"])
        author["seed"] = 0
        author["topk"] = [10, 20, 40, 80]
        self.assertEqual(author, self.base)

    def reference_recobr(self):
        sys.path.insert(0, str(ROOT))
        try:
            return load_module("recobr_selection_test", ROOT / "train.py")
        finally:
            sys.path.pop(0)

    def metric_values(self, recall20, ndcg20, other=.2):
        values = {name: {k: other for k in self.base["topk"]} for name in ("recall", "ndcg")}
        values["recall"][20] = recall20
        values["ndcg"][20] = ndcg20
        return values

    def test_shared_selection_requires_both_strict_improvements(self):
        reference = self.reference_recobr()
        self.assertEqual(reference.SELECTION_RULE, port.SELECTION_RULE)
        baseline = self.metric_values(.16, .121)
        cases = [(.17, .122, True), (.17, .120, False), (.159, .13, False),
                 (.16, .13, False), (.17, .121, False), (.16, .121, False)]
        for recall20, ndcg20, expected in cases:
            # Non-selection metrics jump to .99 in all cases: they must never
            # override the strict conjunction on validation Top-20.
            candidate = self.metric_values(recall20, ndcg20, other=.99)
            self.assertEqual(port.is_better_validation(candidate, baseline), expected)
            self.assertEqual(reference.is_better_validation(candidate, baseline), expected)

    def test_recobr_saves_only_joint_improvements(self):
        reference = self.reference_recobr()
        with tempfile.TemporaryDirectory(prefix="recobr-selection-") as temp:
            directory = Path(temp)
            conf = {"topk": self.base["topk"], "device": torch.device("cpu")}
            best, performance = reference.init_best_metrics(conf)
            model = torch.nn.Linear(2, 2)
            writer = Mock()
            current_epoch = -1
            save_torch = torch.save
            with contextlib.redirect_stdout(io.StringIO()), patch.object(reference.torch, "save", wraps=save_torch) as save:
                for epoch, (recall20, ndcg20) in enumerate([(.16, .121), (.159, .13), (.17, .122)]):
                    values = {"val": self.metric_values(recall20, ndcg20, other=.99 if epoch == 1 else .2),
                              "test": self.metric_values(.99, .99, other=.99)}
                    best, performance, current_epoch = reference.log_metrics(
                        conf, model, values, writer, str(directory / "log.txt"), str(directory / "best.pt"),
                        str(directory / "config.json"), epoch, epoch, best, performance, current_epoch)
                    self.assertEqual(save.call_count, [1, 1, 2][epoch])
                    self.assertEqual(current_epoch, [0, 0, 2][epoch])
            self.assertEqual(best["val"]["ndcg"][20], .122)
            self.assertEqual(json.loads((directory / "config.json").read_text())["selection_rule"], port.SELECTION_RULE)

    def test_recobr_entrypoint_saves_new_selection_protocol(self):
        reference = self.reference_recobr()
        with tempfile.TemporaryDirectory(prefix="recobr-entrypoint-") as temp:
            directory = Path(temp)
            self.make_tiny_data(directory)
            configs = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
            conf = copy.deepcopy(configs["iFashion"])
            conf.update(data_path=str(directory / "data"), batch_size_train=4, batch_size_test=2,
                        epochs=2, test_interval=1)
            for entity, count in (("user", 4), ("bundle", 100)):
                path = directory / (entity + ".pt")
                torch.save(torch.randn(count, 64), path)
                conf["latent_rebuild"]["latent_diffusion_" + entity + "_embedding_path"] = str(path)
            (directory / "config.yaml").write_text(yaml.safe_dump({"iFashion": conf}), encoding="utf-8")
            previous = Path.cwd()
            os.chdir(directory)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    reference.main({"dataset": "iFashion", "model": "ReCoBR", "gpu": "0", "info": "selection_smoke"})
            finally:
                os.chdir(previous)
            saved = list((directory / "checkpoints/iFashion/ReCoBR/conf").iterdir())
            self.assertEqual(len(saved), 1)
            actual = json.loads(saved[0].read_text())
            self.assertEqual(actual["selection_rule"], port.SELECTION_RULE)
            self.assertEqual(len(list((directory / "checkpoints/iFashion/ReCoBR/model").iterdir())), 1)
            # The rule is part of the path fingerprint, separating the old sum
            # rule from the new joint rule under otherwise identical settings.
            old = {key: value for key, value in actual.items() if key != "selection_rule"}
            self.assertNotEqual(reference.build_run_setting(["same"], actual),
                                reference.build_run_setting(["same"], old))

    def test_dcbr_saves_only_joint_improvements(self):
        with tempfile.TemporaryDirectory(prefix="dcbr-selection-") as temp:
            directory = Path(temp)
            path = self.make_tiny_data(directory)
            args = port.get_cmd(["--config", str(path), "--num-workers", "0", "--quiet",
                                 "--output-dir", str(directory / "runs"), "--record-file", str(directory / "record.txt")])
            conf = port.load_config(path, "iFashion", args)
            conf.update(device=torch.device("cpu"), epochs=3)
            evaluations = []
            for recall20, ndcg20 in [(.16, .121), (.159, .13), (.17, .122)]:
                evaluations += [self.metric_values(recall20, ndcg20, other=.99 if ndcg20 == .13 else .2),
                                self.metric_values(.99, .99, other=.99)]
            save_torch = torch.save
            with contextlib.redirect_stdout(io.StringIO()), patch.object(port, "evaluate", side_effect=evaluations), \
                    patch.object(port.torch, "save", wraps=save_torch) as save:
                result = port.run_training(conf, args)
            self.assertEqual([call.args[0]["result"]["best_epoch"] for call in save.call_args_list], [0, 2])
            self.assertEqual(result["result"]["best_epoch"], 2)
            self.assertEqual(result["result"]["selection_metrics"], {"recall@20": .17, "ndcg@20": .122})
            self.assertNotIn("selection_score", result["result"])
            text = (directory / "record.txt").read_text(encoding="utf-8")
            self.assertIn(port.SELECTION_RULE, text)

    def test_full_epoch_core_matches_original(self):
        def trajectory(module):
            port.set_seed(42)
            conf = self.conf()
            model = module.DCBR(conf, self.graphs)
            denoiser = module.DNN([4, 12], [12, 4], 10, norm=False)
            diffusion = module.GaussianDiffusion(.1, .0001, .02, 5)
            optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0)
            diffusion_optimizer = torch.optim.Adam(denoiser.parameters(), lr=.001, weight_decay=0)
            interaction = torch.from_numpy(self.graphs[0].toarray())
            users = torch.arange(3)
            history = []
            for _ in range(2):
                diffusion_optimizer.zero_grad()
                elbo, blcc = diffusion.training_CBDM_losses(
                    denoiser, interaction, model.getUserEmbeds().detach(), model.getBundleEmbeds().detach(), users)
                dloss = elbo + conf["lambda_0"] * blcc
                dloss.backward()
                dgrads = {name: p.grad.clone() for name, p in denoiser.named_parameters()}
                diffusion_optimizer.step()
                with torch.no_grad():
                    predicted = diffusion.p_sample(denoiser, interaction, 0, False)
                    selected = torch.topk(predicted, k=1)[1]
                graph = port.build_rebuilt_graph(users.numpy(), selected.numpy().ravel(), 3, 4, "cpu")
                optimizer.zero_grad()
                bpr, cl = model(graph, self.batch)
                (bpr + cl).backward()
                grads = {name: p.grad.clone() for name, p in model.named_parameters()}
                optimizer.step()
                history.append((torch.stack((elbo, blcc, bpr, cl)).detach(), predicted, graph.to_dense(), dgrads, grads))
            scores = model.evaluate(model.propagate(test=True), users).detach()
            return history, model.state_dict(), denoiser.state_dict(), scores

        # The original hardcodes CUDA. For CPU parity only, make .cuda() an
        # identity operation; no formulas, source files or dtypes are changed.
        with patch.object(torch.Tensor, "cuda", lambda tensor, *args, **kwargs: tensor):
            expected = trajectory(raw)
        actual = trajectory(port)
        for old, new in zip(expected[0], actual[0]):
            for index in range(3):
                self.assert_tensor_equal(old[index], new[index])
            for index in (3, 4):
                self.assertEqual(old[index].keys(), new[index].keys())
                for name in old[index]:
                    self.assert_tensor_equal(old[index][name], new[index][name])
        for index in (1, 2):
            self.assertEqual(expected[index].keys(), actual[index].keys())
            for name in expected[index]:
                self.assert_tensor_equal(expected[index][name], actual[index][name])
        self.assert_tensor_equal(expected[3], actual[3])

    def test_sampling_modes_match_original(self):
        for start_step in (0, 2):
            for sampling_noise in (False, True):
                def sample(module):
                    port.set_seed(18)
                    denoiser = module.DNN([4, 12], [12, 4], 10)
                    diffusion = module.GaussianDiffusion(.1, .0001, .02, 5)
                    return diffusion.p_sample(denoiser, torch.from_numpy(self.graphs[0].toarray()),
                                              start_step, sampling_noise).detach()
                with patch.object(torch.Tensor, "cuda", lambda tensor, *args, **kwargs: tensor):
                    expected = sample(raw)
                self.assert_tensor_equal(expected, sample(port))

    def test_diffusion_rows_and_rebuilt_graph_match_original(self):
        graph = self.graphs[0]
        lazy = port.DiffusionDataset(graph)
        dense = torch.FloatTensor(graph.toarray())
        for index in range(3):
            self.assert_tensor_equal(lazy[index][0], dense[index])
        # Include an unobserved edge to check that candidate selection is NOT
        # accidentally limited to historical positives (unlike ReCoBR).
        rows, columns = np.array([0, 1, 2]), np.array([3, 0, 1])
        ub = sp.coo_matrix((np.ones(3), (rows, columns)), shape=(3, 4), dtype=np.float32)
        a = sp.bmat([[sp.csr_matrix((3, 3)), ub], [ub.T, sp.csr_matrix((4, 4))]])
        a = a + sp.eye(7)
        d = np.power(np.array(a.sum(axis=1)), -.5).flatten()
        d[np.isinf(d)] = 0
        expected = torch.FloatTensor((sp.diags(d).dot(a).dot(sp.diags(d))).toarray())
        self.assert_tensor_equal(expected, port.build_rebuilt_graph(rows, columns, 3, 4, "cpu").to_dense())

    def test_metrics_and_negative_sampling_match_recobr(self):
        # Import the frozen root evaluator only for the comparison test.
        sys.path.insert(0, str(ROOT))
        try:
            reference = load_module("recobr_evaluation_test", ROOT / "train.py")
            root_data = load_module("recobr_data_test", ROOT / "utility.py")
        finally:
            sys.path.pop(0)
        port.set_seed(2026)
        pred = torch.randn(4, 100)
        truth = torch.zeros(4, 100)
        truth[0, [3, 7]] = 1
        truth[1, [10, 15, 60]] = 1
        truth[2, 90] = 1  # fourth user has no positives and must be excluded
        blank = {name: {k: [0, 0] for k in [10, 20, 40, 80]} for name in ["recall", "ndcg"]}
        self.assertEqual(port.get_metrics(copy.deepcopy(blank), truth, pred, [10, 20, 40, 80]),
                         reference.get_metrics(copy.deepcopy(blank), truth, pred, [10, 20, 40, 80]))
        pairs = np.array([[0, 0], [1, 1], [2, 2]])
        new = port.BundleTrainDataset(pairs, self.graphs[0], 4, 1)
        old = root_data.BundleTrainDataset(pairs, self.graphs[0], 4, 1)
        for index in range(3):
            port.set_seed(index)
            expected = old[index]
            port.set_seed(index)
            actual = new[index]
            for left, right in zip(expected, actual):
                self.assert_tensor_equal(left, right)

    def test_evaluation_masks_training_edges_and_excludes_empty_users(self):
        scores = torch.zeros(2, 100)
        scores[0, :80] = 100  # Without masking, all Top-20/40 slots are observed bundles.
        scores[0, 90] = 50
        truth = torch.zeros(2, 100)
        truth[0, 90] = 1  # user 1 has no evaluation positives
        mask = torch.zeros(2, 100)
        mask[0, :80] = 1

        class FixedScores:
            def eval(self):
                pass

            def propagate(self, test=False):
                return None

            def evaluate(self, representations, users):
                return scores[users].clone()

        result = port.evaluate(FixedScores(), [(torch.arange(2), truth, mask)],
                               {"topk": [20, 40], "device": torch.device("cpu")})
        self.assertEqual(result, {"recall": {20: 1.0, 40: 1.0}, "ndcg": {20: 1.0, 40: 1.0}})

    def make_tiny_data(self, root):
        directory = root / "data/iFashion"
        directory.mkdir(parents=True)
        (directory / "iFashion_data_size.txt").write_text("4\t100\t8\n")
        relations = {
            "user_bundle_train": [(u, u * 3 + j) for u in range(4) for j in range(3)],
            "user_bundle_tune": [(u, b) for u in range(4) for b in range(20, 100)],
            "user_bundle_test": [(u, b) for u in range(4) for b in range(12, 20)],
            "user_item": [(u, i) for u in range(4) for i in range(8)],
            "bundle_item": [(b, b % 8) for b in range(100)],
        }
        for name, pairs in relations.items():
            (directory / (name + ".txt")).write_text("".join(f"{a}\t{b}\n" for a, b in pairs))
        conf = copy.deepcopy(self.base)
        conf.update(data_path=str(root / "data"), batch_size_train=4, batch_size_test=2,
                    epochs=2, test_interval=1, steps=3, dims=[12], embedding_size=8)
        path = root / "config.yaml"
        path.write_text(yaml.safe_dump({"iFashion": conf}), encoding="utf-8")
        return path

    def test_snapshot_workers_and_config_validation(self):
        with tempfile.TemporaryDirectory(prefix="dcbr-config-") as temp:
            root = Path(temp)
            path = self.make_tiny_data(root)
            configs = yaml.safe_load(path.read_text(encoding="utf-8"))
            configs["iFashion"]["num_workers"] = 2
            path.write_text(yaml.safe_dump(configs), encoding="utf-8")
            conf = port.load_config(path, "iFashion")
            self.assertEqual(conf["num_workers"], 2)
            # Top-40 is not required for the new checkpoint criterion.
            top20_only = copy.deepcopy(conf)
            top20_only["topk"] = [20]
            port.validate_config(top20_only)
            for key, value in [("num_workers", -1), ("tau", 0), ("topk", [10, 40]),
                               ("sampling_steps", 100), ("lambda_1", float("nan"))]:
                with self.subTest(key=key):
                    invalid = copy.deepcopy(conf)
                    invalid[key] = value
                    with self.assertRaises(ValueError):
                        port.validate_config(invalid)
            # The effective worker count is checked at the actual loader call;
            # stop before any multiprocessing workers or model are created.
            for options, expected in [([], 2), (["--num-workers", "0"], 0)]:
                args = port.get_cmd(["--config", str(path), "--output-dir", str(root / "runs"),
                                     "--record-file", str(root / "records.txt")] + options)
                actual = port.load_config(path, "iFashion", args)
                actual["device"] = torch.device("cpu")
                with patch.object(port, "Datasets", side_effect=RuntimeError("stop after worker selection")) as datasets:
                    with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "worker selection"):
                        port.run_training(actual, args)
                self.assertEqual(datasets.call_args.args[1], expected)

    def test_complete_cli_reload_and_append(self):
        with tempfile.TemporaryDirectory(prefix="dcbr-reproduction-") as temp:
            directory = Path(temp)
            config = self.make_tiny_data(directory)
            record_path = directory / "records.txt"
            command = [sys.executable, str(HERE / "train.py"), "--config", str(config),
                       "--device", "cpu", "--quiet", "--num-workers", "0", "--note", "tiny integration test",
                       "--output-dir", str(directory / "runs"), "--record-file", str(record_path)]
            # Invoke once from the repository root and once from DCBR: default
            # paths must not silently point at the deleted raw datasets.
            for cwd in (ROOT, HERE):
                result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, encoding="utf-8", timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            results = sorted((directory / "runs/iFashion").glob("*/result.json"))
            self.assertEqual(len(results), 2)
            for path in results:
                result = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(result["status"], "SUCCESS")
                self.assertTrue(result["completed"])
                args = port.get_cmd(["--num-workers", "0"])
                with contextlib.redirect_stdout(io.StringIO()):
                    restored = port.evaluate_checkpoint(result["artifacts"]["checkpoint_model"], args, torch.device("cpu"))
                self.assertEqual(json.loads(json.dumps(restored["metrics"])), result["result"]["metrics"])
                self.assertEqual(restored["selection_metrics"], result["result"]["selection_metrics"])
                self.assertEqual(restored["selection_rule"], port.SELECTION_RULE)
                self.assertNotIn("selection_score", result["result"])
                # Saved actual-config snapshots must be valid inputs too.
                replay = port.load_config(result["artifacts"]["config"], "iFashion")
                self.assertEqual(replay["dims"], [12])
            text = record_path.read_text(encoding="utf-8")
            self.assertEqual(text.count("EXECUTION_ID:"), 2)
            self.assertEqual(text.count("END_EXECUTION:"), 2)
            # Determinism under fixed config, seed and CPU execution.
            first, second = [json.loads(p.read_text(encoding="utf-8")) for p in results]
            self.assertEqual(first["result"], second["result"])
            # Even a format-only data change must be detected before reevaluation.
            data_path = directory / "data/iFashion/user_bundle_test.txt"
            data_path.write_bytes(data_path.read_bytes() + b"\n")
            args = port.get_cmd(["--num-workers", "0"])
            with self.assertRaisesRegex(ValueError, "data fingerprints"):
                port.evaluate_checkpoint(first["artifacts"]["checkpoint_model"], args, torch.device("cpu"))

    def test_runtime_failure_is_recorded(self):
        for exception, status in [(RuntimeError, "FAILED"), (KeyboardInterrupt, "INTERRUPTED")]:
            with self.subTest(status=status), tempfile.TemporaryDirectory(prefix="dcbr-failure-") as temp:
                root = Path(temp)
                path = self.make_tiny_data(root)
                args = port.get_cmd(["--config", str(path), "--num-workers", "0", "--quiet",
                                     "--output-dir", str(root / "runs"), "--record-file", str(root / "failed.txt"),
                                     "--result-json", str(root / "external-result.json")])
                conf = port.load_config(path, "iFashion", args)
                conf["device"] = torch.device("cpu")
                with patch.object(port, "train_diffusion_epoch", side_effect=exception("simulated failure")):
                    with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(exception, "simulated failure"):
                        port.run_training(conf, args)
                result = json.loads(next((root / "runs/iFashion").glob("*/result.json")).read_text())
                self.assertEqual(result["status"], status)
                self.assertFalse(result["completed"])
                self.assertEqual(result, json.loads((root / "external-result.json").read_text()))
                self.assertIn("simulated failure", (root / "failed.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
