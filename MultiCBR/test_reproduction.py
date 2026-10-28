"""Necessary author parity, shared protocol and isolated CLI checks; no formal GPU runs."""
import ast
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import scipy.sparse as sp
import torch
import yaml

from MultiCBR import configuration as config
from MultiCBR import data, model, train

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.dont_write_bytecode = True  # Never create bytecode inside protected raw/.


def load_reference(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(directory, dataset="Tiny"):
    directory = Path(directory)
    target = directory / "data" / dataset
    target.mkdir(parents=True)
    (target / (dataset + "_data_size.txt")).write_text("5\t100\t8\n", encoding="utf-8")
    training = [(u, 3*u+j) for u in range(4) for j in range(3)]
    testing = [(u, 95+u) for u in range(5)]
    excluded = set(training + testing)
    relations = {
        "user_bundle_train": training,
        "user_bundle_tune": [(u, b) for u in range(5) for b in range(100) if (u, b) not in excluded],
        "user_bundle_test": testing,
        "user_item": [(u, i) for u in range(5) for i in range(8)],
        "bundle_item": [(b, b % 8) for b in range(100)],
    }
    for name, pairs in relations.items():
        # Include a final line without a newline to catch author s[:-1] parsing pitfalls.
        (target / (name + ".txt")).write_text("\n".join(f"{a}\t{b}" for a, b in pairs), encoding="utf-8")
    conf = copy.deepcopy(config.read_yaml(HERE / "config.yaml")["Youshu"])
    conf.update(data_path=str(directory / "data"), embedding_sizes=[4], batch_size_train=4,
                batch_size_test=2, epochs=2, test_interval=1, num_workers=0)
    path = directory / "config.yaml"
    path.write_text(yaml.safe_dump({dataset: conf}), encoding="utf-8")
    return path, config.load_config(path, dataset)


class ReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            cls.raw = load_reference(HERE / "raw/models/MultiCBR.py", "multicbr_author_model")
        cls.graphs = [sp.csr_matrix(np.asarray(x, dtype=np.float32)) for x in (
            [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]],
            [[1, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 1, 1]],
            [[1, 0, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 1, 1]],
        )]
        cls.batch = [torch.tensor([[0], [1], [2], [0]]),
                     torch.tensor([[0, 3], [1, 0], [2, 1], [1, 3]])]

    def test_all_author_dataset_settings_and_singleton_snapshot_validation(self):
        originals = config.read_yaml(HERE / "raw/config.yaml")
        current = config.read_yaml(HERE / "config.yaml")
        self.assertEqual(set(current), {"Youshu", "NetEase", "iFashion"})
        for name in originals:
            self.assertEqual({k: v for k, v in current[name].items() if k not in {"seed", "num_workers"}},
                             originals[name])
            self.assertEqual(current[name]["seed"], 0)
        with tempfile.TemporaryDirectory() as temporary:
            path, conf = fixture(temporary)
            for overrides in ({"lrs": [.001, .002]}, {"seed": True}, {"c_temps": [0]},
                              {"topk": [20, 40]}, {"num_workers": -1}, {"unknown": 1}):
                with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                    config.validate(dict(conf, **overrides))
            runtime = config.runtime_config(conf, "Tiny", (5, 100, 8), "0", "cpu")
            path.write_text(yaml.safe_dump({"Tiny": runtime}), encoding="utf-8")
            self.assertEqual(config.load_config(path, "Tiny"), conf)
            runtime["lr"] = 1
            path.write_text(yaml.safe_dump({"Tiny": runtime}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "scalar disagrees"):
                config.load_config(path, "Tiny")

    def test_two_update_author_parity_for_noise_ed_md_and_op(self):
        base = config.read_yaml(HERE / "config.yaml")["Youshu"]
        conf = config.runtime_config(base, "Youshu", (3, 4, 5), "0", "cpu")
        conf.update(embedding_size=8, lr=.001)
        for augmentation in ("Noise", "ED", "MD", "OP"):
            settings = dict(conf, aug_type=augmentation)
            if augmentation == "OP":
                settings.update(UB_ratio=0, UI_ratio=0, BI_ratio=0)

            def trajectory(module):
                train.seed_everything(42)
                instance = module.MultiCBR(copy.deepcopy(settings), self.graphs)
                optimizer = torch.optim.Adam(instance.parameters(), lr=settings["lr"], weight_decay=settings["l2_reg"])
                history = []
                for _ in range(2):
                    optimizer.zero_grad()
                    bpr, contrastive = instance(self.batch, ED_drop=augmentation == "ED")
                    loss = bpr + settings["c_lambda"] * contrastive
                    loss.backward()
                    gradients = {name: p.grad.clone() for name, p in instance.named_parameters()}
                    optimizer.step()
                    history.append((torch.stack((bpr, contrastive)).detach(), gradients))
                instance.eval()
                scores = instance.evaluate(instance.get_multi_modal_representations(test=True), torch.arange(3)).detach()
                return history, instance.state_dict(), optimizer.state_dict(), scores

            with self.subTest(augmentation=augmentation), warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                expected, actual = trajectory(self.raw), trajectory(model)
                for old, new in zip(expected[0], actual[0]):
                    torch.testing.assert_close(old[0], new[0], rtol=0, atol=0)
                    for name in old[1]:
                        self.assertTrue(torch.isfinite(new[1][name]).all())
                        torch.testing.assert_close(old[1][name], new[1][name], rtol=0, atol=0)
                for name in expected[1]:
                    torch.testing.assert_close(expected[1][name], actual[1][name], rtol=0, atol=0)
                for key, old in expected[2]["state"].items():
                    for name, value in old.items():
                        torch.testing.assert_close(value, actual[2]["state"][key][name], rtol=0, atol=0)
                torch.testing.assert_close(expected[3], actual[3], rtol=0, atol=0)

    def test_shared_metrics_including_ties_empty_users_and_original_train_mask(self):
        tree = ast.parse((ROOT / "train.py").read_text(encoding="utf-8-sig"))
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name in {"get_metrics", "get_recall", "get_ndcg"}]
        namespace = {"torch": torch}
        exec(compile(ast.Module(body=functions, type_ignores=[]), "root_metrics", "exec"), namespace)
        torch.manual_seed(42)
        scores = torch.randn(4, 100)
        scores[3] = 0
        truth = torch.zeros_like(scores)
        truth[0, [3, 7]] = 1
        truth[1, [15, 19, 60]] = 1
        truth[3, [3, 9]] = 1
        blank = {name: {k: [0, 0] for k in (10, 20, 40, 80)} for name in ("recall", "ndcg")}
        self.assertEqual(train.get_metrics(copy.deepcopy(blank), truth, scores, [10, 20, 40, 80]),
                         namespace["get_metrics"](copy.deepcopy(blank), truth, scores, [10, 20, 40, 80]))

        class Fixed:
            def eval(self):
                pass

            def get_multi_modal_representations(self, test=False):
                return None

            def evaluate(self, representations, users):
                return torch.arange(100, 0, -1, dtype=torch.float32).repeat(len(users), 1)

        positives, mask = torch.zeros(2, 100), torch.zeros(2, 100)
        positives[0, 20] = 1
        mask[0, :20] = 1
        result = train.evaluate(Fixed(), [(torch.arange(2), positives, mask)],
                                {"topk": [10, 20, 40, 80], "device": "cpu"})
        self.assertEqual(result["recall"][20], 1)
        self.assertEqual(result["ndcg"][20], 1)
        # A validation positive at bundle 20 is still a test candidate: only train is masked.
        self.assertEqual(result["ndcg"][10], 1)

    def test_dataset_preserves_inactive_ids_sampling_and_rejects_bad_inputs(self):
        raw_data = load_reference(HERE / "raw/utility.py", "multicbr_author_data")
        with tempfile.TemporaryDirectory() as temporary:
            _, conf = fixture(temporary)
            dataset = data.Datasets(conf, "Tiny")
            self.assertEqual(dataset.sizes, (5, 100, 8))
            self.assertEqual(dataset.graphs[0].getnnz(axis=1)[4], 0)
            self.assertEqual(len(dataset.val_loader.dataset), 5)
            sampler = raw_data.BundleTrainDataset({}, dataset.train_loader.dataset.pairs,
                                                  dataset.graphs[0], 100, None, None, 1)
            for index in range(12):
                np.random.seed(index)
                expected = sampler[index]
                np.random.seed(index)
                actual = dataset.train_loader.dataset[index]
                for old, new in zip(expected, actual):
                    torch.testing.assert_close(old, new, rtol=0, atol=0)
            target = Path(conf["data_path"]) / "Tiny/user_bundle_tune.txt"
            for contents, message in (("0\t0", "overlap"), ("0\t50\n0\t50", "Duplicate"), ("5\t1", "bounds")):
                target.write_text(contents, encoding="utf-8")
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    data.Datasets(conf, "Tiny")

    def test_checkpoint_updates_require_both_metrics_and_keep_the_same_epoch(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            path, conf = fixture(root)
            conf["epochs"] = 3
            path.write_text(yaml.safe_dump({"Tiny": conf}), encoding="utf-8")
            args = train.arguments(["--config", str(path), "-d", "Tiny", "--device", "cpu", "--quiet",
                                    "--output-dir", str(root / "runs"), "--record-file", str(root / "record.txt")])
            evaluations = []
            for recall, ndcg in ((.2, .1), (.19, .3), (.21, .11)):
                evaluations += [{"recall": {k: recall for k in (10,20,40,80)},
                                 "ndcg": {k: ndcg for k in (10,20,40,80)}}] * 2
            save = torch.save
            with patch.object(train, "evaluate", side_effect=evaluations), patch.object(torch, "save", wraps=save) as spy:
                result = train.run_training(conf, args, torch.device("cpu"))
                self.assertEqual([call.args[0]["result"]["best_epoch"] for call in spy.call_args_list], [0, 2])
            self.assertEqual(result["result"]["best_epoch"], 2)
            for recall, ndcg in ((.21,.1),(.2,.11),(.19,.3)):
                self.assertFalse(train.is_better_validation({"recall":{20:recall},"ndcg":{20:ndcg}},
                                                            {"recall":{20:.2},"ndcg":{20:.1}}))

    def test_cli_training_snapshot_replay_and_readonly_checkpoint_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            path, _ = fixture(root, "NetEase")
            args = ["--config", str(path), "-d", "NetEase", "--device", "cpu", "--quiet",
                    "--output-dir", str(root / "runs"), "--record-file", str(root / "record.txt"),
                    "--result-json", str(root / "extra.json")]
            self.assertEqual(train.main(args), 0)
            first = json.loads((root / "extra.json").read_text(encoding="utf-8"))
            self.assertEqual((first["status"], first["completed_epochs"]), ("SUCCESS", 2))
            self.assertTrue(first["completed"])
            self.assertEqual(len(first["sources"]), 4)
            self.assertEqual(len(first["data_files"]), 6)
            for key, expected in first["artifact_hashes"].items():
                self.assertEqual(config.file_hash(first["artifacts"][key]), expected)
            self.assertTrue(list(Path(first["artifacts"]["tensorboard"]).glob("events.*")))
            before = (root / "record.txt").read_bytes()
            evaluation_args = SimpleNamespace(data_path=None, num_workers=0, batch_size_test=None, gpu="0")
            evaluated = train.evaluate_checkpoint(first["artifacts"]["checkpoint_model"], evaluation_args,
                                                   torch.device("cpu"))
            self.assertEqual(json.loads(json.dumps(evaluated["metrics"])), first["result"]["metrics"])
            self.assertEqual((root / "record.txt").read_bytes(), before)
            # Replay the actual snapshot, with every effective parameter retained.
            replay_args = args.copy()
            replay_args[replay_args.index("--config")+1] = first["artifacts"]["config"]
            self.assertEqual(train.main(replay_args), 0)
            second = json.loads((root / "extra.json").read_text(encoding="utf-8"))
            self.assertNotEqual(first["execution_id"], second["execution_id"])
            self.assertEqual(first["result"], second["result"])
            self.assertEqual((root / "record.txt").read_text(encoding="utf-8").count("END_EXECUTION:"), 2)
            self.assertEqual(len(list((root / "runs/NetEase").iterdir())), 2)
            # Exercise the native CLI from a different working directory/process.
            command = [sys.executable, "-B", str(HERE / "train.py"), "--evaluate",
                       first["artifacts"]["checkpoint_model"], "--device", "cpu"]
            process = subprocess.run(command, cwd=root, capture_output=True, text=True, encoding="utf-8", timeout=60)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            self.assertEqual(json.loads(process.stdout)["metrics"], first["result"]["metrics"])
            count = len(list((root / "runs/NetEase").iterdir()))
            readonly = ["--config", str(path), "-d", "NetEase", "--check-config"]
            self.assertEqual(train.main(readonly), 0)
            self.assertEqual(len(list((root / "runs/NetEase").iterdir())), count)
            after = (root / "record.txt").read_bytes()
            (root / "data/NetEase/user_item.txt").write_text("0\t0", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fingerprints"):
                train.evaluate_checkpoint(first["artifacts"]["checkpoint_model"], evaluation_args, torch.device("cpu"))
            self.assertEqual((root / "record.txt").read_bytes(), after)

    def test_failed_and_interrupted_runs_preserve_status_and_partial_best(self):
        for error, status in ((RuntimeError, "FAILED"), (KeyboardInterrupt, "INTERRUPTED")):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
                root = Path(temporary)
                path, conf = fixture(root)
                args = train.arguments(["--config", str(path), "-d", "Tiny", "--device", "cpu", "--quiet",
                                        "--output-dir", str(root / "runs"), "--record-file", str(root / "record.txt"),
                                        "--result-json", str(root / "extra.json")])
                original = train.train_epoch
                def stop_on_second(*arguments):
                    if arguments[4] == 1:
                        raise error("verification interruption")
                    return original(*arguments)
                with patch.object(train, "train_epoch", side_effect=stop_on_second), self.assertRaises(error):
                    train.run_training(conf, args, torch.device("cpu"))
                record = json.loads((root / "extra.json").read_text(encoding="utf-8"))
                self.assertEqual(record["status"], status)
                self.assertFalse(record["completed"])
                self.assertEqual(record["completed_epochs"], 1)
                self.assertEqual(record["result"]["best_epoch"], 0)
                self.assertTrue(Path(record["artifacts"]["checkpoint_model"]).is_file())
                self.assertIn(status, (root / "record.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
