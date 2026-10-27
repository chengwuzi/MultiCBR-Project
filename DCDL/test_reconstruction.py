"""Small numerical checks and an isolated single-run integration test (no real tuning)."""
from copy import deepcopy
import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import scipy.sparse as sp
import torch
from torch.nn import functional as F
import yaml

from DCDL.configuration import DEFAULTS, IMPLEMENTATION, validate
from DCDL.data import Relations, TrainPairs, build_graphs
from DCDL.model import CollaborativeFusionDiffuser, DCDL, SparseGAT, _StaticSparseMM, sinusoidal_embedding
from DCDL.train import evaluate, evaluate_checkpoint, improves, main


def fixture(directory):
    """Includes one user without UB history and one item without any history."""
    directory = Path(directory)
    path = directory / "Tiny"
    path.mkdir()
    (path / "Tiny_data_size.txt").write_text("4\t100\t7\n", encoding="utf-8")
    pairs = {
        "user_bundle_train.txt": [(0, 0), (0, 1), (1, 1), (1, 2), (2, 3)],
        "user_bundle_tune.txt": [],
        "user_bundle_test.txt": [(0, 9), (1, 10), (2, 11), (3, 12)],
        "user_item.txt": [(0, 0), (0, 1), (1, 1), (2, 2), (3, 3)],
        "bundle_item.txt": [(b, b % 6) for b in range(100)],
    }
    excluded = set(pairs["user_bundle_train.txt"] + pairs["user_bundle_test.txt"])
    # Enough validation positives to guarantee R@20 > 0 regardless of initialization.
    pairs["user_bundle_tune.txt"] = [(u, b) for u in range(4) for b in range(100) if (u, b) not in excluded]
    for name, values in pairs.items():
        np.savetxt(path / name, np.asarray(values), fmt="%d", delimiter="\t")
    conf = deepcopy(DEFAULTS)
    conf.update(dataset="Tiny", data_path=str(directory), embedding_size=4, k1=1,
                diffusion_steps=3, alpha_1=4.0, lambda_1=0.15,
                dropout=0.0, batch_size_train=3, batch_size_test=2,
                graph_chunk_size=2, epochs=2, test_interval=1)
    return validate(conf)


class FormulaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(9)
        np.random.seed(9)
        self.temporary = tempfile.TemporaryDirectory(prefix="dcdl-test-")
        self.addCleanup(self.temporary.cleanup)
        self.conf = fixture(self.temporary.name)
        self.data = Relations(self.conf)
        self.graphs = build_graphs(self.data, self.conf)
        self.model = DCDL(self.conf, (4, 100, 7), self.graphs, 5)

    def test_relationship_formula_and_global_norm(self):
        x = self.data.ub.toarray().astype(np.float64)
        p = (x @ x.T > self.conf["k1"]).astype(float)
        q = (x.T @ x > self.conf["k1"]).astype(float)
        r = np.block([[x, p], [q, x.T]])
        gram = r @ r.T
        a = self.graphs["relation"].to_dense().double()
        np.testing.assert_allclose((a @ a).numpy(), gram, rtol=0, atol=0)
        self.assertAlmostEqual(self.graphs["relation_norm"].item(), np.linalg.norm(gram), places=5)
        # Exact threshold, not >=; retained diagonal when its count exceeds k1.
        self.assertEqual(a[0, 0].item(), 1)
        self.assertEqual(a[2, 2].item(), 0)
        embeddings = torch.randn(104, 4, dtype=torch.float64, requires_grad=True)
        sparse = self.graphs["relation"].double()
        actual = torch.sparse.mm(sparse, torch.sparse.mm(sparse, embeddings)) / np.linalg.norm(gram)
        expected = torch.tensor(gram / np.linalg.norm(gram)) @ embeddings
        torch.testing.assert_close(actual, expected)
        actual_grad, = torch.autograd.grad(actual.square().sum(), embeddings, retain_graph=True)
        expected_grad, = torch.autograd.grad(expected.square().sum(), embeddings)
        torch.testing.assert_close(actual_grad, expected_grad)

    def test_item_factorization_outputs_and_gradients(self):
        self.model.double()
        users, bundles = self.model.item_view()
        y = torch.tensor(self.data.ui.toarray(), dtype=torch.float64)
        z = torch.tensor(self.data.bi.toarray(), dtype=torch.float64)
        cu, cb = y.T @ y, z.T @ z
        cu = cu / cu.sum(1, keepdim=True).clamp_min(1)
        cb = cb / cb.sum(1, keepdim=True).clamp_min(1)
        e = self.model.item_embedding.weight
        items = e + F.relu(cu @ e) + F.relu(cb @ e)
        expected_u = y @ items / y.sum(1, keepdim=True).clamp_min(1)
        expected_b = z @ items / z.sum(1, keepdim=True).clamp_min(1)
        torch.testing.assert_close(users, expected_u)
        torch.testing.assert_close(bundles, expected_b)
        g1, = torch.autograd.grad(users.square().sum() + bundles.square().sum(), e, retain_graph=True)
        g2, = torch.autograd.grad(expected_u.square().sum() + expected_b.square().sum(), e)
        torch.testing.assert_close(g1, g2)
        torch.testing.assert_close(items[-1], e[-1])

    def test_sparse_gat_matches_dense_attention_and_gradients(self):
        layer = SparseGAT(4, 0.0, 0.2).double()
        values = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)
        edges = torch.tensor([[0, 0, 1, 1, 2, 2, 3], [0, 1, 0, 1, 1, 2, 3]])
        output = layer(values, edges)
        h = layer.projection(values)
        score = F.leaky_relu((h * layer.attention[0]).sum(1)[:, None] +
                             (h * layer.attention[1]).sum(1)[None, :], 0.2)
        allowed = torch.zeros(4, 4, dtype=torch.bool)
        allowed[edges[0], edges[1]] = True
        expected = F.elu(score.masked_fill(~allowed, -torch.inf).softmax(1) @ h)
        torch.testing.assert_close(output, expected)
        parameters = (values, layer.projection.weight, layer.attention)
        actual_grad = torch.autograd.grad(output.square().sum(), parameters, retain_graph=True)
        expected_grad = torch.autograd.grad(expected.square().sum(), parameters)
        for actual, reference in zip(actual_grad, expected_grad):
            torch.testing.assert_close(actual, reference)

    def test_diffusion_eq17_and_finite_first_step(self):
        diffuser = CollaborativeFusionDiffuser(self.conf)
        x = torch.randn(4, 4, requires_grad=True)
        fused = torch.randn(4, 4, requires_grad=True)
        t = torch.tensor([1, 2, 3, 1])
        noise = torch.randn_like(x)
        middle, loss = diffuser(x, fused, t, noise)
        beta = torch.linspace(self.conf["beta_start"], self.conf["beta_end"], 3, dtype=torch.float64)
        alpha = 1 - beta
        abar = alpha.cumprod(0)
        previous = torch.cat((torch.ones(1, dtype=torch.float64), abar[:-1]))
        posterior = beta * (1 - previous) / (1 - abar)
        posterior[0] = beta[0]
        reference = (previous / (2 * posterior))[t - 1].float() * (
            x - alpha.sqrt()[t - 1, None].float() * middle).square().sum(1)
        torch.testing.assert_close(loss, reference)
        self.assertTrue(torch.isfinite(diffuser.loss_weights).all())
        loss.sum().backward()
        self.assertGreater(x.grad.abs().sum().item(), 0)
        self.assertGreater(fused.grad.abs().sum().item(), 0)
        self.assertEqual(sinusoidal_embedding(t, 5).shape, (4, 5))

    def test_static_sparse_backward_and_cache_dtype_conversion(self):
        dense = torch.tensor([[1., 0., 2.], [0., 0., 0.]], dtype=torch.float64)
        features = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
        a, at = dense.to_sparse_csr(), dense.T.to_sparse_csr()
        self.assertTrue(torch.autograd.gradcheck(lambda x: _StaticSparseMM.apply(x, a, at), (features,)))
        first = self.model.item_view()
        self.assertTrue(self.model._cpu_sparse_cache)
        self.model.double()
        second = self.model.item_view()
        self.assertEqual(second[0].dtype, torch.float64)
        torch.testing.assert_close(first[0].double(), second[0], rtol=1e-5, atol=1e-6)
        for original, cached in self.model._cpu_sparse_cache.values():
            self.assertEqual(original.dtype, cached.dtype)
            self.assertEqual(cached.dtype, torch.float64)

    def test_joint_loss_scaling_and_all_parameter_gradients(self):
        self.model.train()
        users, pos, neg = torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([4, 5])
        steps, noise = torch.tensor([1, 2, 3, 1]), torch.randn(4, 4)
        loss, parts = self.model(users, pos, neg, steps, noise)
        fu, fb, bu = self.model.encode()
        middle, per_user = self.model.diffuser(bu, fu, steps, noise)
        boundary = (middle[users] * self.model.boundary_weight(users)).sum(1)
        expected_uib = (F.softplus(boundary - (fu[users] * fb[pos]).sum(1)) +
                        4 * F.softplus((fu[users] * fb[neg]).sum(1) - boundary)).mean()
        torch.testing.assert_close(parts["uib"], expected_uib)
        torch.testing.assert_close(parts["diffusion"], per_user.sum() / 5)
        torch.testing.assert_close(loss, expected_uib + 0.15 * parts["diffusion"] + 0.0001 * parts["l2"])
        loss.backward()
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(self.model.boundary_weight.weight.grad[:2].abs().sum().item(), 0)

    def test_uib_gradient_direction(self):
        positive, negative, boundary = [torch.tensor(0.3, requires_grad=True) for _ in range(3)]
        loss = F.softplus(boundary - positive) + 4 * F.softplus(negative - boundary)
        loss.backward()
        self.assertLess(positive.grad.item(), 0)
        self.assertGreater(negative.grad.item(), 0)

    def test_ranking_original_train_mask_and_positive_user_denominator(self):
        relations = SimpleNamespace(num_users=3,
            ub=sp.csr_matrix(([1.0], ([0], [0])), shape=(3, 100), dtype=np.float32),
            val=sp.csr_matrix(([1.0], ([0], [1])), shape=(3, 100), dtype=np.float32),
            test=sp.csr_matrix(([1.0], ([0], [11])), shape=(3, 100), dtype=np.float32))
        users = torch.ones(3, 1)
        bundles = torch.arange(100, 0, -1, dtype=torch.float32)[:, None]
        val = evaluate(self.model, relations, self.conf, "val", (users, bundles))
        test = evaluate(self.model, relations, self.conf, "test", (users, bundles))
        self.assertEqual(val["recall@10"], 1)
        self.assertEqual(val["ndcg@10"], 1)
        self.assertEqual(test["recall@10"], 0)  # Validation positive must NOT also be masked.
        self.assertEqual(test["recall@20"], 1)
        self.assertAlmostEqual(test["ndcg@20"], 1 / np.log2(12), places=6)

    def test_selection_uses_same_saved_pair(self):
        saved = {"recall@20": 0.3, "ndcg@20": 0.2}
        self.assertFalse(improves({"recall@20": 0.4, "ndcg@20": 0.19}, saved))
        self.assertFalse(improves({"recall@20": 0.3, "ndcg@20": 0.21}, saved))
        self.assertTrue(improves({"recall@20": 0.31, "ndcg@20": 0.21}, saved))

    def test_metrics_match_project_evaluator_including_ties(self):
        # Read only the three existing metric functions, without importing the
        # root training entry point or its unrelated model dependencies.
        source = Path(__file__).resolve().parents[1] / "train.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        names = {"get_metrics", "get_recall", "get_ndcg"}
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        self.assertEqual(len(functions), 3)
        namespace = {"torch": torch}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
        users = torch.randn(4, 4)
        bundles = torch.randn(100, 4)
        users[3] = 0  # All scores equal: separate topk(K) calls matter here.
        target = self.data.val.copy().tolil()
        target[2] = 0  # Users without positives must not enter the denominator.
        target = target.tocsr()
        relations = SimpleNamespace(num_users=4, ub=self.data.ub, val=target)
        actual = evaluate(self.model, relations, self.conf, "val", (users, bundles))
        totals = {metric: {k: [0, 0] for k in self.conf["topk"]} for metric in ("recall", "ndcg")}
        for start in range(0, 4, self.conf["batch_size_test"]):
            stop = start + self.conf["batch_size_test"]
            scores = users[start:stop] @ bundles.T - 1e8 * torch.from_numpy(self.data.ub[start:stop].toarray())
            truth = torch.from_numpy(target[start:stop].toarray())
            namespace["get_metrics"](totals, truth, scores, self.conf["topk"])
        expected = {f"{metric}@{k}": total / count for metric, values in totals.items()
                    for k, (total, count) in values.items()}
        self.assertEqual(actual, expected)

    def test_negative_samples_and_overlap_validation(self):
        dataset = TrainPairs(self.data)
        for _ in range(20):
            for i in range(len(dataset)):
                user, positive, negative = dataset[i]
                self.assertEqual(self.data.ub[user, positive], 1)
                self.assertEqual(self.data.ub[user, negative], 0)
        (self.data.directory / "user_bundle_tune.txt").write_text("0\t0\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "overlap"):
            Relations(self.conf)

    def test_config_rejects_invalid_or_unknown_values(self):
        for change in ({"beta_start": 0}, {"lambda_1": float("nan")}, {"topk": [10, 40]},
                       {"surprise": 1}, {"neg_num": 2}, {"batch_size_train": True}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate(dict(self.conf, **change))

    def test_isolated_training_checkpoint_and_readonly_reevaluation(self):
        root = Path(self.temporary.name)
        config = root / "config.yaml"
        entry = dict(self.conf)
        entry.pop("dataset")
        config.write_text(yaml.safe_dump({"Tiny": entry}), encoding="utf-8")
        record, extra = root / "record.txt", root / "extra.json"
        self.assertEqual(main(["--config", str(config), "-d", "Tiny", "--device", "cpu",
                               "--output-dir", str(root / "outputs"), "--record-file", str(record),
                               "--result-json", str(extra), "--note", "integration validation only", "--quiet"]), 0)
        result = json.loads(extra.read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["completed_epochs"], 2)
        self.assertEqual(result["implementation"], IMPLEMENTATION)
        self.assertEqual(result["data_statistics"]["train_active_users"], 3)
        checkpoint = Path(result["checkpoint"])
        self.assertTrue(checkpoint.is_file())
        before = record.read_bytes()
        args = SimpleNamespace(data_path=None, batch_size_test=None)
        restored = evaluate_checkpoint(checkpoint, args, torch.device("cpu"))
        self.assertEqual(restored["metrics"], result["metrics"])
        self.assertEqual(record.read_bytes(), before)
        self.assertTrue(restored["source_files_match"])
        with self.assertRaisesRegex(ValueError, "batch_size_test"):
            evaluate_checkpoint(checkpoint, SimpleNamespace(data_path=None, batch_size_test=0), torch.device("cpu"))
        # Read-only config inspection cannot create run artifacts or append records.
        run_count = len(list((root / "outputs" / "Tiny").iterdir()))
        self.assertEqual(main(["--config", str(config), "-d", "Tiny", "--check-config"]), 0)
        self.assertEqual(len(list((root / "outputs" / "Tiny").iterdir())), run_count)
        self.assertEqual(record.read_bytes(), before)
        (self.data.directory / "user_item.txt").write_text("0\t6\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "contents differ"):
            evaluate_checkpoint(checkpoint, args, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
