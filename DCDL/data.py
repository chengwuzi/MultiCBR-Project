"""Original-ID relations, full-ranking inputs and exact sparse preprocessing."""
import hashlib
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def to_torch(matrix):
    coo = matrix.tocoo()
    return torch.sparse_coo_tensor(
        torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64)),
        torch.from_numpy(coo.data.astype(np.float32)), coo.shape, check_invariants=True).coalesce()


class Relations:
    def __init__(self, conf):
        self.directory = Path(conf["data_path"]) / conf["dataset"]
        size_file = self.directory / f'{conf["dataset"]}_data_size.txt'
        sizes = np.loadtxt(size_file, dtype=np.int64).reshape(-1)
        if sizes.size != 3 or np.any(sizes <= 0):
            raise ValueError("Size file must declare positive user, bundle, item counts")
        self.num_users, self.num_bundles, self.num_items = map(int, sizes)
        self.fingerprints = {size_file.name: file_hash(size_file)}
        self.ub, self.pairs = self.read("user_bundle_train.txt", (self.num_users, self.num_bundles))
        self.val, _ = self.read("user_bundle_tune.txt", self.ub.shape)
        self.test, _ = self.read("user_bundle_test.txt", self.ub.shape)
        self.ui, _ = self.read("user_item.txt", (self.num_users, self.num_items))
        self.bi, _ = self.read("bundle_item.txt", (self.num_bundles, self.num_items))
        for left, right in ((self.ub, self.val), (self.ub, self.test), (self.val, self.test)):
            if left.multiply(right).nnz:
                raise ValueError("UB train/validation/test sets overlap")
        if np.any(np.diff(self.ub.indptr) == self.num_bundles):
            raise ValueError("A training user has no unobserved bundle for negative sampling")
        if max(conf["topk"]) > self.num_bundles:
            raise ValueError("topk exceeds the declared bundle count")
        self.statistics = dict(num_users=self.num_users, num_bundles=self.num_bundles,
                               num_items=self.num_items, train=self.ub.nnz, val=self.val.nnz,
                               test=self.test.nnz, ui=self.ui.nnz, bi=self.bi.nnz,
                               train_active_users=int(np.count_nonzero(np.diff(self.ub.indptr))))

    def read(self, name, shape):
        path = self.directory / name
        pairs = np.loadtxt(path, dtype=np.int64, delimiter="\t", ndmin=2, comments=None)
        if pairs.ndim != 2 or pairs.shape[1] != 2 or not len(pairs):
            raise ValueError(f"{name} must contain nonempty tab-separated integer pairs")
        if np.any(pairs < 0) or np.any(pairs[:, 0] >= shape[0]) or np.any(pairs[:, 1] >= shape[1]):
            raise ValueError(f"{name} has an ID outside the declared entity counts")
        graph = sp.csr_matrix((np.ones(len(pairs), dtype=np.float32), pairs.T), shape=shape)
        if graph.nnz != len(pairs):
            raise ValueError(f"{name} contains duplicate interactions")
        self.fingerprints[name] = file_hash(path)
        return graph, pairs


class TrainPairs(Dataset):
    """One uniform negative; reject only the original UB training positives."""
    def __init__(self, relations):
        self.pairs = relations.pairs
        self.observed = [set(relations.ub.indices[relations.ub.indptr[u]:relations.ub.indptr[u+1]])
                         for u in range(relations.num_users)]
        self.num_bundles = relations.num_bundles

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        user, positive = self.pairs[index]
        negative = np.random.randint(self.num_bundles)
        while negative in self.observed[user]:
            negative = np.random.randint(self.num_bundles)
        return int(user), int(positive), int(negative)


def threshold_product(left, right, threshold, chunk_size):
    """Exact strict threshold; retain qualifying diagonal elements."""
    blocks = []
    for start in range(0, left.shape[0], chunk_size):
        block = (left[start:start + chunk_size] @ right).tocsr()
        block.data = (block.data > threshold).astype(np.float32)
        block.eliminate_zeros()
        blocks.append(block)
    return sp.vstack(blocks, format="csr")


def build_graphs(relations, conf, report=lambda message: None):
    # Integer multiplication avoids float/count overflow when constructing co-occurrences.
    x = relations.ub.astype(np.int64)
    chunk = conf["graph_chunk_size"]
    report("Building exact thresholded user-user relations")
    p = threshold_product(x, x.T.tocsr(), conf["k1"], chunk)
    report("Building exact thresholded bundle-bundle relations")
    q = threshold_product(x.T.tocsr(), x, conf["k1"], chunk)
    a = sp.bmat([[p, relations.ub], [relations.ub.T, q]], format="csr", dtype=np.float32)
    # R = A J for an orthogonal column permutation J, hence R R^T = A^2.
    report(f"Computing global Frobenius norm of A^2 in row blocks; A.nnz={a.nnz}")
    squared = 0.0
    a64 = a.astype(np.float64)
    for start in range(0, a.shape[0], chunk):
        block = (a64[start:start + chunk] @ a64).tocsr()
        squared += float(np.dot(block.data, block.data))
    normalizer = squared ** 0.5
    if not np.isfinite(normalizer) or normalizer <= 0:
        raise ValueError("Relationship reinforcement has a zero or nonfinite Frobenius norm")
    # Self-loops apply to the GAT's original UB graph, independently of thresholded P/Q.
    base = sp.bmat([[None, relations.ub], [relations.ub.T, None]], format="csr")
    base = base + sp.eye(base.shape[0], dtype=np.float32, format="csr")
    edge = base.tocoo()
    graphs = dict(
        relation=to_torch(a), relation_norm=torch.tensor(normalizer, dtype=torch.float32),
        gat_edges=torch.from_numpy(np.vstack((edge.row, edge.col)).astype(np.int64)),
    )
    for name, matrix in (("ui", relations.ui), ("bi", relations.bi)):
        row_degree = np.asarray(matrix.sum(axis=1)).ravel()
        co_degree = np.asarray(matrix.T @ row_degree).ravel()
        graphs[name] = to_torch(matrix)
        graphs[name + "_t"] = to_torch(matrix.T)
        graphs[name + "_row_degree"] = torch.tensor(row_degree, dtype=torch.float32).clamp_min(1)
        graphs[name + "_co_degree"] = torch.tensor(co_degree, dtype=torch.float32).clamp_min(1)
    report(f"Static graphs ready; ||R R^T||_F={normalizer:.8g}")
    return graphs
