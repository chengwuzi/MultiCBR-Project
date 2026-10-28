"""Original-ID relations and ReCoBR-compatible sampling; no raw imports."""
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset, DataLoader

if __package__:
    from .configuration import file_hash
else:
    from configuration import file_hash


class BundleTrainDataset(Dataset):
    def __init__(self, pairs, graph, num_bundles):
        self.pairs, self.graph, self.num_bundles = pairs, graph, num_bundles
        active = np.unique(pairs[:, 0])
        if np.any(graph.getnnz(axis=1)[active] >= num_bundles):
            raise ValueError("A training user has no unobserved bundle for negative sampling")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        user, positive = self.pairs[index]
        # Match author's and root ReCoBR's NumPy rejection sampler exactly.
        while True:
            candidate = np.random.randint(self.num_bundles)
            if self.graph[user, candidate] == 0:
                return torch.LongTensor([user]), torch.LongTensor([positive, candidate])


class BundleTestDataset(Dataset):
    def __init__(self, truth, original_train):
        self.truth, self.original_train = truth, original_train

    def __len__(self):
        return self.truth.shape[0]

    def __getitem__(self, index):
        return (index, torch.from_numpy(self.truth[index].toarray()).squeeze(0),
                torch.from_numpy(self.original_train[index].toarray()).squeeze(0))


def worker_seed(worker_id):
    import random
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


class Datasets:
    def __init__(self, conf, dataset):
        directory = Path(conf["data_path"]) / dataset
        size_file = directory / (dataset + "_data_size.txt")
        sizes = np.loadtxt(size_file, dtype=np.int64).reshape(-1)
        if len(sizes) != 3 or np.any(sizes <= 0):
            raise ValueError("Size file must contain positive user, bundle and item counts")
        self.num_users, self.num_bundles, self.num_items = map(int, sizes)
        self.sizes = (self.num_users, self.num_bundles, self.num_items)
        if max(conf["topk"]) > self.num_bundles:
            raise ValueError("topk exceeds declared bundle count")
        self.data_files = {size_file.name: file_hash(size_file)}

        def relation(name, shape):
            path = directory / (name + ".txt")
            pairs = np.loadtxt(path, dtype=np.int64, delimiter="\t", ndmin=2, comments=None)
            if pairs.shape[1] != 2 or not len(pairs):
                raise ValueError(f"Expected nonempty two-column interactions: {path}")
            if np.any(pairs < 0) or np.any(pairs >= np.asarray(shape)):
                raise ValueError(f"Interaction IDs out of bounds: {path}")
            graph = sp.coo_matrix((np.ones(len(pairs), dtype=np.float32), pairs.T), shape=shape).tocsr()
            if graph.nnz != len(pairs):
                raise ValueError(f"Duplicate interactions: {path}")
            self.data_files[path.name] = file_hash(path)
            return pairs, graph

        # Preserve author graph and pair ordering, including inactive entity IDs.
        _, bi = relation("bundle_item", (self.num_bundles, self.num_items))
        _, ui = relation("user_item", (self.num_users, self.num_items))
        pairs, ub = relation("user_bundle_train", (self.num_users, self.num_bundles))
        _, val = relation("user_bundle_tune", ub.shape)
        _, test = relation("user_bundle_test", ub.shape)
        if any(left.multiply(right).nnz for left, right in ((ub, val), (ub, test), (val, test))):
            raise ValueError("User-bundle train/tune/test splits overlap")
        self.graphs = [ub, ui, bi]
        self.statistics = {"num_users": self.num_users, "num_bundles": self.num_bundles,
                           "num_items": self.num_items}
        for name, graph in (("UB_train", ub), ("UB_val", val), ("UB_test", test), ("UI", ui), ("BI", bi)):
            self.statistics[name] = {"edges": int(graph.nnz),
                                     "active_rows": int(np.count_nonzero(graph.getnnz(axis=1)))}
        self.train_loader = DataLoader(BundleTrainDataset(pairs, ub, self.num_bundles),
                                       batch_size=conf["batch_size_train"], shuffle=True,
                                       num_workers=conf["num_workers"], drop_last=True, worker_init_fn=worker_seed)
        if not len(self.train_loader):
            raise ValueError("Training data is smaller than batch_size_train with drop_last=True")
        self.val_loader = DataLoader(BundleTestDataset(val, ub), batch_size=conf["batch_size_test"],
                                     shuffle=False, num_workers=conf["num_workers"], worker_init_fn=worker_seed)
        self.test_loader = DataLoader(BundleTestDataset(test, ub), batch_size=conf["batch_size_test"],
                                      shuffle=False, num_workers=conf["num_workers"], worker_init_fn=worker_seed)
