"""Torch-free configuration, paths and provenance for one MultiCBR run."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import yaml

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
IMPLEMENTATION = "multicbr-author-port-v1"
SELECTION_RULE = "validation Recall@20 and NDCG@20 both strictly improve over the saved checkpoint"
SINGLETONS = {
    "embedding_sizes": "embedding_size", "num_layerss": "num_layers",
    "UB_ratios": "UB_ratio", "UI_ratios": "UI_ratio", "BI_ratios": "BI_ratio",
    "lrs": "lr", "l2_regs": "l2_reg", "c_lambdas": "c_lambda", "c_temps": "c_temp",
}
BASE_KEYS = set(SINGLETONS) | {
    "data_path", "batch_size_train", "batch_size_test", "topk", "neg_num",
    "aug_type", "ed_interval", "fusion_weights", "epochs", "test_interval",
    "seed", "num_workers",
}
RUNTIME_KEYS = set(SINGLETONS.values()) | {
    "dataset", "model", "device", "gpu", "num_users", "num_bundles", "num_items",
    "selection_rule", "implementation", "quiet", "info",
}


def project_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def file_hash(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def source_fingerprints():
    return {name: file_hash(MODEL_DIR / name) for name in
            ("model.py", "data.py", "configuration.py", "train.py")}


def read_yaml(path):
    class UniqueLoader(yaml.SafeLoader):
        pass

    def mapping(loader, node):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node)
            if not isinstance(key, str) or key in result:
                raise ValueError(f"Duplicate or non-string YAML key: {key}")
            result[key] = loader.construct_object(value_node)
        return result

    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    with Path(path).open(encoding="utf-8-sig") as stream:
        return yaml.load(stream, Loader=UniqueLoader)


def validate(conf):
    if set(conf) != BASE_KEYS:
        raise ValueError(f"Configuration missing={sorted(BASE_KEYS-set(conf))} "
                         f"unknown={sorted(set(conf)-BASE_KEYS)}")

    def number(value, name, positive=False, integer=False):
        if (type(value) not in ((int,) if integer else (int, float))
                or not math.isfinite(value) or value < 0 or (positive and value == 0)):
            raise ValueError(f"Invalid {name}: {value}")

    for plural, single in SINGLETONS.items():
        values = conf[plural]
        if not isinstance(values, list) or len(values) != 1:
            raise ValueError(f"{plural} must contain exactly one value; one invocation is one experiment")
        number(values[0], plural, positive=single in {"lr", "c_temp", "embedding_size", "num_layers"},
               integer=single in {"embedding_size", "num_layers"})
    for key in ("epochs", "test_interval", "ed_interval", "batch_size_train", "batch_size_test", "neg_num"):
        number(conf[key], key, positive=True, integer=True)
    for key in ("seed", "num_workers"):
        number(conf[key], key, integer=True)
    if conf["seed"] >= 2**32:
        raise ValueError("seed must be in [0, 2**32-1]")
    if conf["neg_num"] != 1:
        raise ValueError("The reproduction protocol uses one negative per positive")
    if conf["epochs"] < conf["test_interval"]:
        raise ValueError("epochs must reach at least one scheduled evaluation")
    if conf["aug_type"] not in {"Noise", "ED", "MD", "OP"}:
        raise ValueError("Unsupported augmentation")
    for view in ("UB", "UI", "BI"):
        ratio = conf[view + "_ratios"][0]
        if conf["aug_type"] in {"ED", "MD"} and ratio >= 1:
            raise ValueError("Dropout ratios must be smaller than one")
        if conf["aug_type"] == "OP" and ratio != 0:
            raise ValueError("Author OP augmentation requires zero ratios")
    weights = conf["fusion_weights"]
    if not isinstance(weights, dict) or set(weights) != {"modal_weight", "UB_layer", "UI_layer", "BI_layer"}:
        raise ValueError("Invalid fusion_weights keys")
    for key, values in weights.items():
        count = 3 if key == "modal_weight" else conf["num_layerss"][0] + 1
        if not isinstance(values, list) or len(values) != count:
            raise ValueError(f"Invalid fusion weight length: {key}")
        for value in values:
            number(value, key)
        if not math.isclose(sum(values), 1, abs_tol=1e-6):
            raise ValueError(f"Fusion weights must sum to one: {key}")
    ks = conf["topk"]
    if not isinstance(ks, list) or not ks:
        raise ValueError("topk must be a nonempty list")
    for k in ks:
        number(k, "topk", positive=True, integer=True)
    if ks != [10, 20, 40, 80]:
        raise ValueError("Shared reproduction topk must be [10,20,40,80]")
    if not isinstance(conf["data_path"], str) or not conf["data_path"]:
        raise ValueError("data_path must be a nonempty string")
    conf["data_path"] = str(project_path(conf["data_path"]))
    return conf


def load_config(path, dataset, overrides=None):
    document = read_yaml(path)
    if not isinstance(document, dict) or dataset not in document or not isinstance(document[dataset], dict):
        raise ValueError(f"No configuration for dataset {dataset}")
    if not isinstance(dataset, str) or not dataset or Path(dataset).name != dataset or dataset in {".", ".."}:
        raise ValueError("dataset must be a directory name")
    saved = copy.deepcopy(document[dataset])
    for key, expected in (("dataset", dataset), ("model", "MultiCBR"),
                          ("selection_rule", SELECTION_RULE), ("implementation", IMPLEMENTATION)):
        if key in saved and saved[key] != expected:
            raise ValueError(f"Snapshot identity/protocol mismatch: {key}")
    for plural, single in SINGLETONS.items():
        if single in saved and saved.get(plural) != [saved[single]]:
            raise ValueError(f"Snapshot scalar disagrees with {plural}")
    conf = {key: value for key, value in saved.items() if key not in RUNTIME_KEYS}
    if overrides:
        conf.update({key: value for key, value in overrides.items() if value is not None})
    return validate(conf)


def runtime_config(conf, dataset, sizes, gpu, device, quiet=False, info=""):
    result = copy.deepcopy(conf)
    result.update({single: result[plural][0] for plural, single in SINGLETONS.items()})
    result.update(zip(("num_users", "num_bundles", "num_items"), sizes))
    result.update(dataset=dataset, model="MultiCBR", device=str(device), gpu=str(gpu),
                  quiet=quiet, info=info, implementation=IMPLEMENTATION, selection_rule=SELECTION_RULE)
    return result
