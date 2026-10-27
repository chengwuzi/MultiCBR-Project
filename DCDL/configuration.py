"""Configuration shared by the single-run CLI and checkpoint reader."""
from copy import deepcopy
import math
from pathlib import Path

import yaml

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODEL_DIR.parent
IMPLEMENTATION = "dcdl-paper-reconstruction-v1"
SELECTION_RULE = "val Recall@20 AND NDCG@20 strictly improve over the same saved checkpoint"
DEFAULTS = dict(
    data_path="./datasets", embedding_size=64, batch_size_train=2048,
    batch_size_test=256, topk=[10, 20, 40, 80], neg_num=1,
    lr=0.001, adamw_weight_decay=0.0, dropout=0.2,
    gat_negative_slope=0.2, beta_start=0.0001, beta_end=0.03,
    lambda_2=0.0001, epochs=100, test_interval=5, seed=0,
    num_workers=0, graph_chunk_size=256,
)
REQUIRED = {"k1", "diffusion_steps", "alpha_1", "lambda_1"}


def project_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_DIR / path).resolve()


def validate(conf, check_paths=True):
    allowed = set(DEFAULTS) | REQUIRED | {"dataset"}
    unknown = set(conf) - allowed
    missing = allowed - set(conf)
    if unknown or missing:
        raise ValueError(f"Configuration keys: unknown={sorted(unknown)}, missing={sorted(missing)}")
    for key in ("embedding_size", "batch_size_train", "batch_size_test", "neg_num",
                "epochs", "test_interval", "diffusion_steps", "graph_chunk_size"):
        if type(conf[key]) is not int or conf[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("num_workers", "seed", "k1"):
        if type(conf[key]) is not int or conf[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if conf["seed"] >= 2**32:
        raise ValueError("seed must be smaller than 2**32")
    if conf["neg_num"] != 1:
        raise ValueError("This reconstruction uses the paper's one negative per observed interaction")
    if conf["embedding_size"] < 2 or conf["diffusion_steps"] < 2:
        raise ValueError("embedding_size and diffusion_steps must be at least 2")
    if conf["epochs"] < conf["test_interval"]:
        raise ValueError("epochs must cover at least one scheduled evaluation")
    for key in ("lr", "alpha_1", "gat_negative_slope"):
        value = conf[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("lambda_1", "lambda_2", "adamw_weight_decay"):
        value = conf[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    for key in ("dropout", "beta_start", "beta_end"):
        if type(conf[key]) not in (int, float) or not 0 <= conf[key] < 1:
            raise ValueError(f"{key} must be in [0, 1)")
    if not 0 < conf["beta_start"] < conf["beta_end"] < 1:
        raise ValueError("Require 0 < beta_start < beta_end < 1 to avoid a degenerate diffusion schedule")
    if not isinstance(conf["topk"], list) or any(type(k) is not int or k <= 0 for k in conf["topk"]):
        raise ValueError("topk must be a list of positive integers")
    if len(set(conf["topk"])) != len(conf["topk"]) or 20 not in conf["topk"]:
        raise ValueError("topk must have distinct entries and include 20 for checkpoint selection")
    if not isinstance(conf["dataset"], str) or Path(conf["dataset"]).name != conf["dataset"] or conf["dataset"] in (".", "..", ""):
        raise ValueError("dataset must be a directory name")
    if not isinstance(conf["data_path"], str):
        raise ValueError("data_path must be a string")
    conf["data_path"] = str(project_path(conf["data_path"]))
    if check_paths:
        directory = Path(conf["data_path"]) / conf["dataset"]
        for name in (f'{conf["dataset"]}_data_size.txt', "user_bundle_train.txt",
                     "user_bundle_tune.txt", "user_bundle_test.txt", "user_item.txt", "bundle_item.txt"):
            if not (directory / name).is_file():
                raise FileNotFoundError(directory / name)
    return conf


def load_config(path, dataset, overrides=None):
    with Path(path).open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict) or dataset not in document or not isinstance(document[dataset], dict):
        raise ValueError(f"Configuration has no dataset mapping for {dataset}")
    conf = deepcopy(DEFAULTS)
    conf.update(document[dataset])
    conf["dataset"] = dataset
    if overrides:
        conf.update({k: v for k, v in overrides.items() if v is not None})
    return validate(conf)
