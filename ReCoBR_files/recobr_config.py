"""Shared, torch-free configuration and evidence for ReCoBR experiments."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
RULE = "validation Recall@20 and NDCG@20 both strictly improve over the saved checkpoint"
SINGLETONS = {
    "lrs": "lr", "l2_regs": "l2_reg", "embedding_sizes": "embedding_size",
    "num_layerss": "num_layers", "UB_ratios": "UB_ratio", "UI_ratios": "UI_ratio",
    "BI_ratios": "BI_ratio", "c_lambdas": "c_lambda", "c_temps": "c_temp",
}
SOURCE_PATHS = {
    "train.py": MODEL_DIR / "train.py",
    "run_experiments.py": MODEL_DIR / "run_experiments.py",
    "recobr_config.py": MODEL_DIR / "recobr_config.py",
    "root_train.py": PROJECT_ROOT / "train.py",
    "root_utility.py": PROJECT_ROOT / "utility.py",
    "model.py": PROJECT_ROOT / "models" / "model.py",
}
SCIENTIFIC_SOURCES = set(SOURCE_PATHS) - {"run_experiments.py"}
EMBEDDING_KEYS = ("latent_diffusion_user_embedding_path", "latent_diffusion_bundle_embedding_path")


def resolve_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def file_hash(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def merge_config(base, overrides):
    if not isinstance(overrides, dict) or set(overrides) - set(base):
        raise ValueError("Overrides must use existing configuration keys")
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(base[key], dict):
            result[key] = merge_config(base[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def source_fingerprints():
    return {name: file_hash(path) for name, path in SOURCE_PATHS.items()}


def data_fingerprints(conf, dataset):
    directory = resolve_path(conf["data_path"]) / dataset
    names = [dataset + "_data_size.txt"] + [name + ".txt" for name in
             ("user_bundle_train", "user_bundle_tune", "user_bundle_test", "user_item", "bundle_item")]
    hashes = {name: file_hash(directory / name) for name in names}
    hashes.update({key: file_hash(resolve_path(conf["latent_rebuild"][key])) for key in EMBEDDING_KEYS})
    return hashes


def normalize_config(conf, dataset):
    """One trial, never an implicit grid. Fail before starting a GPU process."""
    result = copy.deepcopy(conf)
    # Use the frozen schema, rather than accepting misspelled/ignored model keys.
    import yaml
    schema = yaml.safe_load((MODEL_DIR / "config.yaml").read_text(encoding="utf-8"))["Meal-H"]

    def same_keys(value, template, label):
        if not isinstance(value, dict) or set(value) != set(template):
            raise ValueError(f"Invalid configuration keys at {label}")
        for key in template:
            if isinstance(template[key], dict):
                same_keys(value[key], template[key], label + "." + key)
    same_keys(result, schema, "config")

    def number(value, label, low=0, high=None, integer=False, strict=False):
        if (type(value) not in ((int,) if integer else (int, float)) or not math.isfinite(value)
                or (value <= low if strict else value < low) or (high is not None and value > high)):
            raise ValueError(f"Invalid numeric setting {label}: {value}")

    for plural in SINGLETONS:
        values = result[plural]
        if not isinstance(values, list) or len(values) != 1:
            raise ValueError(f"{plural} must contain exactly one value; declare separate experiments")
        number(values[0], plural, integer=plural in {"embedding_sizes", "num_layerss"},
               strict=plural in {"lrs", "embedding_sizes", "num_layerss", "c_temps"},
               high=1 if plural.endswith("ratios") else None)
        if plural.endswith("ratios") and values[0] >= 1:
            raise ValueError("Dropout must be less than 1")
    for key in ("batch_size_train", "batch_size_test", "epochs", "test_interval", "ed_interval", "neg_num"):
        number(result[key], key, integer=True, strict=True)
    number(result["seed"], "seed", integer=True, high=2**32 - 1)
    if result["neg_num"] != 1 or result["aug_type"] not in {"Noise", "ED", "MD"}:
        raise ValueError("ReCoBR requires neg_num=1 and a supported augmentation")
    ks = result["topk"]
    if not isinstance(ks, list) or 20 not in ks or len(set(ks)) != len(ks):
        raise ValueError("topk must be unique and include 20")
    for k in ks:
        number(k, "topk", integer=True, strict=True)
    if result["epochs"] < result["test_interval"]:
        raise ValueError("Training must include at least one scheduled evaluation")
    for key, values in result["fusion_weights"].items():
        expected = 3 if key == "modal_weight" else result["num_layerss"][0] + 1
        if not isinstance(values, list) or len(values) != expected:
            raise ValueError(f"Invalid fusion coefficient length: {key}")
        for value in values:
            number(value, key)
        if not math.isclose(sum(values), 1, abs_tol=1e-6):
            raise ValueError(f"Fusion coefficients must sum to one: {key}")
    for key in ("ui_bundle_user_agg_beta", "bi_user_bundle_agg_beta"):
        number(result[key], key, high=1)
    rebuild = result["latent_rebuild"]
    anchor = result["anchor_cl"]
    if type(rebuild["enabled"]) is not bool:
        raise ValueError("latent_rebuild.enabled must be boolean")
    if anchor["enabled"] is not False or anchor["anchor_cl_lambda"] != 0:
        raise ValueError("Current study requires pre-fusion anchor CL OFF (lambda=0)")
    for key in ("user_cl", "bundle_cl"):
        if type(anchor[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    for key in set(anchor) - {"enabled", "user_cl", "bundle_cl"}:
        number(anchor[key], key, strict=key == "temp")
    for key, value in rebuild.items():
        if key in EMBEDDING_KEYS or key == "enabled":
            continue
        if key == "latent_diffusion_reset_optimizer":
            if type(value) is not bool:
                raise ValueError(f"{key} must be boolean")
        elif key == "latent_diffusion_reset_after_epoch":
            number(value, key, low=-1, integer=True)
        else:
            integer = key in {"rebuild_k", "latent_diffusion_hidden_dim", "latent_diffusion_time_dim",
                              "latent_diffusion_denoiser_layers", "latent_diffusion_num_steps",
                              "latent_diffusion_batch_size", "latent_diffusion_infer_batch_size"}
            number(value, key, integer=integer, strict=integer or key in
                   {"latent_diffusion_lr", "latent_diffusion_beta_start", "latent_diffusion_beta_end"})
    if not (0 < rebuild["latent_diffusion_beta_start"] <= rebuild["latent_diffusion_beta_end"] < 1
            and 0 <= rebuild["latent_diffusion_dropout"] < 1):
        raise ValueError("Invalid diffusion schedule/dropout")
    result["data_path"] = str(resolve_path(result["data_path"]))
    for key in EMBEDDING_KEYS:
        rebuild[key] = str(resolve_path(rebuild[key]))
        if not Path(rebuild[key]).is_file():
            raise ValueError(f"Missing pretrained embedding: {rebuild[key]}")
    directory = Path(result["data_path"]) / dataset
    sizes = [int(v) for v in (directory / (dataset + "_data_size.txt")).read_text(encoding="utf-8-sig").split()]
    if len(sizes) != 3 or min(sizes) <= 0 or max(ks) > sizes[1]:
        raise ValueError("Invalid dataset sizes or topk exceeds bundle count")
    train_count = 0
    for name in ("user_bundle_train", "user_bundle_tune", "user_bundle_test", "user_item", "bundle_item"):
        path = directory / (name + ".txt")
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Missing/empty dataset file: {path}")
        if name == "user_bundle_train":
            with path.open(encoding="utf-8") as stream:
                train_count = sum(bool(line.strip()) for line in stream)
    if train_count < result["batch_size_train"]:
        raise ValueError("drop_last=True would produce zero training batches")
    return result


def runtime_config(conf, dataset, gpu, device, info, num_workers=0):
    result = copy.deepcopy(conf)
    result.update({single: result[plural][0] for plural, single in SINGLETONS.items()})
    directory = Path(result["data_path"]) / dataset
    sizes = list(map(int, (directory / (dataset + "_data_size.txt")).read_text(encoding="utf-8-sig").split()))
    result.update(zip(("num_users", "num_bundles", "num_items"), sizes))
    result.update(dataset=dataset, model="ReCoBR", gpu=str(gpu), device=str(device), info=info,
                  num_workers=num_workers, selection_rule=RULE)
    return result
