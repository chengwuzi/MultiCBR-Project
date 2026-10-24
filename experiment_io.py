"""Small, model-independent helpers for experiment result files."""

import json
import math
import os
from pathlib import Path


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validation_score(metrics):
    values = []
    for metric in ("recall", "ndcg"):
        for k in (20, 40):
            table = metrics["val"][metric]
            value = table[k] if k in table else table[str(k)]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Invalid validation metric")
            values.append(value)
    return sum(values)


def save_best_result(path, conf, metrics, best_epoch, artifacts):
    if validation_score(metrics) <= 0:
        raise ValueError("Training finished without a valid best checkpoint")
    atomic_write_json(path, {
        "completed": True,
        "config": {key: value for key, value in conf.items() if key != "device"},
        "best_epoch": best_epoch,
        "best_epoch_1based": best_epoch + 1,
        "selection_score": validation_score(metrics),
        "metrics": metrics,
        "artifacts": artifacts,
    })
