#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
import re
from pathlib import Path


OUT_DIR = Path("outputs/module1_ablation")
STATUS_FILE = OUT_DIR / "experiment_status.json"
SUMMARY_TXT = OUT_DIR / "summary.txt"
SUMMARY_CSV = OUT_DIR / "summary.csv"

TOPKS = [10, 20, 40, 80]
METHOD_NAMES = {
    "Raw": "Raw UB baseline",
    "A": "Diffusion Top-K",
    "B": "Random Top-K",
    "C": "Pretrain Similarity Top-K",
    "D": "Popularity Top-K",
    "E": "Pooled z0 Top-K",
}

TEST_METRIC_RE = re.compile(
    r"Best in epoch\s+(?P<epoch>\d+),\s+TOP\s+(?P<topk>\d+):\s+"
    r"REC_T=(?P<recall>[0-9.]+),\s+NDCG_T=(?P<ndcg>[0-9.]+)"
)


def load_status():
    if not STATUS_FILE.exists():
        return {}
    return json.loads(STATUS_FILE.read_text(encoding="utf-8"))


def parse_best_metrics(log_path):
    metrics = {}
    best_epoch = ""
    path = Path(log_path)
    if not path.exists():
        return best_epoch, metrics

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = TEST_METRIC_RE.search(line)
        if not match:
            continue
        topk = int(match.group("topk"))
        if topk not in TOPKS:
            continue
        best_epoch = match.group("epoch")
        metrics[f"recall@{topk}"] = match.group("recall")
        metrics[f"ndcg@{topk}"] = match.group("ndcg")
    return best_epoch, metrics


def read_failure_tail(log_path, max_lines=12):
    path = Path(log_path)
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    tail = [line for line in lines[-max_lines:] if line.strip()]
    return " | ".join(tail)


def build_rows(status_data):
    rows = []
    for experiment_id, entry in sorted(status_data.items()):
        log_path = entry.get("log_path", "")
        best_epoch, metrics = parse_best_metrics(log_path)
        row = {
            "experiment_id": experiment_id,
            "method_group": entry.get("method_group", ""),
            "method_name": METHOD_NAMES.get(entry.get("method_group", ""), ""),
            "strategy": entry.get("strategy", ""),
            "k": entry.get("k", ""),
            "status": entry.get("status", ""),
            "attempts": entry.get("attempts", ""),
            "best_epoch": best_epoch,
            "log_path": log_path,
            "updated_at": entry.get("updated_at", ""),
            "failure_tail": "",
        }
        for topk in TOPKS:
            row[f"recall@{topk}"] = metrics.get(f"recall@{topk}", "")
            row[f"ndcg@{topk}"] = metrics.get(f"ndcg@{topk}", "")
        if row["status"] != "success":
            row["failure_tail"] = read_failure_tail(log_path)
        rows.append(row)
    return rows


def write_csv(rows):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment_id",
        "method_group",
        "method_name",
        "strategy",
        "k",
        "status",
        "attempts",
        "best_epoch",
        "recall@10",
        "ndcg@10",
        "recall@20",
        "ndcg@20",
        "recall@40",
        "ndcg@40",
        "recall@80",
        "ndcg@80",
        "updated_at",
        "log_path",
        "failure_tail",
    ]
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_txt(rows):
    lines = []
    lines.append("iFashion Module-One Ablation Summary")
    lines.append("=" * 80)
    lines.append("")
    if not rows:
        lines.append("No experiment status has been recorded yet.")
    for row in rows:
        method_name = row["method_name"] or row["method_group"]
        lines.append(
            f"[{row['status'].upper()}] {row['experiment_id']} | "
            f"{method_name} | strategy={row['strategy']} | k={row['k']} | "
            f"attempts={row['attempts']} | best_epoch={row['best_epoch'] or 'N/A'}"
        )
        if row["status"] == "success" and row["recall@10"]:
            for topk in TOPKS:
                lines.append(
                    f"  TOP {topk}: REC_T={row[f'recall@{topk}']}, "
                    f"NDCG_T={row[f'ndcg@{topk}']}"
                )
        elif row["failure_tail"]:
            lines.append(f"  Failure tail: {row['failure_tail']}")
        lines.append(f"  Log: {row['log_path']}")
        lines.append("")
    SUMMARY_TXT.write_text("\n".join(lines), encoding="utf-8")


def main():
    status_data = load_status()
    rows = build_rows(status_data)
    write_csv(rows)
    write_txt(rows)
    print(f"Wrote {SUMMARY_TXT} and {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
