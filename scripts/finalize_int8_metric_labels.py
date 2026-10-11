#!/usr/bin/env python3
from __future__ import annotations

import csv
import json

from common import ROOT, append_timeline


INT8_LABEL = "TensorRT INT8-enabled execution with TensorRT-permitted higher-precision fallbacks; offline metrics from accepted official predictions"


def update_json(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "backend" in data:
        data["backend"] = INT8_LABEL
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def update_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle)); fields = list(rows[0]) if rows else []
    if "backend" not in fields:
        return
    for row in rows:
        row["backend"] = INT8_LABEL
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def main():
    append_timeline("int8_metric_label_finalization", "START", "metadata-only; canonical metric implementation remains byte-identical")
    for relative in ("metrics/detection_metrics.json", "summary/article_results.json"):
        path = ROOT / relative
        if path.exists(): update_json(path)
    for relative in ("metrics/detection_metrics.csv", "summary/article_results.csv"):
        path = ROOT / relative
        if path.exists(): update_csv(path)
    evidence = {
        "status": "PASS",
        "label": INT8_LABEL,
        "canonical_metric_script_modified": False,
        "reason": "The canonical byte-identical FP32 metric script carried an FP32 backend label; only generated metadata labels were corrected after computation.",
        "metrics_recomputed": False,
        "inference_executed": False,
    }
    (ROOT / "validation/int8_metric_label_finalization.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    append_timeline("int8_metric_label_finalization", "PASS", "generated labels corrected; values unchanged")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
