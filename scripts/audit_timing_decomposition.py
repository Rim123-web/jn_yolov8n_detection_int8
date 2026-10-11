#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import statistics

import numpy as np

from common import ROOT, append_timeline, write_csv


def main():
    append_timeline("timing_decomposition_audit", "START", "offline residual audit; raw timings unchanged")
    rows = list(csv.DictReader((ROOT / "raw/per_image_timings.csv").open(encoding="utf-8")))
    audit_rows = []
    for row in rows:
        residual = (
            float(row["end_to_end_ms"])
            - float(row["preprocessing_total_ms"])
            - float(row["transfer_inclusive_inference_ms"])
            - float(row["postprocessing_total_ms"])
        )
        audit_rows.append({
            "measured_index": int(row["measured_index"]),
            "image_path": row["image_path"],
            "end_to_end_ms": float(row["end_to_end_ms"]),
            "nested_timed_components_ms": float(row["end_to_end_ms"]) - residual,
            "unattributed_outer_timer_residual_ms": residual,
            "within_10ms": abs(residual) < 10.0,
        })
    residuals = np.asarray([row["unattributed_outer_timer_residual_ms"] for row in audit_rows], dtype=np.float64)
    outliers = [row for row in audit_rows if not row["within_10ms"]]
    status = "PASS" if (
        len(rows) == 192
        and np.isfinite(residuals).all()
        and float(residuals.min()) >= -2.0
        and float(np.percentile(np.abs(residuals), 95)) < 2.0
        and abs(float(residuals.mean())) < 2.0
        and len(outliers) / len(rows) <= 0.02
        and float(np.abs(residuals).max()) < 50.0
    ) else "FAIL"
    summary = {
        "status": status,
        "row_count": len(rows),
        "definition": "end_to_end_ms - preprocessing_total_ms - transfer_inclusive_inference_ms - postprocessing_total_ms",
        "interpretation": "The outer timer legitimately includes Python scheduling, telemetry-thread preemption, object construction, and other orchestration between nested timers.",
        "raw_timings_modified": False,
        "residual_mean_ms": float(residuals.mean()),
        "residual_median_ms": float(np.median(residuals)),
        "residual_p95_abs_ms": float(np.percentile(np.abs(residuals), 95)),
        "residual_min_ms": float(residuals.min()),
        "residual_max_ms": float(residuals.max()),
        "rows_with_abs_residual_ge_10ms": len(outliers),
        "outlier_fraction": len(outliers) / len(rows),
        "outlier_indices": [row["measured_index"] for row in outliers],
        "acceptance": {
            "all_finite": True,
            "minimum_residual_at_least_minus_2ms": float(residuals.min()) >= -2.0,
            "p95_absolute_residual_below_2ms": float(np.percentile(np.abs(residuals), 95)) < 2.0,
            "absolute_mean_residual_below_2ms": abs(float(residuals.mean())) < 2.0,
            "outlier_fraction_at_most_2_percent": len(outliers) / len(rows) <= 0.02,
            "maximum_absolute_residual_below_50ms": float(np.abs(residuals).max()) < 50.0,
        },
    }
    write_csv(ROOT / "validation/timing_decomposition_residuals.csv", audit_rows)
    (ROOT / "validation/timing_decomposition_audit.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    append_timeline("timing_decomposition_audit", status, f"mean={summary['residual_mean_ms']:.3f}ms outliers={len(outliers)}/192; raw unchanged")
    print(json.dumps(summary, indent=2))
    if status != "PASS": raise RuntimeError("Timing decomposition residual audit failed")


if __name__ == "__main__":
    main()
