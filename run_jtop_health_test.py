#!/usr/bin/env python3
from __future__ import annotations

import json
import time

from common import ROOT, append_timeline, percentile, write_csv, write_json
from telemetry import TELEMETRY_FIELDS, collect_blocking


def main():
    append_timeline("jtop_health", "START", "30 consecutive samples")
    started = time.perf_counter()
    samples = collect_blocking(sample_count=30, interval=0.5)
    duration = time.perf_counter() - started
    valid = len(samples) == 30 and all(row["valid"] for row in samples) and all(row["power_tot_w"] is not None for row in samples)
    write_csv(ROOT / "monitoring/jtop_health_samples.csv", samples, TELEMETRY_FIELDS)
    summary = {
        "status": "PASS" if valid else "FAIL", "sample_count": len(samples), "duration_s": duration,
        "requested_interval_s": 0.5, "all_consecutive_valid": all(row["valid"] for row in samples),
        "power_tot_present": all(row["power_tot_w"] is not None for row in samples),
        "minimum_power_w": min(row["power_tot_w"] for row in samples if row["power_tot_w"] is not None),
        "maximum_power_w": max(row["power_tot_w"] for row in samples if row["power_tot_w"] is not None),
        "errors": [row["error_message"] for row in samples if row["error_message"]],
    }
    write_json(ROOT / "monitoring/jtop_health_summary.json", summary)
    append_timeline("jtop_health", summary["status"], f"samples={len(samples)}")
    print(json.dumps(summary, indent=2))
    if not valid:
        raise RuntimeError("jtop health gate failed")


if __name__ == "__main__":
    main()
