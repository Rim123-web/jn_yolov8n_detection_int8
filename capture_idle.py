#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics
import time

from common import ROOT, append_timeline, percentile, write_csv, write_json
from telemetry import TELEMETRY_FIELDS, collect_blocking


def values(rows, field):
    return [float(row[field]) for row in rows if row.get(field) is not None]


def main():
    append_timeline("idle_baseline", "START", "60 seconds, no inference")
    started = time.perf_counter()
    rows = collect_blocking(duration_s=60.0, interval=0.5)
    duration = time.perf_counter() - started
    write_csv(ROOT / "monitoring/idle_samples.csv", rows, TELEMETRY_FIELDS)
    powers = values(rows, "power_tot_w")
    cpus = values(rows, "cpu_utilization_percent")
    gpus = values(rows, "gpu_utilization_percent")
    ram = values(rows, "ram_used_mib")
    temps = values(rows, "temperature_c")
    throttles = [row for row in rows if str(row.get("throttling_status", "NA")).upper() not in {"NA", "NONE", "0", "FALSE", "NOT THROTTLED"}]
    valid = duration >= 60.0 and len(rows) >= 30 and len(powers) == len(rows) and all(row["valid"] for row in rows)
    summary = {
        "status": "PASS" if valid else "FAIL", "sample_count": len(rows), "duration_s": duration,
        "requested_interval_s": 0.5, "power_mean_w": statistics.fmean(powers),
        "power_median_w": statistics.median(powers), "power_min_w": min(powers), "power_max_w": max(powers),
        "power_p95_w": percentile(powers, 95), "cpu_mean_percent": statistics.fmean(cpus), "cpu_max_percent": max(cpus),
        "gpu_mean_percent": statistics.fmean(gpus), "gpu_max_percent": max(gpus),
        "ram_mean_mib": statistics.fmean(ram), "ram_max_mib": max(ram),
        "temperature_mean_c": statistics.fmean(temps), "temperature_max_c": max(temps),
        "throttling_event_count": len(throttles),
    }
    write_json(ROOT / "monitoring/idle_summary.json", summary)
    write_json(ROOT / "monitoring/idle_capture_validation.json", {
        "status": summary["status"], "minimum_duration_met": duration >= 60.0,
        "minimum_samples_met": len(rows) >= 30, "all_samples_valid": all(row["valid"] for row in rows),
        "no_inference_executions": True, "heavy_background_work": "none initiated by benchmark",
    })
    append_timeline("idle_baseline", summary["status"], f"duration={duration:.3f}s samples={len(rows)}")
    print(json.dumps(summary, indent=2))
    if not valid:
        raise RuntimeError("Idle capture gate failed")


if __name__ == "__main__":
    main()
