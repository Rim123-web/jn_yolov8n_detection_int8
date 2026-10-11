#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import statistics

from common import ROOT, append_timeline, percentile, write_csv, write_json
from postprocessing import TIMING_FIELDS as POSTPROCESS_FIELDS
from preprocessing import TIMING_FIELDS as PREPROCESS_FIELDS


RUNTIME_FIELDS = ["h2d_ms", "tensorrt_engine_ms", "d2h_ms", "transfer_inclusive_inference_ms"]
ALL_TIMING_FIELDS = PREPROCESS_FIELDS + RUNTIME_FIELDS + POSTPROCESS_FIELDS + ["end_to_end_ms"]


def stats(values):
    return {
        "count": len(values), "mean_ms": statistics.fmean(values),
        "std_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min_ms": min(values), "p50_ms": percentile(values, 50), "p90_ms": percentile(values, 90),
        "p95_ms": percentile(values, 95), "p99_ms": percentile(values, 99), "max_ms": max(values),
    }


def main():
    append_timeline("timing_analysis", "START", "offline CSV analysis")
    rows = list(csv.DictReader((ROOT / "raw/per_image_timings.csv").open(encoding="utf-8")))
    if len(rows) != 192:
        raise RuntimeError(f"Expected 192 timing rows, found {len(rows)}")
    summaries = []
    e2e_mean = statistics.fmean(float(row["end_to_end_ms"]) for row in rows)
    for field in ALL_TIMING_FIELDS:
        values = [float(row[field]) for row in rows]
        result = {"stage": field, **stats(values)}
        result["percentage_of_end_to_end"] = 100.0 * result["mean_ms"] / e2e_mean
        summaries.append(result)
    write_csv(ROOT / "summary/timing_stage_summary.csv", summaries)
    main_fields = ["tensorrt_engine_ms", "transfer_inclusive_inference_ms", "preprocessing_total_ms", "postprocessing_total_ms", "nms_ms", "end_to_end_ms"]
    write_csv(ROOT / "summary/latency_percentiles.csv", [x for x in summaries if x["stage"] in main_fields])
    provenance = json.loads((ROOT / "raw/execution_count_provenance.json").read_text())
    wall_duration = float(provenance["official_measured_wall_clock_duration_s"])
    by_stage = {row["stage"]: row for row in summaries}
    engine_mean = by_stage["tensorrt_engine_ms"]["mean_ms"]
    transfer_mean = by_stage["transfer_inclusive_inference_ms"]["mean_ms"]
    performance = {
        "pure_tensorrt_engine_latency_mean_ms": engine_mean,
        "engine_fps": 1000.0 / engine_mean,
        "transfer_inclusive_latency_mean_ms": transfer_mean,
        "transfer_inclusive_fps": 1000.0 / transfer_mean,
        "mean_pipeline_latency_ms": e2e_mean,
        "latency_derived_pipeline_fps": 1000.0 / e2e_mean,
        "wall_clock_pipeline_duration_s": wall_duration,
        "wall_clock_pipeline_fps": len(rows) / wall_duration,
        "measured_images": len(rows),
        "definitions": {
            "engine": "CUDA-event time around execute_async_v2 only",
            "transfer_inclusive": "wall time around synchronized H2D + TensorRT execute + D2H",
            "pipeline": "per-image decode/preprocess + transfer/runtime + decode/filter/NMS/rescale/format",
            "wall_clock": "complete official image loop including result buffering, excluding warmup, telemetry setup/teardown, metrics, plots and file writes",
        },
    }
    write_json(ROOT / "summary/performance_summary.json", performance)
    (ROOT / "summary/fps_definitions.md").write_text(
        "# FPS definitions\n\n"
        f"- Engine FPS: `1000 / {engine_mean:.6f} ms = {performance['engine_fps']:.6f}` images/s.\n"
        f"- Transfer-inclusive FPS: `1000 / {transfer_mean:.6f} ms = {performance['transfer_inclusive_fps']:.6f}` images/s.\n"
        f"- Latency-derived pipeline FPS: `1000 / {e2e_mean:.6f} ms = {performance['latency_derived_pipeline_fps']:.6f}` images/s.\n"
        f"- Wall-clock pipeline FPS: `192 / {wall_duration:.6f} s = {performance['wall_clock_pipeline_fps']:.6f}` images/s.\n\n"
        "Engine FPS is not labeled application FPS. Warmups and all offline analysis are excluded.\n",
        encoding="utf-8",
    )
    stage_means = {row["stage"]: row["mean_ms"] for row in summaries}
    outside = 100.0 * (e2e_mean - engine_mean) / e2e_mean
    aggregate_fields = {
        "preprocessing_total_ms", "transfer_inclusive_inference_ms",
        "postprocessing_total_ms", "end_to_end_ms",
    }
    largest = max((row for row in summaries if row["stage"] not in aggregate_fields), key=lambda x: x["mean_ms"])
    contributions = [{"stage": row["stage"], "mean_ms": row["mean_ms"], "percentage_of_e2e": row["percentage_of_end_to_end"]} for row in summaries if row["stage"] != "end_to_end_ms"]
    write_csv(ROOT / "analysis/stage_latency_contributions.csv", contributions)
    bottleneck = {
        "largest_mean_latency_stage": largest["stage"], "largest_stage_mean_ms": largest["mean_ms"],
        "bottleneck_selection_scope": "atomic independently timed stages only; aggregate totals excluded",
        "percentage_outside_tensorrt": outside,
        "preprocessing_share_percent": 100 * stage_means["preprocessing_total_ms"] / e2e_mean,
        "h2d_share_percent": 100 * stage_means["h2d_ms"] / e2e_mean,
        "engine_share_percent": 100 * engine_mean / e2e_mean,
        "d2h_share_percent": 100 * stage_means["d2h_ms"] / e2e_mean,
        "postprocessing_share_percent": 100 * stage_means["postprocessing_total_ms"] / e2e_mean,
        "nms_share_percent": 100 * stage_means["nms_ms"] / e2e_mean,
        "cpu_bound_stages": ["file_access_decode", "PIL resize/letterbox", "NumPy output decode/NMS/rescale"],
        "gpu_bound_stages": ["H2D", "TensorRT engine", "D2H"],
        "stage_means_ms": stage_means,
    }
    write_json(ROOT / "analysis/bottleneck_summary.json", bottleneck)
    (ROOT / "analysis/bottleneck_report.md").write_text(
        f"# Bottleneck analysis\n\nThe largest measured stage is `{largest['stage']}` at {largest['mean_ms']:.3f} ms. "
        f"TensorRT engine execution accounts for {100 * engine_mean/e2e_mean:.2f}% of mean end-to-end latency; "
        f"{outside:.2f}% is outside the TensorRT engine. NMS alone contributes {100 * stage_means['nms_ms']/e2e_mean:.2f}%.\n",
        encoding="utf-8",
    )
    append_timeline("timing_analysis", "PASS", f"engine={engine_mean:.3f}ms e2e={e2e_mean:.3f}ms")
    print(json.dumps(performance, indent=2))


if __name__ == "__main__":
    main()
