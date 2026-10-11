#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import psutil

from common import (
    CLASS_NAMES, CONFIDENCE_THRESHOLD, HEIGHT, ROOT, WIDTH, append_timeline,
    collect_images, sha256_file, write_csv, write_json,
)
from postprocessing import TIMING_FIELDS as POSTPROCESS_FIELDS
from postprocessing import postprocess_yolov8
from preprocessing import TIMING_FIELDS as PREPROCESS_FIELDS
from preprocessing import preprocess_image
from telemetry import JtopMonitor, TELEMETRY_FIELDS
from tensorrt_runtime import TensorRTRunner


RUNTIME_FIELDS = ["h2d_ms", "tensorrt_engine_ms", "d2h_ms", "transfer_inclusive_inference_ms"]
OFFICIAL_FIELDS = [
    "measured_index", "image_path", "image_sha256", "original_width", "original_height",
    "input_width", "input_height", *PREPROCESS_FIELDS, *RUNTIME_FIELDS,
    *POSTPROCESS_FIELDS, "end_to_end_ms", "detection_count", "process_rss_mib",
    "success", "error_message",
]


def prediction_rows(image_index, image_path, detections):
    rows = []
    for detection_index, detection in enumerate(detections):
        x1, y1, x2, y2 = detection["box"]
        rows.append({
            "image_index": image_index, "image_path": str(image_path), "detection_index": detection_index,
            "class_id": detection["class_id"], "class_name": detection["class_name"],
            "confidence": detection["confidence"], "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        })
    return rows


def execute_one(runner, image_path, image_hash, index_name, index_value):
    e2e_start = time.perf_counter_ns()
    array, meta, preprocess_timing = preprocess_image(image_path)
    outputs, runtime_timing = runner.run_timed(array)
    detections, postprocess_timing = postprocess_yolov8(outputs[0], meta, confidence_threshold=CONFIDENCE_THRESHOLD)
    e2e_end = time.perf_counter_ns()
    row = {
        index_name: index_value, "image_path": str(image_path.resolve()), "image_sha256": image_hash,
        "original_width": meta["original_width"], "original_height": meta["original_height"],
        "input_width": WIDTH, "input_height": HEIGHT, **preprocess_timing, **runtime_timing,
        **postprocess_timing, "end_to_end_ms": (e2e_end - e2e_start) / 1_000_000.0,
        "detection_count": len(detections), "process_rss_mib": psutil.Process().memory_info().rss / (1024 ** 2),
        "success": True, "error_message": "",
    }
    return row, detections, outputs[0], meta


def main():
    append_timeline("official_benchmark", "START", "INT8-enabled: 10 FP32-matched real warmups then one 192-image measured pass")
    images = collect_images()
    if len(images) != 192:
        raise RuntimeError(f"Expected 192 images, found {len(images)}")
    manifest_rows = {row["image_path"]: row for row in csv.DictReader((ROOT / "dataset/test_image_manifest.csv").open(encoding="utf-8"))}
    engine = ROOT / "engine/oct_albument2_v6_tensorrt_int8_1x3x640x960.engine"
    warmup_rows = []
    warmup_predictions = []
    official_rows = []
    official_predictions = []
    coco_predictions = []
    raw_outputs = None
    failures = []
    with TensorRTRunner(engine) as runner:
        write_json(ROOT / "engine/runtime_metadata_official.json", runner.metadata())
        output_shape = tuple(runner.binding_metadata[runner.output_indices[0]]["runtime_shape"])
        raw_outputs = np.empty((len(images),) + output_shape[1:], dtype=np.float32)
        reference = json.loads((ROOT / "reference/fp32_reference_selection.json").read_text())
        fp32_root = Path(reference["selected_path"])
        expected_order = (fp32_root / "raw/official_image_order.txt").read_text(encoding="utf-8").splitlines()
        expected_warmups = json.loads((fp32_root / "raw/warmup_execution_provenance.json").read_text())["image_paths"]
        if [str(path.resolve()) for path in images] != expected_order:
            raise RuntimeError("INT8 deterministic image order differs from authoritative FP32")
        if [str(path.resolve()) for path in images[:10]] != expected_warmups:
            raise RuntimeError("INT8 warm-up paths/order differ from authoritative FP32")
        for warmup_index, image_path in enumerate(images[:10], 1):
            image_hash = manifest_rows[str(image_path.resolve())]["sha256"]
            row, detections, _, _ = execute_one(runner, image_path, image_hash, "warmup_index", warmup_index)
            warmup_rows.append(row)
            warmup_predictions.append({"warmup_index": warmup_index, "image_path": str(image_path.resolve()), "detections": detections})
            print(f"Warmup {warmup_index}/10 complete", flush=True)
        if len(warmup_rows) != 10 or len({row["image_path"] for row in warmup_rows}) != 10 or not all(row["success"] for row in warmup_rows):
            raise RuntimeError("Warmup validation failed")
        write_csv(ROOT / "raw/warmup_timings.csv", warmup_rows, ["warmup_index"] + OFFICIAL_FIELDS[1:])
        with (ROOT / "raw/warmup_predictions.jsonl").open("w", encoding="utf-8") as stream:
            for item in warmup_predictions:
                stream.write(json.dumps(item) + "\n")
        write_json(ROOT / "raw/warmup_execution_provenance.json", {
            "status": "PASS", "configured_warmups": 10, "recorded_rows": len(warmup_rows),
            "unique_image_paths": len({row["image_path"] for row in warmup_rows}), "engine_executions": 10,
            "real_images": True, "full_pipeline": True, "preprocessing_included": True,
            "h2d_included": True, "engine_included": True, "d2h_included": True,
            "postprocessing_and_nms_included": True, "drawing_or_saving_included": False,
            "excluded_from_official_latency_fps_metrics_resources_energy": True,
            "image_paths": [row["image_path"] for row in warmup_rows], "failures": 0,
            "paths_match_authoritative_fp32": True,
        })

        monitor = JtopMonitor(interval=0.5, include_process=True)
        monitor.start()
        wall_start = time.perf_counter()
        for measured_index, image_path in enumerate(images, 1):
            image_hash = manifest_rows[str(image_path.resolve())]["sha256"]
            try:
                row, detections, output, meta = execute_one(runner, image_path, image_hash, "measured_index", measured_index)
                official_rows.append(row)
                np.copyto(raw_outputs[measured_index - 1], output[0].astype(np.float32, copy=False))
                image_prediction_rows = prediction_rows(measured_index, image_path, detections)
                official_predictions.extend(image_prediction_rows)
                for prediction in image_prediction_rows:
                    coco_predictions.append({
                        "image_id": measured_index,
                        "category_id": prediction["class_id"],
                        "bbox": [prediction["x1"], prediction["y1"], prediction["x2"] - prediction["x1"], prediction["y2"] - prediction["y1"]],
                        "score": prediction["confidence"],
                    })
            except BaseException as exc:
                failures.append({"measured_index": measured_index, "image_path": str(image_path), "error": repr(exc)})
                print(f"Measured image {measured_index} failed: {exc!r}", flush=True)
                break
            if measured_index % 20 == 0 or measured_index == len(images):
                print(f"Measured {measured_index}/{len(images)}", flush=True)
        wall_end = time.perf_counter()
        monitor.stop()
    measured_duration = wall_end - wall_start
    active_samples = [row for row in monitor.samples if wall_start <= row["monotonic_s"] <= wall_end]
    if failures or len(official_rows) != len(images) or len({row["image_path"] for row in official_rows}) != len(images):
        write_json(ROOT / "raw/official_failures.json", failures)
        raise RuntimeError(f"Official measured pass failed: rows={len(official_rows)} failures={failures}")
    if not active_samples or not all(row["valid"] for row in active_samples):
        raise RuntimeError(f"Official active telemetry invalid: samples={len(active_samples)}")
    write_csv(ROOT / "raw/per_image_timings.csv", official_rows, OFFICIAL_FIELDS)
    write_csv(ROOT / "raw/predictions.csv", official_predictions)
    with (ROOT / "raw/predictions.jsonl").open("w", encoding="utf-8") as stream:
        grouped = {index: [] for index in range(1, len(images) + 1)}
        for prediction in official_predictions:
            grouped[prediction["image_index"]].append(prediction)
        for index, image_path in enumerate(images, 1):
            stream.write(json.dumps({"image_index": index, "image_path": str(image_path.resolve()), "confidence_threshold": CONFIDENCE_THRESHOLD, "detections": grouped[index]}) + "\n")
    write_json(ROOT / "raw/coco_predictions.json", coco_predictions)
    np.savez_compressed(ROOT / "raw/official_raw_outputs.npz", outputs=raw_outputs)
    (ROOT / "raw/official_image_order.txt").write_text("\n".join(str(path.resolve()) for path in images) + "\n", encoding="utf-8")
    write_csv(ROOT / "monitoring/active_samples.csv", active_samples, TELEMETRY_FIELDS)
    correctness = json.loads((ROOT / "validation/int8_correctness_summary.json").read_text())
    rejected_attempt_metadata_path = ROOT / "stability/rejected_official_attempt_1_postrun_provenance_bug/rejected_attempt_metadata.json"
    rejected_attempt_metadata = json.loads(rejected_attempt_metadata_path.read_text()) if rejected_attempt_metadata_path.exists() else None
    provenance = {
        "status": "PASS", "correctness_executions": correctness["int8_tensorrt_executions"],
        "warmup_executions": 10, "measured_executions": len(official_rows),
        "accepted_total_benchmark_executions": 10 + len(official_rows),
        "hidden_executions": 0, "failed_measured_images": len(failures), "discarded_measured_rows": 0,
        "first_measured_image_discarded": False, "failed_images_retried": False,
        "shape_discovery_inference": False, "deterministic_sorted_order": True,
        "official_measured_wall_clock_duration_s": measured_duration,
        "official_wall_start_monotonic_s": wall_start,
        "official_wall_end_monotonic_s": wall_end,
        "active_telemetry_samples": len(active_samples),
        "prior_rejected_attempts": 1 if rejected_attempt_metadata else 0,
        "prior_rejected_inference_executions": rejected_attempt_metadata.get("inference_executions", 0) if rejected_attempt_metadata else 0,
        "prior_rejected_attempt_evidence": str(rejected_attempt_metadata_path) if rejected_attempt_metadata else None,
        "accepted_execution_count_excludes_prior_rejected_attempts": True,
        "raw_output_buffering": "One host-to-host copy per measured output occurs after end_to_end_ms but inside wall-clock duration; no extra engine execution.",
        "metric_calculation_inside_timed_loop": False, "drawing_or_saving_inside_timed_loop": False,
    }
    write_json(ROOT / "raw/execution_count_provenance.json", provenance)
    write_json(ROOT / "provenance/execution_count_summary.json", provenance)
    append_timeline("official_benchmark", "PASS", f"warmup=10 measured={len(official_rows)} duration={measured_duration:.3f}s")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
