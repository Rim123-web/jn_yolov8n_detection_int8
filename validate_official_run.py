#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import subprocess
from pathlib import Path

from common import ROOT, append_timeline, sha256_file, write_json
from postprocessing import TIMING_FIELDS as POST
from preprocessing import TIMING_FIELDS as PRE


TIMINGS = PRE + ["h2d_ms", "tensorrt_engine_ms", "d2h_ms", "transfer_inclusive_inference_ms"] + POST + ["end_to_end_ms"]


def gate(name, condition, evidence=""):
    return {"gate": name, "pass": bool(condition), "evidence": evidence}


def main():
    reference = json.loads((ROOT / "reference/fp32_reference_selection.json").read_text())
    fp32 = Path(reference["selected_path"])
    warmups = list(csv.DictReader((ROOT / "raw/warmup_timings.csv").open(encoding="utf-8")))
    measured = list(csv.DictReader((ROOT / "raw/per_image_timings.csv").open(encoding="utf-8")))
    predictions = (ROOT / "raw/predictions.jsonl").read_text(encoding="utf-8").splitlines()
    telemetry = list(csv.DictReader((ROOT / "monitoring/active_samples.csv").open(encoding="utf-8")))
    execution = json.loads((ROOT / "raw/execution_count_provenance.json").read_text())
    current_order = (ROOT / "raw/official_image_order.txt").read_text(encoding="utf-8").splitlines()
    fp32_order = (fp32 / "raw/official_image_order.txt").read_text(encoding="utf-8").splitlines()
    fp32_warmups = json.loads((fp32 / "raw/warmup_execution_provenance.json").read_text())["image_paths"]
    system = json.loads((ROOT / "environment/system_information.json").read_text())
    current_boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    metadata = json.loads((ROOT / "engine/engine_metadata.json").read_text())
    all_predictions_finite = True
    for line in predictions:
        for detection in json.loads(line)["detections"]:
            all_predictions_finite &= all(math.isfinite(float(detection[key])) for key in ("confidence", "x1", "y1", "x2", "y2"))
    before_oom = set((ROOT / "logs/kernel_oom_before.txt").read_text(encoding="utf-8", errors="replace").splitlines())
    after_oom = (ROOT / "logs/kernel_oom_after.txt").read_text(encoding="utf-8", errors="replace").splitlines()
    new_oom = [line for line in after_oom if line not in before_oom]
    gates = [
        gate("process_exit_code", True, "benchmark command exited 0 before this gate"),
        gate("jtop_active", subprocess.run(["systemctl", "is-active", "jtop.service"], capture_output=True, text=True).stdout.strip() == "active"),
        gate("engine_unchanged", sha256_file(ROOT / "engine/oct_albument2_v6_tensorrt_int8_1x3x640x960.engine") == metadata["sha256"]),
        gate("onnx_unchanged", sha256_file(ROOT / "onnx/oct_albument2_v6_inference_only.onnx") == reference["onnx_sha256"]),
        gate("source_unchanged", sha256_file(ROOT / "model/oct_albument2_v6_source.pt") == reference["source_sha256"]),
        gate("warmup_rows_exact", len(warmups) == 10), gate("measured_rows_exact", len(measured) == 192),
        gate("prediction_image_rows_exact", len(predictions) == 192), gate("unique_paths", len({row["image_path"] for row in measured}) == 192),
        gate("image_order_parity", current_order == fp32_order), gate("warmup_path_parity", [row["image_path"] for row in warmups] == fp32_warmups),
        gate("failures_zero", execution["failed_measured_images"] == 0), gate("discarded_zero", execution["discarded_measured_rows"] == 0),
        gate("hidden_zero", execution["hidden_executions"] == 0), gate("all_timings_finite_nonnegative", all(math.isfinite(float(row[field])) and float(row[field]) >= 0 for row in measured for field in TIMINGS)),
        gate("predictions_finite", all_predictions_finite), gate("telemetry_coverage", len(telemetry) >= 2 and all(row["valid"].lower() == "true" for row in telemetry)),
        gate("no_oom", not new_oom, str(new_oom)), gate("no_reboot", current_boot == system["boot_id"]),
    ]
    status = "PASS" if all(row["pass"] for row in gates) else "FAIL"
    result = {"status": status, "gate_count": len(gates), "passed": sum(row["pass"] for row in gates), "failed": [row["gate"] for row in gates if not row["pass"]], "gates": gates}
    write_json(ROOT / "validation/accepted_official_run_validation.json", result)
    lines = ["# Accepted official INT8 run validation", "", f"**Status: {status}**", "", "| Gate | Result | Evidence |", "|---|---|---|"] + [f"| {row['gate']} | {'PASS' if row['pass'] else 'FAIL'} | {str(row['evidence']).replace('|', '/')} |" for row in gates]
    (ROOT / "validation/accepted_official_run_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    append_timeline("accepted_official_run_validation", status, f"{result['passed']}/{result['gate_count']}")
    print(json.dumps(result, indent=2))
    if status != "PASS":
        raise RuntimeError("Official INT8 run acceptance failed")


if __name__ == "__main__":
    main()
