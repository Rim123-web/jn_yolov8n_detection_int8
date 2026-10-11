#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(os.environ["BENCHMARK_DIR"]).resolve()
BENCH_ROOT = Path("/home/jetson/Downloads/jetson_folder/yolov8n_detection_fp32_benchmark")
FP32 = BENCH_ROOT / "yolov8n_detection_tensorrt_fp32_640x960_batch1_warmup10_20260720_165642"
FP16 = BENCH_ROOT / "yolov8n_detection_tensorrt_fp16_640x960_batch1_warmup10_20260720_180924"
SOURCE = Path("/home/jetson/Downloads/jetson_folder/oct_albument2_v6.pt")
CANONICAL = [
    "preprocessing.py", "tensorrt_runtime.py", "postprocessing.py", "nms.py",
    "run_jtop_health_test.py", "capture_idle.py", "calculate_detection_metrics.py",
    "analyze_timings.py", "analyze_energy.py", "analyze_resources.py", "telemetry.py",
    "create_manifest.py", "package_results.sh", "common.py",
]
DIRS = [
    "model", "onnx", "engine", "calibration", "config", "dataset", "scripts", "logs",
    "raw", "monitoring", "energy", "resources", "temperature", "throttling", "metrics",
    "summary", "analysis", "comparison", "figures/png", "figures/svg", "figures/pdf",
    "figures/data", "reports", "validation", "provenance", "reference", "stability",
    "environment",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def command(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, timeout=120).stdout.strip()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def verify_manifest(package: Path):
    manifest = package / "provenance/hash_manifest.csv"
    rows = list(csv.DictReader(manifest.open(encoding="utf-8-sig")))
    failures = []
    for row in rows:
        p = package / row["relative_path"]
        if not p.is_file() or p.stat().st_size != int(row["size_bytes"]) or sha256(p) != row["sha256"]:
            failures.append(row["relative_path"])
    return len(rows), failures


def package_snapshot(package: Path):
    rows = list(csv.DictReader((package / "provenance/hash_manifest.csv").open(encoding="utf-8-sig")))
    return {
        "path": str(package), "directory_mtime_ns": package.stat().st_mtime_ns,
        "manifest_sha256": sha256(package / "provenance/hash_manifest.csv"),
        "manifest_entries": len(rows),
        "files": {row["relative_path"]: {"size_bytes": int(row["size_bytes"]), "sha256": row["sha256"]} for row in rows},
    }


def reference_info(package: Path, precision: str):
    validation = read_json(package / "validation/final_validation_report.json")
    entries, failures = verify_manifest(package)
    source_hash = (package / "model/source_model_sha256.txt").read_text().split()[0]
    onnx_hash = (package / "onnx/onnx_sha256.txt").read_text().split()[0]
    engine_hash = (package / "engine/engine_sha256.txt").read_text().split()[0]
    image_order = (package / "raw/official_image_order.txt").read_bytes()
    warm = read_json(package / "raw/warmup_execution_provenance.json")
    execution = read_json(package / "raw/execution_count_provenance.json")
    status = validation.get("status")
    if status != "PASS" or failures or warm.get("engine_executions") != 10 or execution.get("measured_executions") != 192:
        raise RuntimeError(f"Reference {precision} failed validation: status={status}, manifest={failures[:3]}")
    return {
        "status": "PASS", "precision": precision, "selected_path": str(package),
        "selection_basis": "accepted final validation PASS plus complete per-file manifest verification",
        "manifest_entries": entries, "manifest_failures": failures,
        "source_sha256": source_hash, "onnx_sha256": onnx_hash, "engine_sha256": engine_hash,
        "input_shape": [1, 3, 640, 960], "batch_size": 1, "warmup_count": 10,
        "measured_image_count": 192, "test_order_sha256": hashlib.sha256(image_order).hexdigest(),
        "warmup_paths": warm.get("warmup_image_paths", warm.get("image_paths", [])),
    }


def environment():
    commands = {
        "hostname": "hostname", "ip": "hostname -I", "date": "date --iso-8601=seconds",
        "boot_id": "cat /proc/sys/kernel/random/boot_id", "uptime": "uptime -p",
        "architecture": "uname -m", "ubuntu": "lsb_release -ds 2>/dev/null || true",
        "kernel": "uname -a", "l4t": "cat /etc/nv_tegra_release 2>/dev/null || true",
        "cuda": "nvcc --version 2>/dev/null || cat /usr/local/cuda/version.txt 2>/dev/null || true",
        "tensorrt": "dpkg-query -W -f='${Version}' libnvinfer8 2>/dev/null || true",
        "cudnn": "dpkg-query -W -f='${Version}' libcudnn8 2>/dev/null || true",
        "nvpmodel": "nvpmodel -q 2>/dev/null || true", "jetson_clocks": "jetson_clocks --show 2>/dev/null || true",
        "cpu_governor": "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || true",
        "online_cores": "cat /sys/devices/system/cpu/online", "memory": "free -b", "disk": "df -B1 /home/jetson/Downloads/jetson_folder",
        "temperatures": "for f in /sys/class/thermal/thermal_zone*/temp; do printf '%s=' \"$f\"; cat \"$f\"; done",
        "processes": "ps -eo pid,ppid,stat,etimes,%cpu,%mem,rss,args --sort=-%cpu",
        "jetson_stats": "jetson_release -v 2>/dev/null || true", "service": "systemctl is-active jtop.service 2>/dev/null || systemctl is-active jetson_stats.service 2>/dev/null || true",
    }
    info = {k: command(v) for k, v in commands.items()}
    info["python_executable"] = sys.executable
    info["python_version"] = platform.python_version()
    versions = command("python - <<'PY'\nimport importlib\nfor n in ['torch','torchvision','ultralytics','numpy','PIL','cv2','onnx','onnxruntime','jtop','psutil']:\n try:\n  m=importlib.import_module(n); print(n+'='+str(getattr(m,'__version__','version_not_available')))\n except BaseException as e: print(n+'=UNAVAILABLE:'+repr(e))\nPY")
    info["python_packages"] = versions
    write_json(ROOT / "environment/system_information.json", info)
    (ROOT / "environment/system_information.txt").write_text("\n\n".join(f"[{k}]\n{v}" for k, v in info.items()) + "\n", encoding="utf-8")
    write_json(ROOT / "environment/software_versions.json", {"python": platform.python_version(), "packages_raw": versions})
    (ROOT / "environment/python_environment.txt").write_text(command("python -V; which python; python -m pip freeze 2>/dev/null || true") + "\n", encoding="utf-8")
    (ROOT / "environment/storage_and_memory.txt").write_text(info["memory"] + "\n\n" + info["disk"] + "\n", encoding="utf-8")
    rows = []
    for line in info["processes"].splitlines()[1:]:
        parts = line.split(None, 7)
        if len(parts) == 8:
            rows.append(parts)
    with (ROOT / "environment/preflight_processes.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["pid","ppid","stat","etimes","cpu_percent","memory_percent","rss_kib","command"]); w.writerows(rows)
    write_json(ROOT / "environment/power_clock_state.json", {k: info[k] for k in ["nvpmodel","jetson_clocks","cpu_governor","online_cores","temperatures"]})


def main():
    if ROOT.exists():
        raise FileExistsError(f"Fresh package already exists: {ROOT}")
    for rel in DIRS:
        (ROOT / rel).mkdir(parents=True, exist_ok=True)
    environment()
    fp32 = reference_info(FP32, "FP32")
    fp16 = reference_info(FP16, "FP16")
    if fp32["source_sha256"] != fp16["source_sha256"] or fp32["onnx_sha256"] != fp16["onnx_sha256"]:
        raise RuntimeError("FP32/FP16 source or ONNX hash mismatch")
    if fp32["test_order_sha256"] != fp16["test_order_sha256"] or fp32["warmup_paths"] != fp16["warmup_paths"]:
        raise RuntimeError("FP32/FP16 image or warm-up order mismatch")
    write_json(ROOT / "reference/fp32_reference_selection.json", fp32)
    write_json(ROOT / "reference/fp16_reference_selection.json", fp16)
    parity = {"status":"PASS", "source_sha256_equal":True, "onnx_sha256_equal":True,
              "test_order_equal":True, "warmup_order_equal":True, "input_shape":[1,3,640,960], "batch_size":1}
    write_json(ROOT / "reference/reference_protocol_parity.json", parity)
    write_json(ROOT / "reference/reference_hashes.json", {"fp32":fp32, "fp16":fp16})
    before = {"captured_at": datetime.now(timezone.utc).astimezone().isoformat(), "fp32":package_snapshot(FP32), "fp16":package_snapshot(FP16)}
    write_json(ROOT / "reference/reference_immutability_before.json", before)

    shutil.copy2(SOURCE, ROOT / "model/oct_albument2_v6_source.pt")
    shared_onnx = FP32 / "onnx/oct_albument2_v6_inference_only.onnx"
    shutil.copy2(shared_onnx, ROOT / "onnx/oct_albument2_v6_inference_only.onnx")
    for name in CANONICAL:
        shutil.copy2(FP32 / "scripts" / name, ROOT / "scripts" / name)
    for source, destination in [
        (FP32 / "dataset/test_image_manifest.csv", ROOT / "dataset/test_image_manifest.csv"),
        (FP32 / "dataset/annotation_manifest.csv", ROOT / "dataset/annotation_manifest.csv"),
        (FP32 / "dataset/annotation_validation.json", ROOT / "dataset/annotation_validation.json"),
        (FP32 / "dataset/dataset_yaml_snapshot.yaml", ROOT / "dataset/dataset_yaml_snapshot.yaml"),
        (FP32 / "config/detection_thresholds.json", ROOT / "config/detection_thresholds.json"),
        (FP32 / "config/nms_configuration.json", ROOT / "config/nms_configuration.json"),
        (FP32 / "config/preprocessing.json", ROOT / "config/preprocessing.json"),
        (FP32 / "config/letterbox_configuration.json", ROOT / "config/letterbox_configuration.json"),
        (FP32 / "config/input_shape_decision.json", ROOT / "config/input_shape_decision.json"),
        (FP32 / "raw/official_image_order.txt", ROOT / "reference/fp32_official_image_order.txt"),
        (FP32 / "raw/warmup_execution_provenance.json", ROOT / "reference/fp32_warmup_execution_provenance.json"),
        (FP16 / "raw/official_image_order.txt", ROOT / "reference/fp16_official_image_order.txt"),
        (FP16 / "raw/warmup_execution_provenance.json", ROOT / "reference/fp16_warmup_execution_provenance.json"),
    ]:
        if source.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    source_copy = ROOT / "model/oct_albument2_v6_source.pt"
    onnx_copy = ROOT / "onnx/oct_albument2_v6_inference_only.onnx"
    source_meta = {"path":str(source_copy), "size_bytes":source_copy.stat().st_size, "sha256":sha256(source_copy)}
    onnx_meta = read_json(FP32 / "onnx/onnx_metadata.json")
    onnx_meta.update({"path":str(onnx_copy), "size_bytes":onnx_copy.stat().st_size, "sha256":sha256(onnx_copy), "copied_without_reexport":True})
    if source_meta["sha256"] != fp32["source_sha256"] or onnx_meta["sha256"] != fp32["onnx_sha256"] or onnx_meta["inputs"][0]["shape"] != [1,3,640,960]:
        raise RuntimeError("Shared source/ONNX parity failed")
    write_json(ROOT / "model/source_model_metadata.json", source_meta)
    (ROOT / "model/source_model_sha256.txt").write_text(f"{source_meta['sha256']}  {source_copy.name}\n")
    write_json(ROOT / "onnx/onnx_metadata.json", onnx_meta)
    (ROOT / "onnx/onnx_sha256.txt").write_text(f"{onnx_meta['sha256']}  {onnx_copy.name}\n")
    write_json(ROOT / "validation/shared_source_onnx_parity.json", {"status":"PASS", "source_hash_parity":True, "onnx_hash_parity":True, "input_shape":[1,3,640,960], "embedded_nms":False})
    print(json.dumps({"status":"PASS", "root":str(ROOT), "fp32":fp32, "fp16":fp16}, indent=2))


if __name__ == "__main__":
    main()
