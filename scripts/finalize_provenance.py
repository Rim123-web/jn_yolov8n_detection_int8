#!/usr/bin/env python3
from __future__ import annotations

import json

from common import ROOT, append_timeline, read_json, utc_now


def main():
    system = read_json(ROOT / "environment/system_information.json", {})
    software = read_json(ROOT / "environment/software_versions.json", {})
    python_environment = (ROOT / "environment/python_environment.txt").read_text(encoding="utf-8", errors="replace")
    lines = [
        "YOLOv8n detection TensorRT INT8-enabled benchmark software environment",
        f"Generated: {utc_now()}",
        "",
        "Approved runtime environment variables (credentials excluded):",
        "PYTHONNOUSERSITE=1",
        f"BENCHMARK_DIR={ROOT}",
        "PYTHONPATH=<result>/scripts for runtime; existing jetson_benchmark_env site-packages added only for ONNX/ONNXRuntime phases; known TensorRT user-site inserted narrowly by package-local TensorRT modules",
        "",
        "System information JSON:",
        json.dumps(system, indent=2, ensure_ascii=False),
        "",
        "Software versions JSON:",
        json.dumps(software, indent=2, ensure_ascii=False),
        "",
        "Python environment evidence:",
        python_environment.rstrip(),
        "",
        "Security statement: no password, SSH key, credential, token, or secret is stored in this file or benchmark package.",
    ]
    (ROOT / "provenance/software_environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    append_timeline("provenance_finalization", "PASS", "software environment and approved variables recorded; secrets excluded")
    print(ROOT / "provenance/software_environment.txt")


if __name__ == "__main__":
    main()
