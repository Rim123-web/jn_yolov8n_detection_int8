#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

from common import ROOT, file_metadata, sha256_file, write_csv, write_json


EXCLUDED = {
    "provenance/hash_manifest.csv",
    "provenance/hash_manifest.sha256",
    "validation/final_validation_report.json",
    "validation/final_validation_report.md",
}


def eligible(path: Path) -> bool:
    relative = path.relative_to(ROOT).as_posix()
    return path.is_file() and relative not in EXCLUDED and "__pycache__" not in path.parts and not path.name.endswith((".pyc", ".publication_ready.tar.gz"))


def create():
    write_json(ROOT / "provenance/file_provenance.json", {
        "manifest": "provenance/hash_manifest.csv",
        "manifest_sha256_file": "provenance/hash_manifest.sha256",
        "excluded_self_referential_files": sorted(EXCLUDED),
        "source_model": file_metadata(ROOT / "model/oct_albument2_v6_source.pt"),
        "onnx": file_metadata(ROOT / "onnx/oct_albument2_v6_inference_only.onnx"),
        "engine": file_metadata(ROOT / "engine/oct_albument2_v6_tensorrt_int8_1x3x640x960.engine"),
        "calibration_cache": file_metadata(ROOT / "calibration/oct_albument2_v6_int8_calibration.cache"),
    })
    rows = []
    for path in sorted((p for p in ROOT.rglob("*") if eligible(p)), key=lambda p: p.relative_to(ROOT).as_posix()):
        rows.append({"relative_path": path.relative_to(ROOT).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = ROOT / "provenance/hash_manifest.csv"
    write_csv(manifest, rows, ["relative_path", "size_bytes", "sha256"])
    digest = sha256_file(manifest)
    (ROOT / "provenance/hash_manifest.sha256").write_text(f"{digest}  hash_manifest.csv\n", encoding="utf-8")
    return rows, digest


def verify():
    manifest = ROOT / "provenance/hash_manifest.csv"
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    failures = []
    for row in rows:
        path = ROOT / row["relative_path"]
        if not path.is_file():
            failures.append(f"missing:{row['relative_path']}")
        elif path.stat().st_size != int(row["size_bytes"]):
            failures.append(f"size:{row['relative_path']}")
        elif sha256_file(path) != row["sha256"]:
            failures.append(f"hash:{row['relative_path']}")
    expected = (ROOT / "provenance/hash_manifest.sha256").read_text().split()[0]
    if sha256_file(manifest) != expected:
        failures.append("manifest_sha256")
    return rows, failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        rows, failures = verify()
        print({"status": "PASS" if not failures else "FAIL", "entries": len(rows), "failures": failures})
        if failures:
            raise SystemExit(1)
    else:
        rows, digest = create()
        print({"status": "PASS", "entries": len(rows), "manifest_sha256": digest})


if __name__ == "__main__":
    main()
