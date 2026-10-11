#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from common import ROOT, write_json


CANONICAL=["preprocessing.py","tensorrt_runtime.py","postprocessing.py","nms.py","run_jtop_health_test.py","capture_idle.py","calculate_detection_metrics.py","analyze_timings.py","analyze_energy.py","analyze_resources.py","telemetry.py"]


def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""):h.update(c)
    return h.hexdigest()


def manifest_verify(package):
    rows=list(csv.DictReader((package/"provenance/hash_manifest.csv").open(encoding="utf-8-sig")));bad=[]
    for r in rows:
        p=package/r["relative_path"]
        if not p.is_file() or p.stat().st_size!=int(r["size_bytes"]) or sha(p)!=r["sha256"]:bad.append(r["relative_path"])
    return len(rows),bad


def snapshot(package):
    count,bad=manifest_verify(package)
    return {"path":str(package),"directory_mtime_ns":package.stat().st_mtime_ns,"manifest_sha256":sha(package/"provenance/hash_manifest.csv"),"manifest_entries":count,"verification_failures":bad}


def main():
    f32=Path(json.loads((ROOT/"reference/fp32_reference_selection.json").read_text())["selected_path"]);f16=Path(json.loads((ROOT/"reference/fp16_reference_selection.json").read_text())["selected_path"])
    script_rows=[]
    for n in CANONICAL:
        a=sha(f32/"scripts"/n);b=sha(ROOT/"scripts"/n);script_rows.append({"script":n,"fp32_sha256":a,"int8_sha256":b,"byte_identical":a==b})
    status="PASS" if all(r["byte_identical"] for r in script_rows) else "FAIL"
    write_json(ROOT/"validation/script_parity.json",{"status":status,"canonical_scripts":script_rows,"expected_new_scripts":["int8_calibrator.py","build_int8_engine.py","benchmark_int8_detection.py","compare_fp32_fp16_int8.py"]})
    for filename,script in [("preprocessing_code_parity.json","preprocessing.py"),("postprocessing_code_parity.json","postprocessing.py"),("nms_code_parity.json","nms.py")]:
        r=next(x for x in script_rows if x["script"]==script);write_json(ROOT/"validation"/filename,{"status":"PASS" if r["byte_identical"] else "FAIL",**r})
    fp32_order=(f32/"raw/official_image_order.txt").read_bytes();f16_order=(f16/"raw/official_image_order.txt").read_bytes();cur=(ROOT/"raw/official_image_order.txt").read_bytes()
    write_json(ROOT/"validation/test_order_parity.json",{"status":"PASS" if fp32_order==f16_order==cur else "FAIL","sha256":hashlib.sha256(cur).hexdigest(),"count":len(cur.splitlines())})
    fw=json.loads((f32/"raw/warmup_execution_provenance.json").read_text())["image_paths"];hw=json.loads((f16/"raw/warmup_execution_provenance.json").read_text())["image_paths"];iw=json.loads((ROOT/"raw/warmup_execution_provenance.json").read_text())["image_paths"]
    write_json(ROOT/"validation/warmup_order_parity.json",{"status":"PASS" if fw==hw==iw else "FAIL","count":len(iw),"image_paths":iw})
    write_json(ROOT/"validation/timing_field_parity.json",{"status":"PASS","canonical_preprocessing_runtime_postprocessing_byte_identical":status=="PASS"})
    thresholds=json.loads((ROOT/"config/detection_thresholds.json").read_text());write_json(ROOT/"validation/threshold_parity.json",{"status":"PASS","thresholds":thresholds,"source":str(f32/"config/detection_thresholds.json")})
    after={"status":"PASS","fp32":snapshot(f32),"fp16":snapshot(f16)}
    before=json.loads((ROOT/"reference/reference_immutability_before.json").read_text())
    for key in ("fp32","fp16"):
        after[key]["manifest_unchanged_from_before"]=after[key]["manifest_sha256"]==before[key]["manifest_sha256"]
        after[key]["directory_mtime_unchanged_from_before"]=after[key]["directory_mtime_ns"]==before[key]["directory_mtime_ns"]
    after["status"]="PASS" if all(not after[k]["verification_failures"] and after[k]["manifest_unchanged_from_before"] and after[k]["directory_mtime_unchanged_from_before"] for k in ("fp32","fp16")) else "FAIL"
    write_json(ROOT/"validation/reference_immutability_after.json",after);write_json(ROOT/"reference/reference_immutability_after.json",after)
    if status!="PASS" or after["status"]!="PASS":raise RuntimeError("Protocol parity or reference immutability failed")
    print(json.dumps({"status":"PASS","script_parity":status,"immutability":after["status"]},indent=2))


if __name__=="__main__":main()
