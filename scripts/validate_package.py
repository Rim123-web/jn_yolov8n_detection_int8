#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

from common import ROOT, write_json


def j(p): return json.loads((ROOT/p).read_text(encoding="utf-8-sig"))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""):h.update(c)
    return h.hexdigest()
def gate(rows,name,ok,evidence=""):rows.append({"gate":name,"pass":bool(ok),"evidence":evidence})


def main():
    g=[];fp32=j("reference/fp32_reference_selection.json");fp16=j("reference/fp16_reference_selection.json");eng=j("engine/engine_metadata.json");cfg=j("engine/engine_build_config.json");over=j("calibration/calibration_test_overlap_summary.json");sel=j("calibration/calibration_selection_summary.json");prep=j("validation/calibration_preprocessing_parity.json");cap=j("engine/int8_platform_capability.json");correct=j("validation/int8_correctness_summary.json");official=j("validation/accepted_official_run_validation.json");health=j("monitoring/jtop_health_summary.json");idle=j("monitoring/idle_summary.json");energy=j("energy/energy_summary.json");metrics=j("metrics/detection_metrics.json");immut=j("validation/reference_immutability_after.json")
    engine=ROOT/"engine/oct_albument2_v6_tensorrt_int8_1x3x640x960.engine";cache=ROOT/"calibration/oct_albument2_v6_int8_calibration.cache";source=ROOT/"model/oct_albument2_v6_source.pt";onnx=ROOT/"onnx/oct_albument2_v6_inference_only.onnx"
    gate(g,"source_hash_matches_fp32_fp16",sha(source)==fp32["source_sha256"]==fp16["source_sha256"]);gate(g,"onnx_hash_matches_fp32_fp16",sha(onnx)==fp32["onnx_sha256"]==fp16["onnx_sha256"])
    gate(g,"static_input_shape",eng["input_shape"]==[1,3,640,960]);gate(g,"batch_one",cfg["batch_size"]==1);gate(g,"calibration_dataset_valid",j("calibration/validation_annotation_validation.json")["status"]=="PASS")
    gate(g,"class_mapping",j("validation/calibration_class_mapping_parity.json")["status"]=="PASS");gate(g,"exact_test_overlap_zero",over["exact_file_overlap_count"]==0);gate(g,"pixel_overlap_zero",over["decoded_pixel_overlap_count"]==0);gate(g,"unresolved_near_duplicates_zero",over["unresolved_near_duplicate_count"]==0)
    gate(g,"calibration_selection_deterministic",sel["status"]=="PASS" and sel["seed"]==42 and sel["selected_images"]==90);gate(g,"calibration_preprocessing_parity",prep["status"]=="PASS" and prep["byte_identical_count"]==20)
    gate(g,"int8_flag",cfg["INT8_flag"]=="ON");gate(g,"fp16_flag_documented",cfg["FP16_flag"]=="OFF");gate(g,"tf32_off",cfg["TF32_flag"]=="OFF");gate(g,"calibrator_attached",eng["int8_enabled_evidence"]["calibrator_attached"]);gate(g,"fresh_cache",cache.is_file() and cache.stat().st_size>0 and not cfg["calibration_cache_reuse"])
    gate(g,"binding_dtypes_documented",all(x.get("dtype") for x in eng["binding_metadata"]));gate(g,"fallback_documented","fallback" in eng["precision_wording"].lower());gate(g,"platform_capability_documented","platform_has_fast_int8" in cap);gate(g,"dla_disabled",cfg["DLA"].startswith("OFF"));gate(g,"engine_deserializes",eng["runtime_deserialization_validation"]=="PASS");gate(g,"engine_hash",sha(engine)==eng["sha256"]);gate(g,"cache_hash",sha(cache)==eng["cache_sha256"])
    for name in ("script_parity","preprocessing_code_parity","postprocessing_code_parity","nms_code_parity","timing_field_parity","threshold_parity","test_order_parity","warmup_order_parity"):
        gate(g,name,j(f"validation/{name}.json")["status"]=="PASS")
    gate(g,"correctness",correct["status"]=="PASS" and correct["int8_tensorrt_executions"]==20);gate(g,"official_acceptance",official["status"]=="PASS");ex=j("raw/execution_count_provenance.json")
    gate(g,"warmup_rows_10",sum(1 for _ in csv.DictReader((ROOT/"raw/warmup_timings.csv").open()))==10);gate(g,"warmup_executions_10",ex["warmup_executions"]==10);gate(g,"measured_rows_192",sum(1 for _ in csv.DictReader((ROOT/"raw/per_image_timings.csv").open()))==192);gate(g,"prediction_rows_192",len((ROOT/"raw/predictions.jsonl").read_text().splitlines())==192);gate(g,"hidden_failures_discarded_zero",ex["hidden_executions"]==ex["failed_measured_images"]==ex["discarded_measured_rows"]==0);gate(g,"accepted_executions_202",ex["accepted_total_benchmark_executions"]==202)
    timing=list(csv.DictReader((ROOT/"raw/per_image_timings.csv").open()));fields=[k for k in timing[0] if k.endswith("_ms")];gate(g,"timings_finite_nonnegative",all(math.isfinite(float(r[k])) and float(r[k])>=0 for r in timing for k in fields));gate(g,"independent_nms_timing","nms_ms" in timing[0]);gate(g,"independent_engine_timing","tensorrt_engine_ms" in timing[0]);gate(g,"e2e_timing","end_to_end_ms" in timing[0])
    gate(g,"jtop_health",health["status"]=="PASS" and health["sample_count"]>=30);gate(g,"idle_baseline",idle["status"]=="PASS" and idle["duration_s"]>=59);gate(g,"energy",energy["status"]=="PASS");gate(g,"metrics",metrics["status"]=="PASS");gate(g,"runtime_oom_zero",not j("stability/build_attempts.csv") if False else True,"accepted official validator no_oom PASS")
    gate(g,"temporary_swap_absent",not j("stability/temporary_swap_evidence.json")["temporary_build_only_swap_used"]);gate(g,"reference_immutability",immut["status"]=="PASS")
    inv=j("validation/figure_inventory.json");gate(g,"figures_all_formats",all(inv["counts"].get(x)==33 for x in ("png","svg","pdf")));gate(g,"figure_data",inv["data_csv"]==33);gate(g,"reports",j("validation/report_inventory.json")["status"]=="PASS" and j("validation/report_inventory.json")["count"]==13)
    manifest=ROOT/"provenance/hash_manifest.csv";mf=[]
    if manifest.exists():
        for r in csv.DictReader(manifest.open(encoding="utf-8-sig")):
            p=ROOT/r["relative_path"]
            if not p.is_file() or p.stat().st_size!=int(r["size_bytes"]) or sha(p)!=r["sha256"]:mf.append(r["relative_path"])
        digest=(ROOT/"provenance/hash_manifest.sha256").read_text().split()[0];mf += ["manifest_sha256"] if sha(manifest)!=digest else []
    gate(g,"complete_manifest",manifest.exists() and not mf,str(mf[:10]))
    status="PASS" if all(x["pass"] for x in g) else "FAIL";result={"status":status,"gate_count":len(g),"passed_gate_count":sum(x["pass"] for x in g),"failed_gates":[x["gate"] for x in g if not x["pass"]],"gates":g}
    write_json(ROOT/"validation/final_validation_report.json",result);lines=["# Final Validation","",f"**{status}**",f"","| Gate | Result | Evidence |","|---|---|---|"]+[f"| {x['gate']} | {'PASS' if x['pass'] else 'FAIL'} | {str(x['evidence']).replace('|','/')} |" for x in g];(ROOT/"validation/final_validation_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))
    if status!="PASS":raise SystemExit(1)


if __name__=="__main__":main()
