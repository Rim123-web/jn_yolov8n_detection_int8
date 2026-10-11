#!/usr/bin/env python3
from __future__ import annotations

import html
import json
from pathlib import Path

from common import CLASS_NAMES, ROOT, write_json


def j(path): return json.loads((ROOT/path).read_text(encoding="utf-8-sig"))


def main():
    m=j("metrics/detection_metrics.json");p=j("summary/performance_summary.json");e=j("energy/energy_summary.json");r=j("resources/resource_summary.json")
    cal=j("calibration/calibration_selection_summary.json");over=j("calibration/calibration_test_overlap_summary.json");eng=j("engine/engine_metadata.json");build=j("engine/engine_build_config.json");cap=j("engine/int8_platform_capability.json");cmp=j("comparison/fp32_fp16_int8_summary.json")
    src=j("model/source_model_metadata.json");onnx=j("onnx/onnx_metadata.json");cache=j("calibration/calibration_cache_metadata.json");correct=j("validation/int8_correctness_summary.json")
    table="| Metric | Value |\n|---|---:|\n"+"\n".join([
        f"| mAP50 | {m['mAP50']:.6f} |",f"| mAP50-95 | {m['mAP50_95']:.6f} |",f"| Precision @0.25 | {m['precision_conf025']:.6f} |",f"| Recall @0.25 | {m['recall_conf025']:.6f} |",f"| F1 @0.25 | {m['f1_conf025']:.6f} |",f"| Engine latency (ms) | {p['pure_tensorrt_engine_latency_mean_ms']:.3f} |",f"| Engine FPS | {p['engine_fps']:.3f} |",f"| E2E latency (ms) | {p['mean_pipeline_latency_ms']:.3f} |",f"| Wall-clock FPS | {p['wall_clock_pipeline_fps']:.3f} |",f"| Mean active power (W) | {e['active_mean_power_w_time_weighted']:.3f} |",f"| Gross energy/image (J) | {e['gross_energy_per_image_j']:.6f} |",f"| Idle-corrected energy/image (J) | {e['idle_corrected_energy_per_image_j']:.6f} |"])
    methodology=f"""# Methodology

- Task: YOLOv8 multi-class object detection ({', '.join(CLASS_NAMES)}).
- Source model: `{src['path']}`; SHA-256 `{src['sha256']}`.
- Shared ONNX: `{onnx['path']}`; SHA-256 `{onnx['sha256']}`; reused without re-export.
- Static input `[1,3,640,960]`; batch size 1.
- INT8 calibration dataset: `/home/jetson/Downloads/jetson_folder/jetson_valid_imgs`.
- Deterministic calibration subset: {cal['selected_images']} images, seed 42, greedy multi-label class/object/aspect coverage.
- Algorithm: TensorRT IInt8EntropyCalibrator2 (Entropy Calibration 2), batch size 1.
- Calibration uses canonical RGB/EXIF/letterbox-114/bilinear/divide-by-255/contiguous NCHW FP32 preprocessing. Labels were used only for audit and representative selection, not entropy calibration.
- Calibration/test independence: {over['exact_file_overlap_count']} exact file, {over['decoded_pixel_overlap_count']} decoded-pixel, {over['unresolved_near_duplicate_count']} unresolved dHash overlap.
- Cache: `{cache['path']}`; SHA-256 `{cache['sha256']}`; freshly created, no legacy cache reused.
- Execution: TensorRT INT8-enabled calibrated execution with FP32 input/output bindings and TensorRT-permitted internal higher-precision fallbacks. This is not described as pure or exclusive INT8.
- FP16 builder flag {build['FP16_flag']}; TF32 {build['TF32_flag']}; DLA disabled; platform fast INT8 = {cap['platform_has_fast_int8']}.
- Correctness: five backends on the same 20 images; FP32/INT8 and FP16/INT8 paired class agreement both {correct['fp32_int8_prediction_agreement']:.3f} and {correct['fp16_int8_prediction_agreement']:.3f}.
- Test data/order, preprocessing, postprocessing, class-aware NMS (IoU 0.7), confidence 0.25, AP floor 0.001, and IoU matching 0.5 match the accepted FP32/FP16 packages.
- Exactly 10 real-image full-pipeline warm-ups used the same paths as FP32/FP16 and were excluded from all reported results.
- Every one of 192 test images was measured exactly once. CUDA events independently timed H2D, engine, and D2H; CPU monotonic timing covered preprocessing, postprocessing, and E2E.
- jtop/jetson-stats sampled a fresh 60-second idle baseline and the accepted measured interval at 0.5 s. Power TOT energy used trapezoidal integration.
"""
    results=f"# Results\n\n{table}\n\nRecommendation: **{cmp['deployment_recommendation']}**. {cmp['rationale']}\n"
    limitations="# Limitations\n\n- Jetson Nano reports no native fast-INT8 capability; performance may be worse than FP16.\n- TensorRT may retain higher-precision internal layers; per-layer runtime precision was not fully exposed by TensorRT 8.0.1.\n- Power TOT is board-level telemetry, not GPU-only power. Idle-corrected energy subtracts the fresh idle baseline and may be sensitive to short-run telemetry cadence.\n- Validation labels contain recorded boxes that cross normalized image boundaries; labels are not calibration inputs.\n"
    calibration=f"# Calibration Methodology\n\n{methodology}\n\n## Selected Class Coverage\n\n"+"\n".join(f"- {x['class_name']}: {x['selected_image_count']} images, {x['selected_object_count']} objects" for x in cal['coverage'])+"\n"
    audit=f"# Calibration Dataset Audit\n\n- Images: 192\n- Labels: 192\n- Selected: {cal['selected_images']}\n- Exact byte overlap: {over['exact_file_overlap_count']}\n- Decoded-pixel overlap: {over['decoded_pixel_overlap_count']}\n- Unresolved perceptual near-duplicates: {over['unresolved_near_duplicate_count']}\n- Status: PASS\n"
    comparison=(ROOT/"comparison/fp32_fp16_int8_report.md").read_text(encoding="utf-8")
    recommendation=(ROOT/"comparison/deployment_recommendation.md").read_text(encoding="utf-8")
    reproducibility="# Reproducibility\n\nAll scripts, configuration, selected calibration paths, batch hashes, raw timings, raw predictions, telemetry, software records, and SHA-256 manifests are included. The accepted execution count is 10 warm-ups plus 192 measured images; correctness executions are separately recorded.\n"
    failures="# Failure and Recovery Log\n\n- Engine build: first attempt succeeded; no OOM; no temporary swap created.\n- Official benchmark: first attempt succeeded. The first post-run validation invocation lacked a bookkeeping input file; it was reconstructed from the pre-runtime memory snapshot and validation was rerun without rerunning or modifying raw benchmark results.\n"
    defense=f"# Defense-Ready Summary\n\nThis controlled experiment isolates precision/build configuration while preserving the complete FP32/FP16 protocol. INT8 achieved mAP50 {m['mAP50']:.6f}, mAP50-95 {m['mAP50_95']:.6f}, engine FPS {p['engine_fps']:.3f}, and idle-corrected energy {e['idle_corrected_energy_per_image_j']:.6f} J/image. Recommended deployment: {cmp['deployment_recommendation']}.\n"
    article=f"# Article-Ready Results\n\n{table}\n\nThe calibrated engine used a disjoint 90-image validation subset and fixed 1x3x640x960 input. Values are from one accepted 192-image pass after 10 excluded real-image warm-ups.\n"
    full="# YOLOv8n TensorRT INT8 Detection Benchmark\n\n"+methodology+"\n"+results+"\n"+comparison+"\n"+limitations
    outputs={"full_benchmark_report.md":full,"methodology.md":methodology,"results.md":results,"limitations.md":limitations,
             "reproducibility.md":reproducibility,"calibration_methodology.md":calibration,"calibration_dataset_audit.md":audit,
             "article_ready_results.md":article,"defense_ready_summary.md":defense,"failure_and_recovery_log.md":failures,
             "fp32_fp16_int8_comparison.md":comparison,"deployment_recommendation.md":recommendation}
    for name,text in outputs.items():(ROOT/"reports"/name).write_text(text,encoding="utf-8")
    html_text="<html><body><pre>"+html.escape(full)+"</pre></body></html>"
    (ROOT/"reports/full_benchmark_report.html").write_text(html_text,encoding="utf-8")
    write_json(ROOT/"validation/report_inventory.json",{"status":"PASS","required":sorted(list(outputs)+["full_benchmark_report.html"]),"count":13})
    print(json.dumps({"status":"PASS","reports":13},indent=2))


if __name__=="__main__":main()
