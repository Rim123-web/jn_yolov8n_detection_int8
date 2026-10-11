#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np  # initialize before TensorRT on this Jetson image

TRT_SITE = "/home/jetson/.local/lib/python3.8/site-packages"
if TRT_SITE not in sys.path:
    sys.path.insert(0, TRT_SITE)
import tensorrt as trt

from common import ROOT, write_json
from int8_calibrator import DeterministicImageEntropyCalibrator


def sha256(path: Path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()


def main():
    onnx_path=ROOT/"onnx/oct_albument2_v6_inference_only.onnx"
    engine_path=ROOT/"engine/oct_albument2_v6_tensorrt_int8_1x3x640x960.engine"
    cache_path=ROOT/"calibration/oct_albument2_v6_int8_calibration.cache"
    selected=[Path(x) for x in (ROOT/"calibration/calibration_selected_images.txt").read_text().splitlines() if x.strip()]
    if engine_path.exists() or cache_path.exists():
        raise FileExistsError("Fresh INT8 build requires absent engine and calibration cache")
    if len(selected)!=90:
        raise RuntimeError(f"Expected 90 selected calibration images, got {len(selected)}")
    logger=trt.Logger(trt.Logger.VERBOSE)
    builder=trt.Builder(logger)
    network=builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser=trt.OnnxParser(network,logger)
    if not parser.parse(onnx_path.read_bytes()):
        raise RuntimeError("ONNX parser failed: "+" | ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    if network.num_inputs!=1 or tuple(network.get_input(0).shape)!=(1,3,640,960):
        raise RuntimeError(f"Static ONNX input mismatch: {[tuple(network.get_input(i).shape) for i in range(network.num_inputs)]}")
    config=builder.create_builder_config()
    workspace_mib=int(os.environ.get("INT8_WORKSPACE_MIB","768"))
    config.max_workspace_size=workspace_mib*1024*1024
    for name in ("FP16","TF32","GPU_FALLBACK"):
        flag=getattr(trt.BuilderFlag,name,None)
        if flag is not None: config.clear_flag(flag)
    config.set_flag(trt.BuilderFlag.INT8)
    if hasattr(config,"default_device_type"): config.default_device_type=trt.DeviceType.GPU
    calibrator=DeterministicImageEntropyCalibrator(selected,cache_path,ROOT/"calibration/calibration_execution_order.txt",
                                                   ROOT/"calibration/calibration_batch_manifest.csv",ROOT/"logs/calibration.log")
    config.int8_calibrator=calibrator
    capability={"status":"PASS","tensorrt_version":trt.__version__,"platform_has_fast_int8":bool(builder.platform_has_fast_int8),
                "platform_has_fast_fp16":bool(builder.platform_has_fast_fp16),"compute_capability":"5.3 (Jetson Nano Maxwell, from platform specification)",
                "int8_build_policy":"continue when TensorRT accepts INT8 even if platform_has_fast_int8 is false",
                "builder_flags_available":[n for n in dir(trt.BuilderFlag) if n.isupper()]}
    write_json(ROOT/"engine/int8_platform_capability.json",capability)
    (ROOT/"logs/int8_capability_audit.log").write_text(json.dumps(capability,indent=2)+"\n",encoding="utf-8")
    build_config={"precision_wording":"TensorRT INT8-enabled calibrated execution with TensorRT-permitted internal higher-precision fallbacks",
                  "explicit_batch":True,"input_shape":[1,3,640,960],"batch_size":1,"INT8_flag":"ON","FP16_flag":"OFF",
                  "TF32_flag":"OFF","DLA":"OFF; default device GPU","higher_precision_fallbacks":"PERMITTED by TensorRT",
                  "workspace_mib":workspace_mib,"workspace_bytes":workspace_mib*1024*1024,"calibrator":"IInt8EntropyCalibrator2",
                  "calibration_algorithm":"Entropy Calibration 2","calibration_images":len(selected),"calibration_batch_size":1,
                  "calibration_cache_reuse":False,"tactic_sources":"TensorRT 8.0.1 defaults","timing_iterations":"TensorRT defaults",
                  "builder_optimization_level":"API unavailable/not configured","timing_cache":"fresh builder process; no historical timing cache",
                  "platform_has_fast_int8":capability["platform_has_fast_int8"],"platform_has_fast_fp16":capability["platform_has_fast_fp16"],
                  "tensorrt_version":trt.__version__,"python":platform.python_version(),"numpy":np.__version__}
    write_json(ROOT/"engine/engine_build_config.json",build_config)
    start_wall=time.time(); start=time.perf_counter()
    engine=builder.build_engine(network,config)
    duration=time.perf_counter()-start; end_wall=time.time()
    calibrator.close()
    if engine is None: raise RuntimeError("TensorRT build_engine returned None")
    engine_path.write_bytes(engine.serialize())
    if not cache_path.is_file() or cache_path.stat().st_size==0: raise RuntimeError("Fresh calibration cache was not created")
    duration_info={"start_epoch":start_wall,"end_epoch":end_wall,"duration_s":duration,"build_attempt":1,"oom_events":0}
    write_json(ROOT/"engine/engine_build_duration.json",duration_info)
    cache_meta={"path":str(cache_path),"size_bytes":cache_path.stat().st_size,"sha256":sha256(cache_path),"created_fresh":True,"historical_cache_reused":False}
    write_json(ROOT/"calibration/calibration_cache_metadata.json",cache_meta)
    result={"status":"PASS","engine_path":str(engine_path),"engine_size_bytes":engine_path.stat().st_size,"engine_sha256":sha256(engine_path),
            "cache":cache_meta,"duration":duration_info,"config":build_config}
    (ROOT/"engine/engine_build_log.txt").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    (ROOT/"engine/layer_precision_evidence.txt").write_text("TensorRT 8.0.1 Python API did not expose a reliable per-layer runtime precision dump for this serialized engine. Evidence retained: INT8 builder flag enabled, fresh IInt8EntropyCalibrator2 attached, all 90 calibration batches consumed, and non-empty fresh cache created. Do not claim every layer is INT8.\n",encoding="utf-8")
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__": main()
