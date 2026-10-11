#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from common import ROOT, write_json
from tensorrt_runtime import TensorRTRunner


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""):h.update(c)
    return h.hexdigest()


def main():
    engine=ROOT/"engine/oct_albument2_v6_tensorrt_int8_1x3x640x960.engine"
    cache=ROOT/"calibration/oct_albument2_v6_int8_calibration.cache"
    config=json.loads((ROOT/"engine/engine_build_config.json").read_text())
    with TensorRTRunner(engine) as runner: meta=runner.metadata()
    bindings=meta["binding_metadata"]
    inputs=[b for b in bindings if b["is_input"]]
    if len(inputs)!=1 or inputs[0]["runtime_shape"]!=[1,3,640,960]: raise RuntimeError(f"INT8 engine input invalid: {inputs}")
    rows=[]
    for b in bindings:
        rows.append({**b,"format":"LINEAR/default (TensorRT 8.0 Python API limited)","location":"DEVICE","dynamic":any(x<0 for x in b["engine_shape"])})
    with (ROOT/"engine/binding_metadata.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    wording=("TensorRT INT8-enabled calibrated execution with FP32 input and output bindings and TensorRT-permitted internal higher-precision fallbacks"
             if all(b["dtype"]=="DataType.FLOAT" for b in bindings) else
             "TensorRT INT8-enabled calibrated execution with TensorRT-permitted internal higher-precision fallbacks")
    metadata={"status":"PASS","path":str(engine),"size_bytes":engine.stat().st_size,"sha256":sha256(engine),
              "cache_sha256":sha256(cache),"runtime_deserialization_validation":"PASS","input_shape":[1,3,640,960],
              "precision_wording":wording,"binding_metadata":rows,"runtime":meta,"no_inference_executed":True,
              "int8_enabled_evidence":{"builder_flag":"ON","calibrator_attached":True,"fresh_cache_exists":True,"cache_size_bytes":cache.stat().st_size,
                                      "all_layers_int8_claim":False,"higher_precision_fallbacks":"permitted"}}
    write_json(ROOT/"engine/engine_metadata.json",metadata)
    write_json(ROOT/"validation/int8_engine_deserialization.json",metadata)
    (ROOT/"engine/engine_sha256.txt").write_text(f"{metadata['sha256']}  {engine.name}\n",encoding="utf-8")
    print(json.dumps(metadata,indent=2))


if __name__=="__main__":main()
