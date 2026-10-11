#!/usr/bin/env python3
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np

from common import ROOT, append_timeline, collect_images, write_csv, write_json
from postprocessing import postprocess_yolov8
from preprocessing import preprocess_image
from tensorrt_runtime import TensorRTRunner


def box_iou(a,b):
    x1,y1=max(a[0],b[0]),max(a[1],b[1]);x2,y2=min(a[2],b[2]),min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1); aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]);bb=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/max(aa+bb-inter,1e-12)


def compare(reference,candidate):
    candidates=sorted(((box_iou(r["box"],c["box"]),i,j) for i,r in enumerate(reference) for j,c in enumerate(candidate)),key=lambda x:(-x[0],x[1],x[2]))
    ur,uc,pairs=set(),set(),[]
    for iou,i,j in candidates:
        if i in ur or j in uc: continue
        ur.add(i);uc.add(j);pairs.append((i,j,iou))
        if len(pairs)==min(len(reference),len(candidate)):break
    conf=[abs(reference[i]["confidence"]-candidate[j]["confidence"]) for i,j,_ in pairs]
    box=[max(abs(a-b) for a,b in zip(reference[i]["box"],candidate[j]["box"])) for i,j,_ in pairs]
    mean_box=[sum(abs(a-b) for a,b in zip(reference[i]["box"],candidate[j]["box"]))/4 for i,j,_ in pairs]
    return {"matching_method":"deterministic greedy maximum-IoU spatial pairing","matched_detection_count":len(pairs),
            "unmatched_detection_count":abs(len(reference)-len(candidate)),"class_agreement_count":sum(reference[i]["class_id"]==candidate[j]["class_id"] for i,j,_ in pairs),
            "minimum_matched_iou":min((q for _,_,q in pairs),default=1.0),"max_confidence_difference":max(conf,default=0.0),
            "mean_confidence_difference":float(np.mean(conf)) if conf else 0.0,"max_box_coordinate_difference_px":max(box,default=0.0),
            "mean_box_coordinate_difference_px":float(np.mean(mean_box)) if mean_box else 0.0}


def valid(ds,meta):
    for d in ds:
        x1,y1,x2,y2=d["box"]
        if d["class_id"] not in range(6) or not np.isfinite(d["confidence"]) or not 0<=d["confidence"]<=1:return False
        if not all(np.isfinite(x) for x in d["box"]) or x2<x1 or y2<y1:return False
        if x1<-1e-4 or y1<-1e-4 or x2>meta["original_width"]+1e-4 or y2>meta["original_height"]+1e-4:return False
    return True


def run_tensorrt(engine, arrays, metas):
    out=[]
    with TensorRTRunner(engine) as runner:
        for a,m in zip(arrays,metas):
            raw=runner.run_timed(a)[0][0]
            if not np.isfinite(raw).all(): raise RuntimeError(f"Nonfinite raw output: {engine}")
            out.append(postprocess_yolov8(raw,m)[0])
    return out


def main():
    append_timeline("int8_correctness","START","20 shared correctness images; five backends executed sequentially")
    fp32=Path(json.loads((ROOT/"reference/fp32_reference_selection.json").read_text())["selected_path"])
    fp16=Path(json.loads((ROOT/"reference/fp16_reference_selection.json").read_text())["selected_path"])
    expected=[r["image_path"] for r in json.loads((fp32/"validation/fp32_correctness_summary.json").read_text())["rows"]]
    images=collect_images()[:20]
    if [str(p.resolve()) for p in images]!=expected:raise RuntimeError("Correctness image set/order differs from references")
    arrays=[];metas=[]
    for p in images:
        a,m,_=preprocess_image(p);arrays.append(a);metas.append(m)
    detections={}
    import torch
    from ultralytics import YOLO
    model=YOLO(str(ROOT/"model/oct_albument2_v6_source.pt")).model.to("cuda:0").float().eval()
    values=[]
    for a,m in zip(arrays,metas):
        t=torch.from_numpy(a).to("cuda:0",dtype=torch.float32);torch.cuda.synchronize()
        with torch.inference_mode(): raw=model(t)
        torch.cuda.synchronize();raw=raw[0] if isinstance(raw,(tuple,list)) else raw
        values.append(postprocess_yolov8(raw.detach().cpu().numpy(),m)[0])
    detections["PyTorch FP32"]=values
    del model;torch.cuda.empty_cache();gc.collect()
    import onnxruntime as ort
    session=ort.InferenceSession(str(ROOT/"onnx/oct_albument2_v6_inference_only.onnx"),providers=["CPUExecutionProvider"])
    name=session.get_inputs()[0].name;outname=session.get_outputs()[0].name
    detections["ONNX FP32"]=[postprocess_yolov8(session.run([outname],{name:a})[0],m)[0] for a,m in zip(arrays,metas)]
    del session;gc.collect()
    detections["TensorRT FP32"]=run_tensorrt(fp32/"engine/oct_albument2_v6_tensorrt_fp32_1x3x640x960.engine",arrays,metas)
    gc.collect()
    detections["TensorRT FP16-enabled"]=run_tensorrt(fp16/"engine/oct_albument2_v6_tensorrt_fp16_1x3x640x960.engine",arrays,metas)
    gc.collect()
    detections["TensorRT INT8-enabled"]=run_tensorrt(ROOT/"engine/oct_albument2_v6_tensorrt_int8_1x3x640x960.engine",arrays,metas)
    rows=[];pred_rows=[]
    for idx,(p,m) in enumerate(zip(images,metas),1):
        per={k:v[idx-1] for k,v in detections.items()};a=compare(per["TensorRT FP32"],per["TensorRT INT8-enabled"]);b=compare(per["TensorRT FP16-enabled"],per["TensorRT INT8-enabled"])
        finite_valid=all(valid(v,m) for v in per.values())
        rows.append({"sample_index":idx,"image_path":str(p.resolve()),"source_detection_count":len(per["PyTorch FP32"]),"onnx_detection_count":len(per["ONNX FP32"]),
                     "fp32_detection_count":len(per["TensorRT FP32"]),"fp16_detection_count":len(per["TensorRT FP16-enabled"]),"int8_detection_count":len(per["TensorRT INT8-enabled"]),
                     **{f"fp32_int8_{k}":v for k,v in a.items()},**{f"fp16_int8_{k}":v for k,v in b.items()},"finite_valid":finite_valid})
        for backend,ds in per.items():
            for j,d in enumerate(ds): pred_rows.append({"sample_index":idx,"image_path":str(p.resolve()),"backend":backend,"detection_index":j,"class_id":d["class_id"],"class_name":d["class_name"],"confidence":d["confidence"],"x1":d["box"][0],"y1":d["box"][1],"x2":d["box"][2],"y2":d["box"][3]})
        print(f"INT8 correctness comparison {idx}/20",flush=True)
    fp32_matches=sum(r["fp32_int8_class_agreement_count"] for r in rows);fp32_pairs=sum(r["fp32_int8_matched_detection_count"] for r in rows)
    fp16_matches=sum(r["fp16_int8_class_agreement_count"] for r in rows);fp16_pairs=sum(r["fp16_int8_matched_detection_count"] for r in rows)
    fp32_rate=fp32_matches/max(1,fp32_pairs);fp16_rate=fp16_matches/max(1,fp16_pairs)
    int8_count=sum(r["int8_detection_count"] for r in rows);ref_count=sum(r["fp32_detection_count"] for r in rows)
    status="PASS" if all(r["finite_valid"] for r in rows) and fp32_rate>=0.90 and fp16_rate>=0.90 and int8_count>=0.5*max(1,ref_count) else "FAIL"
    summary={"status":status,"sample_count":20,"source_pt_executions":20,"onnxruntime_executions":20,"fp32_tensorrt_executions":20,"fp16_tensorrt_executions":20,"int8_tensorrt_executions":20,
             "fp32_int8_prediction_agreement":fp32_rate,"fp16_int8_prediction_agreement":fp16_rate,"int8_detection_count":int8_count,"fp32_detection_count":ref_count,
             "acceptance":"finite valid outputs, >=90% spatially paired class agreement to FP32 and FP16, and INT8 count >=50% of FP32","rows":rows}
    write_csv(ROOT/"validation/int8_correctness_predictions.csv",pred_rows)
    write_json(ROOT/"validation/int8_correctness_summary.json",summary)
    write_json(ROOT/"validation/fp32_fp16_int8_correctness_comparison.json",summary)
    (ROOT/"logs/int8_correctness.log").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    append_timeline("int8_correctness",status,"five backends x20; sequential; separate from benchmark")
    if status!="PASS":raise RuntimeError("INT8 correctness gate failed")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
