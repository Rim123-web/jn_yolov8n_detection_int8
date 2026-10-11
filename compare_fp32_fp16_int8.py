#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import ROOT, write_csv, write_json


def load_json(p): return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def group_predictions(path):
    out=defaultdict(list)
    for r in csv.DictReader(Path(path).open(encoding="utf-8-sig")):
        out[r["image_path"]].append({"class_id":int(r["class_id"]),"confidence":float(r["confidence"]),"box":[float(r[k]) for k in ("x1","y1","x2","y2")]})
    return out


def iou(a,b):
    x1,y1=max(a[0],b[0]),max(a[1],b[1]);x2,y2=min(a[2],b[2]),min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1);aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]);bb=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/max(aa+bb-inter,1e-12)


def match(a,b):
    pairs=[];ua=set();ub=set()
    cand=sorted(((iou(x["box"],y["box"]),i,j) for i,x in enumerate(a) for j,y in enumerate(b)),key=lambda z:(-z[0],z[1],z[2]))
    for q,i,j in cand:
        if i in ua or j in ub:continue
        ua.add(i);ub.add(j);pairs.append((i,j,q))
        if len(pairs)==min(len(a),len(b)):break
    conf=[abs(a[i]["confidence"]-b[j]["confidence"]) for i,j,_ in pairs]
    coord=[max(abs(x-y) for x,y in zip(a[i]["box"],b[j]["box"])) for i,j,_ in pairs]
    return {"matched":len(pairs),"unmatched":abs(len(a)-len(b)),"class_agreement":sum(a[i]["class_id"]==b[j]["class_id"] for i,j,_ in pairs),
            "mean_confidence_difference":float(np.mean(conf)) if conf else 0.0,"max_confidence_difference":max(conf,default=0.0),
            "mean_coordinate_difference_px":float(np.mean(coord)) if coord else 0.0,"max_coordinate_difference_px":max(coord,default=0.0)}


def row(name,root):
    m=load_json(root/"metrics/detection_metrics.json");p=load_json(root/"summary/performance_summary.json");e=load_json(root/"energy/energy_summary.json");r=load_json(root/"resources/resource_summary.json")
    return {"precision":name,"mAP50":m["mAP50"],"mAP50_95":m["mAP50_95"],"precision_conf025":m["precision_conf025"],"recall_conf025":m["recall_conf025"],"f1_conf025":m["f1_conf025"],
            "engine_latency_ms":p["pure_tensorrt_engine_latency_mean_ms"],"transfer_inclusive_latency_ms":p["transfer_inclusive_latency_mean_ms"],"e2e_latency_ms":p["mean_pipeline_latency_ms"],
            "engine_fps":p["engine_fps"],"transfer_inclusive_fps":p["transfer_inclusive_fps"],"latency_pipeline_fps":p["latency_derived_pipeline_fps"],"wall_clock_fps":p["wall_clock_pipeline_fps"],
            "active_power_w":e["active_mean_power_w_time_weighted"],"gross_energy_per_image_j":e["gross_energy_per_image_j"],"idle_corrected_energy_per_image_j":e["idle_corrected_energy_per_image_j"],
            "cpu_mean_percent":r["cpu_mean_percent"],"gpu_mean_percent":r["gpu_mean_percent"],"ram_mean_mib":r["ram_mean_mib"],"process_rss_mean_mib":r["process_rss_mean_mib"]}


def main():
    fp32=Path(load_json(ROOT/"reference/fp32_reference_selection.json")["selected_path"]);fp16=Path(load_json(ROOT/"reference/fp16_reference_selection.json")["selected_path"])
    rows=[row("FP32",fp32),row("FP16",fp16),row("INT8",ROOT)]
    write_csv(ROOT/"comparison/fp32_fp16_int8_metrics.csv",rows)
    timing=[{k:v for k,v in r.items() if k=="precision" or "latency" in k or "fps" in k} for r in rows]
    energy=[{k:v for k,v in r.items() if k=="precision" or "power" in k or "energy" in k} for r in rows]
    write_csv(ROOT/"comparison/fp32_fp16_int8_timing.csv",timing);write_csv(ROOT/"comparison/fp32_fp16_int8_energy.csv",energy)
    pred={"FP32":group_predictions(fp32/"raw/predictions.csv"),"FP16":group_predictions(fp16/"raw/predictions.csv"),"INT8":group_predictions(ROOT/"raw/predictions.csv")}
    per=[]
    for path in sorted(pred["INT8"]):
        a=match(pred["FP32"][path],pred["INT8"][path]);b=match(pred["FP16"][path],pred["INT8"][path])
        per.append({"image_path":path,"fp32_count":len(pred["FP32"][path]),"fp16_count":len(pred["FP16"][path]),"int8_count":len(pred["INT8"][path]),
                    **{f"fp32_int8_{k}":v for k,v in a.items()},**{f"fp16_int8_{k}":v for k,v in b.items()}})
    write_csv(ROOT/"comparison/fp32_fp16_int8_per_image.csv",per)
    f32=sum(x["fp32_int8_class_agreement"] for x in per)/max(1,sum(x["fp32_int8_matched"] for x in per));f16=sum(x["fp16_int8_class_agreement"] for x in per)/max(1,sum(x["fp16_int8_matched"] for x in per))
    int8=rows[2];fp16r=rows[1]
    recommend_int8=(int8["mAP50_95"]>=fp16r["mAP50_95"]-0.02 and int8["engine_fps"]>fp16r["engine_fps"] and int8["idle_corrected_energy_per_image_j"]<fp16r["idle_corrected_energy_per_image_j"])
    recommendation=("INT8" if recommend_int8 else "FP16")
    rationale=("INT8 improves engine speed and idle-corrected energy while preserving mAP50-95 within 0.02." if recommend_int8 else
               "FP16 remains preferred because INT8 does not simultaneously improve speed and energy while preserving mAP50-95; the Nano reports no native fast-INT8 support.")
    summary={"status":"PASS","rows":rows,"fp32_int8_prediction_agreement":f32,"fp16_int8_prediction_agreement":f16,
             "int8_minus_fp32":{"mAP50":rows[2]["mAP50"]-rows[0]["mAP50"],"mAP50_95":rows[2]["mAP50_95"]-rows[0]["mAP50_95"],"engine_fps_percent":100*(rows[2]["engine_fps"]/rows[0]["engine_fps"]-1),"energy_percent":100*(rows[2]["idle_corrected_energy_per_image_j"]/rows[0]["idle_corrected_energy_per_image_j"]-1)},
             "int8_minus_fp16":{"mAP50":rows[2]["mAP50"]-rows[1]["mAP50"],"mAP50_95":rows[2]["mAP50_95"]-rows[1]["mAP50_95"],"engine_fps_percent":100*(rows[2]["engine_fps"]/rows[1]["engine_fps"]-1),"energy_percent":100*(rows[2]["idle_corrected_energy_per_image_j"]/rows[1]["idle_corrected_energy_per_image_j"]-1)},
             "deployment_recommendation":recommendation,"rationale":rationale}
    write_json(ROOT/"comparison/fp32_fp16_int8_summary.json",summary)
    write_json(ROOT/"comparison/deployment_recommendation.json",{"status":"PASS","recommended_precision":recommendation,"rationale":rationale,"platform_has_fast_int8":False})
    md="# FP32 / FP16 / INT8 Comparison\n\n| Precision | mAP50 | mAP50-95 | Engine ms | Engine FPS | E2E ms | Gross J/image | Net J/image |\n|---|---:|---:|---:|---:|---:|---:|---:|\n"+"\n".join(f"| {r['precision']} | {r['mAP50']:.6f} | {r['mAP50_95']:.6f} | {r['engine_latency_ms']:.3f} | {r['engine_fps']:.3f} | {r['e2e_latency_ms']:.3f} | {r['gross_energy_per_image_j']:.6f} | {r['idle_corrected_energy_per_image_j']:.6f} |" for r in rows)+f"\n\n**Recommendation: {recommendation}.** {rationale}\n"
    (ROOT/"comparison/fp32_fp16_int8_report.md").write_text(md,encoding="utf-8");(ROOT/"comparison/deployment_recommendation.md").write_text(f"# Deployment Recommendation\n\n**{recommendation}**\n\n{rationale}\n",encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
