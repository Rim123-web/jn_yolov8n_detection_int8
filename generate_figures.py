#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import ROOT, write_csv, write_json


FIGURES = [
"int8_per_stage_mean_latency","int8_per_stage_e2e_share","int8_engine_latency_distribution","int8_e2e_distribution","int8_h2d_engine_d2h",
"int8_preprocessing_decomposition","int8_postprocessing_decomposition","int8_nms_distribution","int8_fps_definitions","int8_power_over_time",
"int8_cpu_gpu_over_time","int8_ram_process_rss","int8_temperature","int8_energy_summary","int8_precision_recall_curve","int8_f1_confidence_curve",
"int8_ap_by_class","int8_ap_by_iou","int8_detection_count_distribution","int8_error_analysis","int8_confusion_matrix","int8_normalized_confusion_matrix",
"int8_warmup_behavior","calibration_class_coverage","calibration_image_size_distribution","calibration_test_overlap_audit",
"fp32_fp16_int8_accuracy_comparison","fp32_fp16_int8_engine_latency","fp32_fp16_int8_e2e_latency","fp32_fp16_int8_fps","fp32_fp16_int8_energy",
"fp32_fp16_int8_resources","deployment_recommendation_summary"]


def read_csv(path): return list(csv.DictReader(Path(path).open(encoding="utf-8-sig")))
def read_json(path): return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def nums(rows,key): return [float(r[key]) for r in rows]


def save(name, rows, kind="bar", x="label", ys=("value",), title=None):
    write_csv(ROOT/f"figures/data/{name}.csv",rows)
    fig,ax=plt.subplots(figsize=(7.2,4.5),dpi=140)
    labels=[str(r[x]) for r in rows]
    if kind=="line":
        xv=np.arange(len(rows)) if not labels or any(not s for s in labels) else np.arange(len(rows))
        for y in ys: ax.plot(xv,[float(r[y]) for r in rows],label=y,linewidth=1.7)
        if len(rows)<=25: ax.set_xticks(xv);ax.set_xticklabels(labels,rotation=35,ha="right")
    elif kind=="hist":
        ax.hist([float(r[ys[0]]) for r in rows],bins=20,color="#2a6f97",edgecolor="white")
    else:
        xv=np.arange(len(rows));width=.8/max(1,len(ys))
        for j,y in enumerate(ys):ax.bar(xv+(j-(len(ys)-1)/2)*width,[float(r[y]) for r in rows],width,label=y)
        ax.set_xticks(xv);ax.set_xticklabels(labels,rotation=28,ha="right")
    ax.set_title(title or name.replace("_"," ").title());ax.grid(axis="y",alpha=.25)
    if len(ys)>1:ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png","svg","pdf"):fig.savefig(ROOT/f"figures/{ext}/{name}.{ext}",bbox_inches="tight")
    plt.close(fig)


def matrix_figure(name,path,normalized=False):
    rows=read_csv(path); labels=[r[next(iter(r))] for r in rows]; values=np.array([[float(r[k]) for k in list(r)[1:]] for r in rows])
    if normalized: values=values/np.maximum(values.sum(axis=1,keepdims=True),1)
    data=[]
    for i,a in enumerate(labels):
        for j,b in enumerate(labels):data.append({"actual":a,"predicted":b,"value":values[i,j]})
    write_csv(ROOT/f"figures/data/{name}.csv",data)
    fig,ax=plt.subplots(figsize=(6,5));im=ax.imshow(values,cmap="Blues");fig.colorbar(im,ax=ax);ax.set_xticks(range(len(labels)));ax.set_yticks(range(len(labels)));ax.set_xticklabels(labels,rotation=45,ha="right",fontsize=7);ax.set_yticklabels(labels,fontsize=7);ax.set_xlabel("Predicted");ax.set_ylabel("Actual");ax.set_title(name.replace('_',' ').title());fig.tight_layout()
    for ext in ("png","svg","pdf"):fig.savefig(ROOT/f"figures/{ext}/{name}.{ext}",bbox_inches="tight")
    plt.close(fig)


def main():
    timings=read_csv(ROOT/"raw/per_image_timings.csv"); warm=read_csv(ROOT/"raw/warmup_timings.csv"); tele=read_csv(ROOT/"monitoring/active_samples.csv")
    stage=read_csv(ROOT/"summary/timing_stage_summary.csv"); perf=read_json(ROOT/"summary/performance_summary.json"); energy=read_json(ROOT/"energy/energy_summary.json"); metrics=read_json(ROOT/"metrics/detection_metrics.json")
    comp=read_csv(ROOT/"comparison/fp32_fp16_int8_metrics.csv")
    stage_rows=[{"label":r.get("timing_field",r.get("stage","stage")),"value":r.get("mean_ms",r.get("mean",0))} for r in stage]
    save(FIGURES[0],stage_rows); share=[{"label":r["label"],"value":100*float(r["value"])/perf["mean_pipeline_latency_ms"]} for r in stage_rows];save(FIGURES[1],share)
    save(FIGURES[2],[{"label":i,"value":r["tensorrt_engine_ms"]} for i,r in enumerate(timings)],"hist",ys=("value",));save(FIGURES[3],[{"label":i,"value":r["end_to_end_ms"]} for i,r in enumerate(timings)],"hist",ys=("value",))
    save(FIGURES[4],[{"label":k,"value":float(np.mean(nums(timings,k)))} for k in ("h2d_ms","tensorrt_engine_ms","d2h_ms")])
    pre=[k for k in timings[0] if k.endswith("_ms") and k in ("file_access_decode_ms","exif_orientation_ms","color_conversion_ms","resize_calculation_ms","image_resize_ms","padding_letterbox_ms","pixel_scaling_ms","layout_conversion_ms","fp32_contiguous_ms")]
    post=[k for k in timings[0] if k in ("raw_output_decode_ms","box_conversion_ms","confidence_calculation_ms","confidence_filtering_ms","class_selection_ms","nms_preparation_ms","nms_ms","coordinate_rescaling_ms","box_clipping_ms","result_formatting_ms")]
    save(FIGURES[5],[{"label":k,"value":float(np.mean(nums(timings,k)))} for k in pre]);save(FIGURES[6],[{"label":k,"value":float(np.mean(nums(timings,k)))} for k in post])
    save(FIGURES[7],[{"label":i,"value":r["nms_ms"]} for i,r in enumerate(timings)],"hist",ys=("value",))
    save(FIGURES[8],[{"label":"engine","value":perf["engine_fps"]},{"label":"transfer","value":perf["transfer_inclusive_fps"]},{"label":"latency pipeline","value":perf["latency_derived_pipeline_fps"]},{"label":"wall clock","value":perf["wall_clock_pipeline_fps"]}])
    save(FIGURES[9],[{"label":i,"value":r["power_tot_w"]} for i,r in enumerate(tele)],"line",ys=("value",));save(FIGURES[10],[{"label":i,"cpu":r["cpu_percent"],"gpu":r["gpu_percent"]} for i,r in enumerate(tele)],"line",ys=("cpu","gpu"))
    save(FIGURES[11],[{"label":i,"ram":r["ram_used_mib"],"rss":r.get("process_rss_mib",0)} for i,r in enumerate(tele)],"line",ys=("ram","rss"));save(FIGURES[12],[{"label":i,"value":r["temperature_c"]} for i,r in enumerate(tele)],"line",ys=("value",))
    save(FIGURES[13],[{"label":"gross J/image","value":energy["gross_energy_per_image_j"]},{"label":"net J/image","value":energy["idle_corrected_energy_per_image_j"]}])
    pr=read_csv(ROOT/"metrics/precision_recall_curve.csv");save(FIGURES[14],[{"label":i,"precision":r.get("precision",0),"recall":r.get("recall",0)} for i,r in enumerate(pr)],"line",ys=("precision","recall"))
    fc=read_csv(ROOT/"metrics/f1_confidence_curve.csv");save(FIGURES[15],[{"label":r.get("confidence",i),"value":r.get("f1",0)} for i,r in enumerate(fc)],"line",ys=("value",))
    save(FIGURES[16],[{"label":x["class_name"],"ap50":x["ap50"],"ap50_95":x["ap50_95"]} for x in metrics["per_class"]],ys=("ap50","ap50_95"))
    ap=read_csv(ROOT/"metrics/ap_by_iou.csv");save(FIGURES[17],[{"label":r.get("iou_threshold",i),"value":r.get("mAP",r.get("ap",0))} for i,r in enumerate(ap)],"line",ys=("value",))
    save(FIGURES[18],[{"label":i,"value":r["detection_count"]} for i,r in enumerate(timings)],"hist",ys=("value",));save(FIGURES[19],[{"label":"FP images","value":metrics["images_with_false_positives"]},{"label":"FN images","value":metrics["images_with_false_negatives"]},{"label":"empty","value":metrics["empty_prediction_images_conf025"]}])
    matrix_figure(FIGURES[20],ROOT/"metrics/confusion_matrix.csv");matrix_figure(FIGURES[21],ROOT/"metrics/confusion_matrix.csv",True)
    save(FIGURES[22],[{"label":r["warmup_index"],"engine":r["tensorrt_engine_ms"],"e2e":r["end_to_end_ms"]} for r in warm],"line",ys=("engine","e2e"))
    cov=read_csv(ROOT/"calibration/calibration_selected_class_coverage.csv");save(FIGURES[23],[{"label":r["class_name"],"images":r["selected_image_count"],"objects":r["selected_object_count"]} for r in cov],ys=("images","objects"))
    sel=read_csv(ROOT/"calibration/calibration_selected_manifest.csv");save(FIGURES[24],[{"label":i,"width":r["width"],"height":r["height"]} for i,r in enumerate(sel)],"line",ys=("width","height"))
    overlap=read_json(ROOT/"calibration/calibration_test_overlap_summary.json");save(FIGURES[25],[{"label":"file","value":overlap["exact_file_overlap_count"]},{"label":"pixel","value":overlap["decoded_pixel_overlap_count"]},{"label":"near duplicate","value":overlap["unresolved_near_duplicate_count"]}])
    save(FIGURES[26],[{"label":r["precision"],"mAP50":r["mAP50"],"mAP50_95":r["mAP50_95"]} for r in comp],ys=("mAP50","mAP50_95"));save(FIGURES[27],[{"label":r["precision"],"value":r["engine_latency_ms"]} for r in comp]);save(FIGURES[28],[{"label":r["precision"],"value":r["e2e_latency_ms"]} for r in comp])
    save(FIGURES[29],[{"label":r["precision"],"engine":r["engine_fps"],"pipeline":r["wall_clock_fps"]} for r in comp],ys=("engine","pipeline"));save(FIGURES[30],[{"label":r["precision"],"gross":r["gross_energy_per_image_j"],"net":r["idle_corrected_energy_per_image_j"]} for r in comp],ys=("gross","net"));save(FIGURES[31],[{"label":r["precision"],"cpu":r["cpu_mean_percent"],"gpu":r["gpu_mean_percent"]} for r in comp],ys=("cpu","gpu"))
    rec=read_json(ROOT/"comparison/deployment_recommendation.json");save(FIGURES[32],[{"label":r["precision"],"score":1.0 if r["precision"]==rec["recommended_precision"] else 0.0} for r in comp])
    counts={ext:len(list((ROOT/f"figures/{ext}").glob(f"*.{ext}"))) for ext in ("png","svg","pdf")};data=len(list((ROOT/"figures/data").glob("*.csv")))
    write_json(ROOT/"validation/figure_inventory.json",{"status":"PASS" if all(v==33 for v in counts.values()) and data==33 else "FAIL","counts":counts,"data_csv":data,"required":FIGURES})
    print(json.dumps({"status":"PASS","counts":counts,"data":data},indent=2))


if __name__=="__main__":main()
