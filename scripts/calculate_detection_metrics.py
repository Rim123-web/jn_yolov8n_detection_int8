#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict

import numpy as np

from common import (
    CLASS_NAMES, CONFIDENCE_THRESHOLD, MATCHING_IOU_THRESHOLD,
    METRIC_CONFIDENCE_FLOOR, NMS_IOU_THRESHOLD, ROOT, append_timeline,
    collect_images, label_path_for_image, load_yolo_labels, write_csv, write_json,
)
from postprocessing import postprocess_yolov8


def box_iou(boxes_a, boxes_b):
    boxes_a = np.asarray(boxes_a, dtype=np.float32).reshape(-1, 4)
    boxes_b = np.asarray(boxes_b, dtype=np.float32).reshape(-1, 4)
    if not len(boxes_a) or not len(boxes_b):
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    area_a = np.maximum(0, boxes_a[:, 2] - boxes_a[:, 0]) * np.maximum(0, boxes_a[:, 3] - boxes_a[:, 1])
    area_b = np.maximum(0, boxes_b[:, 2] - boxes_b[:, 0]) * np.maximum(0, boxes_b[:, 3] - boxes_b[:, 1])
    tl = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    br = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = np.maximum(0, br - tl)
    intersection = wh[..., 0] * wh[..., 1]
    return intersection / (area_a[:, None] + area_b[None, :] - intersection + 1e-7)


def compute_ap(recall, precision):
    recall = np.asarray(recall, dtype=np.float64)
    precision = np.asarray(precision, dtype=np.float64)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    x = np.linspace(0.0, 1.0, 101)
    return float(np.trapz(np.interp(x, mrec, mpre), x))


def match_class_predictions(predictions, ground_truth, class_id, iou_threshold):
    relevant = sorted((p for p in predictions if p["class_id"] == class_id), key=lambda x: x["confidence"], reverse=True)
    gt_by_image = {}
    count_gt = 0
    for image_path, labels in ground_truth.items():
        boxes = [x["box"] for x in labels if x["cls"] == class_id]
        gt_by_image[image_path] = {"boxes": np.asarray(boxes, dtype=np.float32).reshape(-1, 4), "used": np.zeros(len(boxes), dtype=bool)}
        count_gt += len(boxes)
    tp = np.zeros(len(relevant), dtype=np.float64)
    fp = np.zeros(len(relevant), dtype=np.float64)
    for index, prediction in enumerate(relevant):
        data = gt_by_image[prediction["image_path"]]
        if not len(data["boxes"]):
            fp[index] = 1
            continue
        ious = box_iou([prediction["box"]], data["boxes"])[0]
        order = np.argsort(ious)[::-1]
        matched = False
        for gt_index in order:
            if ious[gt_index] >= iou_threshold and not data["used"][gt_index]:
                data["used"][gt_index] = True
                tp[index] = 1
                matched = True
                break
        if not matched:
            fp[index] = 1
    cumulative_tp = np.cumsum(tp)
    cumulative_fp = np.cumsum(fp)
    recall = cumulative_tp / max(count_gt, 1)
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12)
    return relevant, tp, fp, recall, precision, count_gt


def point_metrics(predictions, ground_truth, confidence_threshold, iou_threshold):
    tp = fp = 0
    total_gt = sum(len(x) for x in ground_truth.values())
    per_class = {}
    image_errors = []
    missed_rows = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        filtered = [x for x in predictions if x["confidence"] >= confidence_threshold]
        relevant, flags_tp, flags_fp, _, _, gt_count = match_class_predictions(filtered, ground_truth, class_id, iou_threshold)
        class_tp = int(flags_tp.sum())
        class_fp = int(flags_fp.sum())
        class_fn = gt_count - class_tp
        tp += class_tp
        fp += class_fp
        precision = class_tp / max(class_tp + class_fp, 1)
        recall = class_tp / max(gt_count, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_class[class_id] = {"precision": precision, "recall": recall, "f1": f1, "tp": class_tp, "fp": class_fp, "fn": class_fn, "gt": gt_count}
    fn = total_gt - tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(total_gt, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "ground_truth": total_gt}, per_class


def confusion_and_errors(predictions, ground_truth, threshold=0.25, iou_threshold=0.5):
    nc = len(CLASS_NAMES)
    matrix = np.zeros((nc + 1, nc + 1), dtype=np.int64)
    errors = []
    details = []
    for image_path, gt_labels in ground_truth.items():
        preds = [p for p in predictions if p["image_path"] == image_path and p["confidence"] >= threshold]
        preds = sorted(preds, key=lambda p: p["confidence"], reverse=True)
        gt_boxes = np.asarray([g["box"] for g in gt_labels], dtype=np.float32).reshape(-1, 4)
        gt_used = np.zeros(len(gt_labels), dtype=bool)
        image_fp = 0
        for pred in preds:
            if len(gt_boxes):
                ious = box_iou([pred["box"]], gt_boxes)[0]
                order = np.argsort(ious)[::-1]
            else:
                ious, order = np.empty((0,)), np.empty((0,), dtype=int)
            match = next((int(i) for i in order if ious[i] >= iou_threshold and not gt_used[i]), None)
            if match is None:
                matrix[nc, pred["class_id"]] += 1
                image_fp += 1
                details.append({"image_path": image_path, "error_type": "false_positive", "true_class": "background", "predicted_class": pred["class_name"], "confidence": pred["confidence"], "iou": 0.0})
            else:
                gt_used[match] = True
                true_cls = gt_labels[match]["cls"]
                matrix[true_cls, pred["class_id"]] += 1
                if true_cls != pred["class_id"]:
                    details.append({"image_path": image_path, "error_type": "misclassification", "true_class": CLASS_NAMES[true_cls], "predicted_class": pred["class_name"], "confidence": pred["confidence"], "iou": float(ious[match])})
        image_fn = 0
        for index, used in enumerate(gt_used):
            if not used:
                true_cls = gt_labels[index]["cls"]
                matrix[true_cls, nc] += 1
                image_fn += 1
                details.append({"image_path": image_path, "error_type": "false_negative", "true_class": CLASS_NAMES[true_cls], "predicted_class": "background", "confidence": 0.0, "iou": 0.0})
        errors.append({"image_path": image_path, "false_positives": image_fp, "false_negatives": image_fn, "has_false_positive": image_fp > 0, "has_false_negative": image_fn > 0})
    return matrix, errors, details


def main():
    append_timeline("offline_detection_metrics", "START", "no inference; official raw outputs only")
    images = collect_images()
    raw = np.load(ROOT / "raw/official_raw_outputs.npz")["outputs"]
    manifest = {row["image_path"]: row for row in csv.DictReader((ROOT / "dataset/test_image_manifest.csv").open(encoding="utf-8"))}
    predictions = []
    ground_truth = {}
    raw_prediction_rows = []
    with (ROOT / "raw/metric_predictions_conf001.jsonl").open("w", encoding="utf-8") as jsonl:
        for index, image_path in enumerate(images, 1):
            row = manifest[str(image_path.resolve())]
            original_width, original_height = int(row["width"]), int(row["height"])
            ratio = min(960 / original_width, 640 / original_height)
            meta = {"original_width": original_width, "original_height": original_height, "ratio": ratio, "pad_w": (960 - round(original_width * ratio)) / 2.0, "pad_h": (640 - round(original_height * ratio)) / 2.0}
            detections, _ = postprocess_yolov8(raw[index - 1][None], meta, confidence_threshold=METRIC_CONFIDENCE_FLOOR)
            image_key = str(image_path.resolve())
            ground_truth[image_key] = load_yolo_labels(image_path)
            normalized = []
            for detection_index, detection in enumerate(detections):
                pred = {"image_path": image_key, **detection}
                predictions.append(pred)
                x1, y1, x2, y2 = detection["box"]
                raw_prediction_rows.append({"image_index": index, "image_path": image_key, "detection_index": detection_index, "class_id": detection["class_id"], "class_name": detection["class_name"], "confidence": detection["confidence"], "x1": x1, "y1": y1, "x2": x2, "y2": y2})
                normalized.append(detection)
            jsonl.write(json.dumps({"image_index": index, "image_path": image_key, "confidence_floor": METRIC_CONFIDENCE_FLOOR, "detections": normalized}) + "\n")
    write_csv(ROOT / "raw/metric_predictions_conf001.csv", raw_prediction_rows)

    iou_thresholds = np.arange(0.50, 0.951, 0.05)
    per_class_rows = []
    ap_iou_rows = []
    class_ap = {}
    for class_id, class_name in enumerate(CLASS_NAMES):
        aps = []
        for iou_threshold in iou_thresholds:
            _, _, _, recall, precision, gt_count = match_class_predictions(predictions, ground_truth, class_id, float(iou_threshold))
            ap = compute_ap(recall, precision) if gt_count else 0.0
            aps.append(ap)
            ap_iou_rows.append({"class_id": class_id, "class_name": class_name, "iou_threshold": float(iou_threshold), "ap": ap})
        class_ap[class_id] = aps
    point, point_by_class = point_metrics(predictions, ground_truth, CONFIDENCE_THRESHOLD, MATCHING_IOU_THRESHOLD)
    for class_id, class_name in enumerate(CLASS_NAMES):
        row = point_by_class[class_id]
        per_class_rows.append({
            "class_id": class_id, "class_name": class_name, "ground_truth_objects": row["gt"],
            "ap50": class_ap[class_id][0], "ap50_95": float(np.mean(class_ap[class_id])),
            "precision_conf025": row["precision"], "recall_conf025": row["recall"],
            "f1_conf025": row["f1"], "tp": row["tp"], "fp": row["fp"], "fn": row["fn"],
        })
    map50 = float(np.mean([class_ap[i][0] for i in range(len(CLASS_NAMES))]))
    map50_95 = float(np.mean([np.mean(class_ap[i]) for i in range(len(CLASS_NAMES))]))
    write_csv(ROOT / "metrics/per_class_metrics.csv", per_class_rows)
    write_csv(ROOT / "metrics/ap_by_iou.csv", ap_iou_rows)

    # Global confidence curve uses exact point matching at each saved prediction confidence.
    thresholds = np.unique(np.concatenate(([METRIC_CONFIDENCE_FLOOR, CONFIDENCE_THRESHOLD], np.linspace(0.0, 1.0, 201))))
    curve_rows = []
    for threshold in thresholds:
        result, _ = point_metrics(predictions, ground_truth, float(threshold), MATCHING_IOU_THRESHOLD)
        curve_rows.append({"confidence_threshold": float(threshold), "precision": result["precision"], "recall": result["recall"], "f1": result["f1"], "tp": result["tp"], "fp": result["fp"], "fn": result["fn"]})
    best = max(curve_rows, key=lambda row: row["f1"])
    write_csv(ROOT / "metrics/precision_recall_curve.csv", curve_rows)
    write_csv(ROOT / "metrics/f1_confidence_curve.csv", curve_rows)

    matrix, error_rows, details = confusion_and_errors(predictions, ground_truth, CONFIDENCE_THRESHOLD, MATCHING_IOU_THRESHOLD)
    labels = CLASS_NAMES + ["background"]
    matrix_rows = [{"true_class": labels[i], **{labels[j]: int(matrix[i, j]) for j in range(len(labels))}} for i in range(len(labels))]
    normalized = matrix.astype(np.float64) / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    normalized_rows = [{"true_class": labels[i], **{labels[j]: float(normalized[i, j]) for j in range(len(labels))}} for i in range(len(labels))]
    write_csv(ROOT / "metrics/confusion_matrix.csv", matrix_rows)
    write_csv(ROOT / "metrics/confusion_matrix_normalized.csv", normalized_rows)
    write_csv(ROOT / "metrics/error_analysis.csv", error_rows)
    write_csv(ROOT / "metrics/misclassified_or_missed_objects.csv", details)

    image_prediction_counts = Counter(p["image_path"] for p in predictions if p["confidence"] >= CONFIDENCE_THRESHOLD)
    metrics = {
        "status": "PASS", "backend": "TensorRT strict FP32; offline metrics from official measured raw outputs",
        "test_images": len(images), "classes": CLASS_NAMES, "ground_truth_objects": point["ground_truth"],
        "metric_prediction_count_conf001": len(predictions),
        "point_prediction_count_conf025": sum(1 for p in predictions if p["confidence"] >= CONFIDENCE_THRESHOLD),
        "mAP50": map50, "mAP50_95": map50_95,
        "precision_conf025": point["precision"], "recall_conf025": point["recall"], "f1_conf025": point["f1"],
        "tp_conf025": point["tp"], "fp_conf025": point["fp"], "fn_conf025": point["fn"],
        "best_f1": best["f1"], "best_f1_confidence_threshold": best["confidence_threshold"],
        "average_detections_per_image_conf025": sum(image_prediction_counts.values()) / len(images),
        "empty_prediction_images_conf025": sum(image_prediction_counts[str(path.resolve())] == 0 for path in images),
        "images_with_false_positives": sum(bool(row["has_false_positive"]) for row in error_rows),
        "images_with_false_negatives": sum(bool(row["has_false_negative"]) for row in error_rows),
        "definitions": {
            "metric_confidence_floor": METRIC_CONFIDENCE_FLOOR, "reported_point_confidence": CONFIDENCE_THRESHOLD,
            "matching_iou": MATCHING_IOU_THRESHOLD, "nms_iou": NMS_IOU_THRESHOLD,
            "ap_interpolation": "101-point interpolated precision envelope", "map_iou_range": "0.50:0.05:0.95",
            "class_averaging": "unweighted macro mean across six classes", "ignored_or_difficult": "none in YOLO labels",
        },
        "per_class": per_class_rows,
    }
    write_json(ROOT / "metrics/detection_metrics.json", metrics)
    write_csv(ROOT / "metrics/detection_metrics.csv", [{k: v for k, v in metrics.items() if not isinstance(v, (dict, list))}])
    write_json(ROOT / "summary/article_results.json", metrics)
    write_csv(ROOT / "summary/article_results.csv", [{k: v for k, v in metrics.items() if not isinstance(v, (dict, list))}])
    append_timeline("offline_detection_metrics", "PASS", f"mAP50={map50:.6f} mAP50_95={map50_95:.6f}")
    print(json.dumps({k: metrics[k] for k in ("mAP50", "mAP50_95", "precision_conf025", "recall_conf025", "f1_conf025", "best_f1", "best_f1_confidence_threshold")}, indent=2))


if __name__ == "__main__":
    main()
