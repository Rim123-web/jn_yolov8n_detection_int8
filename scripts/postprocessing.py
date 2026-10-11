#!/usr/bin/env python3
from __future__ import annotations

import time
from typing import Any

import numpy as np

from common import CLASS_NAMES, CONFIDENCE_THRESHOLD, NMS_IOU_THRESHOLD
from nms import class_aware_nms, prepare_class_aware_nms


TIMING_FIELDS = [
    "raw_output_decode_ms",
    "box_conversion_ms",
    "confidence_calculation_ms",
    "confidence_filtering_ms",
    "class_selection_ms",
    "nms_preparation_ms",
    "nms_ms",
    "coordinate_rescaling_ms",
    "box_clipping_ms",
    "result_formatting_ms",
    "postprocessing_total_ms",
]


def _elapsed_ms(a: int, b: int) -> float:
    return (b - a) / 1_000_000.0


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    result = np.empty_like(boxes)
    result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return result


def postprocess_yolov8(
    output: np.ndarray,
    meta: dict[str, Any],
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    nms_iou_threshold: float = NMS_IOU_THRESHOLD,
    num_classes: int = len(CLASS_NAMES),
):
    total_start = time.perf_counter_ns()

    t0 = time.perf_counter_ns()
    prediction = np.asarray(output)
    while prediction.ndim > 2:
        prediction = prediction[0]
    if prediction.shape[0] == 4 + num_classes and prediction.shape[0] < prediction.shape[1]:
        prediction = prediction.T
    if prediction.ndim != 2 or prediction.shape[1] != 4 + num_classes:
        raise RuntimeError(f"Expected raw YOLOv8 output [N,{4 + num_classes}], got {prediction.shape}")
    raw_boxes = prediction[:, :4]
    class_scores = prediction[:, 4:]
    t1 = time.perf_counter_ns()

    t2 = time.perf_counter_ns()
    boxes = _xywh_to_xyxy(raw_boxes)
    t3 = time.perf_counter_ns()

    t4 = time.perf_counter_ns()
    scores = np.max(class_scores, axis=1)
    t5 = time.perf_counter_ns()

    t6 = time.perf_counter_ns()
    mask = scores >= float(confidence_threshold)
    boxes = boxes[mask]
    scores = scores[mask]
    class_scores = class_scores[mask]
    t7 = time.perf_counter_ns()

    t8 = time.perf_counter_ns()
    classes = np.argmax(class_scores, axis=1).astype(np.int64) if class_scores.size else np.empty((0,), dtype=np.int64)
    t9 = time.perf_counter_ns()

    t10 = time.perf_counter_ns()
    order = prepare_class_aware_nms(scores)
    t11 = time.perf_counter_ns()

    t12 = time.perf_counter_ns()
    keep = class_aware_nms(boxes, scores, classes, float(nms_iou_threshold), order) if boxes.size else np.empty((0,), dtype=np.int64)
    boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
    t13 = time.perf_counter_ns()

    t14 = time.perf_counter_ns()
    if boxes.size:
        boxes[:, [0, 2]] -= float(meta["pad_w"])
        boxes[:, [1, 3]] -= float(meta["pad_h"])
        boxes[:, :4] /= float(meta["ratio"])
    t15 = time.perf_counter_ns()

    t16 = time.perf_counter_ns()
    if boxes.size:
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0.0, float(meta["original_width"]))
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0.0, float(meta["original_height"]))
    t17 = time.perf_counter_ns()

    t18 = time.perf_counter_ns()
    detections = [
        {
            "box": [float(v) for v in boxes[i]],
            "confidence": float(scores[i]),
            "class_id": int(classes[i]),
            "class_name": CLASS_NAMES[int(classes[i])],
        }
        for i in range(len(boxes))
    ]
    t19 = time.perf_counter_ns()
    total_end = time.perf_counter_ns()

    timings = {
        "raw_output_decode_ms": _elapsed_ms(t0, t1),
        "box_conversion_ms": _elapsed_ms(t2, t3),
        "confidence_calculation_ms": _elapsed_ms(t4, t5),
        "confidence_filtering_ms": _elapsed_ms(t6, t7),
        "class_selection_ms": _elapsed_ms(t8, t9),
        "nms_preparation_ms": _elapsed_ms(t10, t11),
        "nms_ms": _elapsed_ms(t12, t13),
        "coordinate_rescaling_ms": _elapsed_ms(t14, t15),
        "box_clipping_ms": _elapsed_ms(t16, t17),
        "result_formatting_ms": _elapsed_ms(t18, t19),
        "postprocessing_total_ms": _elapsed_ms(total_start, total_end),
    }
    return detections, timings
