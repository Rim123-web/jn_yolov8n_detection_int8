#!/usr/bin/env python3
from __future__ import annotations

import numpy as np


def box_iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.empty((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return intersection / (area_a + area_b - intersection + 1e-7)


def prepare_class_aware_nms(scores: np.ndarray) -> np.ndarray:
    return np.argsort(scores)[::-1].astype(np.int64, copy=False)


def class_aware_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    iou_threshold: float,
    order: np.ndarray,
) -> np.ndarray:
    """Deterministic CPU NumPy class-aware NMS using a precomputed score order."""
    keep: list[int] = []
    order = order.copy()
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        remaining = order[1:]
        same_class = classes[remaining] == classes[current]
        ious = np.zeros(remaining.shape[0], dtype=np.float32)
        if np.any(same_class):
            ious[same_class] = box_iou_one_to_many(boxes[current], boxes[remaining[same_class]])
        order = remaining[ious <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)
