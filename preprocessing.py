#!/usr/bin/env python3
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from common import HEIGHT, WIDTH


TIMING_FIELDS = [
    "file_access_decode_ms",
    "exif_orientation_ms",
    "color_conversion_ms",
    "resize_calculation_ms",
    "image_resize_ms",
    "padding_letterbox_ms",
    "pixel_scaling_ms",
    "layout_conversion_ms",
    "fp32_contiguous_ms",
    "preprocessing_total_ms",
]


def _ms(start_ns: int, end_ns: int) -> float:
    return (end_ns - start_ns) / 1_000_000.0


def preprocess_image(path: str | Path, height: int = HEIGHT, width: int = WIDTH):
    """Decode and letterbox one image while independently timing each stage."""
    path = Path(path)
    total_start = time.perf_counter_ns()

    t0 = time.perf_counter_ns()
    image = Image.open(path)
    image.load()
    t1 = time.perf_counter_ns()

    t2 = time.perf_counter_ns()
    image = ImageOps.exif_transpose(image)
    t3 = time.perf_counter_ns()

    t4 = time.perf_counter_ns()
    image = image.convert("RGB")
    t5 = time.perf_counter_ns()
    original_width, original_height = image.size

    t6 = time.perf_counter_ns()
    ratio = min(width / original_width, height / original_height)
    resized_width = int(round(original_width * ratio))
    resized_height = int(round(original_height * ratio))
    pad_w = (width - resized_width) / 2.0
    pad_h = (height - resized_height) / 2.0
    paste_x = int(round(pad_w - 0.1))
    paste_y = int(round(pad_h - 0.1))
    t7 = time.perf_counter_ns()

    t8 = time.perf_counter_ns()
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    resized = image.resize((resized_width, resized_height), resampling)
    t9 = time.perf_counter_ns()

    t10 = time.perf_counter_ns()
    canvas = Image.new("RGB", (width, height), (114, 114, 114))
    canvas.paste(resized, (paste_x, paste_y))
    t11 = time.perf_counter_ns()

    t12 = time.perf_counter_ns()
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    t13 = time.perf_counter_ns()

    t14 = time.perf_counter_ns()
    array = np.transpose(array, (2, 0, 1))[None, ...]
    t15 = time.perf_counter_ns()

    t16 = time.perf_counter_ns()
    array = np.ascontiguousarray(array, dtype=np.float32)
    t17 = time.perf_counter_ns()
    total_end = time.perf_counter_ns()

    timings = {
        "file_access_decode_ms": _ms(t0, t1),
        "exif_orientation_ms": _ms(t2, t3),
        "color_conversion_ms": _ms(t4, t5),
        "resize_calculation_ms": _ms(t6, t7),
        "image_resize_ms": _ms(t8, t9),
        "padding_letterbox_ms": _ms(t10, t11),
        "pixel_scaling_ms": _ms(t12, t13),
        "layout_conversion_ms": _ms(t14, t15),
        "fp32_contiguous_ms": _ms(t16, t17),
        "preprocessing_total_ms": _ms(total_start, total_end),
    }
    meta = {
        "image_path": str(path.resolve()),
        "original_width": original_width,
        "original_height": original_height,
        "input_width": width,
        "input_height": height,
        "ratio": ratio,
        "resized_width": resized_width,
        "resized_height": resized_height,
        "pad_w": pad_w,
        "pad_h": pad_h,
        "paste_x": paste_x,
        "paste_y": paste_y,
        "padding_color": [114, 114, 114],
        "interpolation": "PIL bilinear",
        "scale_up": True,
        "center": True,
        "auto_padding": False,
        "layout": "contiguous NCHW",
        "dtype": "float32",
        "normalization": "divide by 255.0 only",
    }
    return array, meta, timings
