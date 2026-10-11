#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(os.environ.get("BENCHMARK_DIR", Path(__file__).resolve().parents[1])).resolve()
SOURCE_PT = Path("/home/jetson/Downloads/jetson_folder/oct_albument2_v6.pt")
DATASET_ROOT = Path("/home/jetson/Downloads/jetson_folder/jetson_test_imgs")
IMAGE_DIR = DATASET_ROOT / "images" / "test"
LABEL_DIR = DATASET_ROOT / "labels" / "test"
DATA_YAML = DATASET_ROOT / "data.yaml"
HEIGHT = 640
WIDTH = 960
BATCH = 1
INPUT_SHAPE = (1, 3, HEIGHT, WIDTH)
CLASS_NAMES = [
    "leccino",
    "varietafrancese",
    "cimadimelfimisto",
    "ulivigrandi",
    "montalbanosecolari",
    "irriguo",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CONFIDENCE_THRESHOLD = 0.25
METRIC_CONFIDENCE_FLOOR = 0.001
NMS_IOU_THRESHOLD = 0.70
MATCHING_IOU_THRESHOLD = 0.50
SEED = 42


def utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, value: Any) -> None:
    ensure_parent(path)
    def json_default(obj):
        if isinstance(obj, Path):
            return str(obj)
        if hasattr(obj, "item"):
            return obj.item()
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=json_default) + "\n", encoding="utf-8")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return default


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    ensure_parent(path)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_metadata(path: Path) -> dict:
    st = path.stat()
    return {
        "path": str(path.resolve()),
        "filename": path.name,
        "size_bytes": st.st_size,
        "size_mib": st.st_size / (1024 ** 2),
        "modified_time": datetime.fromtimestamp(st.st_mtime).astimezone().isoformat(),
        "sha256": sha256_file(path),
    }


def collect_images() -> list[Path]:
    return sorted(
        (p for p in IMAGE_DIR.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: str(p).lower(),
    )


def label_path_for_image(image_path: Path) -> Path:
    rel = image_path.relative_to(IMAGE_DIR)
    return (LABEL_DIR / rel).with_suffix(".txt")


def percentile(values: Iterable[float], q: float) -> float | None:
    values = sorted(float(x) for x in values)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * q / 100.0
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return values[low]
    weight = position - low
    return values[low] * (1.0 - weight) + values[high] * weight


def finite_nonnegative(value: Any) -> bool:
    try:
        number = float(value)
        return math.isfinite(number) and number >= 0.0
    except Exception:
        return False


def append_timeline(stage: str, status: str, details: str = "") -> None:
    path = ROOT / "provenance" / "workflow_timeline.csv"
    ensure_parent(path)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["timestamp", "stage", "status", "details"])
        if not exists:
            writer.writeheader()
        writer.writerow({"timestamp": utc_now(), "stage": stage, "status": status, "details": details})


def record_command(command: str) -> None:
    path = ROOT / "provenance" / "execution_commands.txt"
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"[{utc_now()}] {command}\n")


def load_yolo_labels(image_path: Path) -> list[dict]:
    from PIL import Image

    label_path = label_path_for_image(image_path)
    width, height = Image.open(image_path).size
    labels = []
    if not label_path.exists():
        return labels
    for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Malformed label {label_path}:{line_no}: {line}")
        cls = int(float(parts[0]))
        cx, cy, bw, bh = map(float, parts[1:])
        labels.append({
            "cls": cls,
            "box": [
                (cx - bw / 2) * width,
                (cy - bh / 2) * height,
                (cx + bw / 2) * width,
                (cy + bh / 2) * height,
            ],
        })
    return labels
