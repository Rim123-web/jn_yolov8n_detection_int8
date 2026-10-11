#!/usr/bin/env python3
from __future__ import annotations

import csv
import ctypes
import ctypes.util
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

TRT_SITE = "/home/jetson/.local/lib/python3.8/site-packages"
if TRT_SITE not in sys.path:
    sys.path.insert(0, TRT_SITE)
import tensorrt as trt

from preprocessing import preprocess_image


def preprocess_calibration_image(path: Path):
    array, metadata, _ = preprocess_image(path)
    return array, metadata


def _load_cudart():
    candidates = [ctypes.util.find_library("cudart"), "libcudart.so", "/usr/local/cuda/lib64/libcudart.so",
                  "/usr/local/cuda-10.2/targets/aarch64-linux/lib/libcudart.so"]
    for candidate in candidates:
        if candidate:
            try:
                return ctypes.CDLL(candidate), candidate
            except OSError:
                pass
    raise RuntimeError("Could not load CUDA Runtime for INT8 calibrator")


class DeterministicImageEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, image_paths, cache_path, execution_order_path, batch_manifest_path, log_path):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.image_paths = [Path(p) for p in image_paths]
        self.cache_path = Path(cache_path)
        self.execution_order_path = Path(execution_order_path)
        self.batch_manifest_path = Path(batch_manifest_path)
        self.log_path = Path(log_path)
        self.index = 0
        self.batch_size = 1
        self.shape = (1, 3, 640, 960)
        self.host = np.empty(self.shape, dtype=np.float32)
        self.cudart, self.cudart_name = _load_cudart()
        self.cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.cudart.cudaMalloc.restype = ctypes.c_int
        self.cudart.cudaFree.argtypes = [ctypes.c_void_p]
        self.cudart.cudaFree.restype = ctypes.c_int
        self.cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self.cudart.cudaMemcpy.restype = ctypes.c_int
        self.device = ctypes.c_void_p()
        code = self.cudart.cudaMalloc(ctypes.byref(self.device), self.host.nbytes)
        if code != 0:
            raise RuntimeError(f"calibrator cudaMalloc failed with CUDA error {code}")
        self.rows = []
        self.execution_order_path.write_text("\n".join(str(p.resolve()) for p in self.image_paths) + "\n", encoding="utf-8")
        self._log(f"initialized IInt8EntropyCalibrator2 images={len(self.image_paths)} batch=1 shape={self.shape} cache_preexists={self.cache_path.exists()}")

    def _log(self, message):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().astimezone().isoformat()}] {message}\n")

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.index >= len(self.image_paths):
            self._write_manifest()
            self._log("all calibration batches consumed")
            return None
        image_path = self.image_paths[self.index]
        array, metadata = preprocess_calibration_image(image_path)
        if tuple(array.shape) != self.shape or array.dtype != np.float32 or not array.flags.c_contiguous:
            raise RuntimeError(f"Invalid calibration tensor: {array.shape} {array.dtype} contiguous={array.flags.c_contiguous}")
        np.copyto(self.host, array)
        code = self.cudart.cudaMemcpy(self.device, ctypes.c_void_p(self.host.ctypes.data), self.host.nbytes, 1)
        if code != 0:
            raise RuntimeError(f"calibrator cudaMemcpy H2D failed with CUDA error {code}")
        self.rows.append({"batch_index":self.index,"image_path":str(image_path.resolve()),
                          "input_sha256":hashlib.sha256(self.host.tobytes()).hexdigest(),
                          "input_shape":"1x3x640x960","input_dtype":"float32",
                          "original_width":metadata["original_width"],"original_height":metadata["original_height"]})
        self.index += 1
        if self.index == 1 or self.index % 10 == 0 or self.index == len(self.image_paths):
            self._log(f"provided batch {self.index}/{len(self.image_paths)}: {image_path}")
        return [int(self.device.value)]

    def read_calibration_cache(self):
        if self.cache_path.exists():
            raise RuntimeError(f"Fresh-build rule violated: calibration cache already exists: {self.cache_path}")
        self._log("read_calibration_cache: no cache returned (fresh calibration required)")
        return b""

    def write_calibration_cache(self, cache):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_bytes(cache)
        self._write_manifest()
        self._log(f"wrote calibration cache bytes={len(cache)} sha256={hashlib.sha256(cache).hexdigest()}")

    def _write_manifest(self):
        self.batch_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        fields=["batch_index","image_path","input_sha256","input_shape","input_dtype","original_width","original_height"]
        with self.batch_manifest_path.open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(self.rows)

    def close(self):
        if getattr(self, "device", None) and self.device.value:
            self.cudart.cudaFree(self.device); self.device = ctypes.c_void_p()

    def __del__(self):
        try: self.close()
        except Exception: pass

