#!/usr/bin/env python3
from __future__ import annotations

import json
import threading
import time
from datetime import datetime

import psutil
from jtop import jtop


TELEMETRY_FIELDS = [
    "timestamp", "monotonic_s", "elapsed_s", "power_tot_w", "cpu_utilization_percent",
    "gpu_utilization_percent", "ram_used_mib", "temperature_c",
    "cpu_frequency_mhz", "gpu_frequency_mhz", "emc_frequency_mhz",
    "throttling_status", "process_rss_mib", "valid", "error_message",
]


def _safe_float(value, default=None):
    try:
        return float(value)
    except Exception:
        return default


def extract_sample(jetson, elapsed_s: float, process: psutil.Process | None = None) -> dict:
    stats = dict(jetson.stats)
    power_mw = _safe_float(stats.get("Power TOT"))
    cpu_values = [_safe_float(stats.get(f"CPU{i}")) for i in range(1, 5)]
    cpu_values = [x for x in cpu_values if x is not None]
    gpu = _safe_float(stats.get("GPU"))
    temperatures = [_safe_float(v) for k, v in stats.items() if str(k).startswith("Temp ")]
    temperatures = [x for x in temperatures if x is not None]
    try:
        memory = jetson.memory
        ram_mib = float(memory["RAM"]["used"]) / 1024.0
        emc_mhz = float(memory["EMC"]["cur"]) / 1000.0
    except Exception:
        ram_mib = None
        emc_mhz = None
    try:
        cpus = jetson.cpu["cpu"]
        cpu_freqs = [float(x["freq"]["cur"]) / 1000.0 for x in cpus if x.get("online")]
        cpu_freq = sum(cpu_freqs) / len(cpu_freqs) if cpu_freqs else None
    except Exception:
        cpu_freq = None
    try:
        gpu_data = jetson.gpu
        gpu_freq = float(gpu_data["gpu"]["freq"]["cur"]) / 1000.0
    except Exception:
        gpu_freq = None
    throttle = stats.get("throttling", stats.get("Throttle", "NA"))
    rss = process.memory_info().rss / (1024 ** 2) if process is not None else None
    valid = power_mw is not None and power_mw > 0 and gpu is not None and cpu_values and ram_mib is not None and temperatures
    return {
        "timestamp": datetime.now().astimezone().isoformat(),
        "monotonic_s": time.perf_counter(),
        "elapsed_s": float(elapsed_s),
        "power_tot_w": power_mw / 1000.0 if power_mw is not None else None,
        "cpu_utilization_percent": sum(cpu_values) / len(cpu_values) if cpu_values else None,
        "gpu_utilization_percent": gpu,
        "ram_used_mib": ram_mib,
        "temperature_c": max(temperatures) if temperatures else None,
        "cpu_frequency_mhz": cpu_freq,
        "gpu_frequency_mhz": gpu_freq,
        "emc_frequency_mhz": emc_mhz,
        "throttling_status": str(throttle),
        "process_rss_mib": rss,
        "valid": bool(valid),
        "error_message": "" if valid else "missing required Power TOT/CPU/GPU/RAM/temperature field",
    }


class JtopMonitor:
    def __init__(self, interval: float = 0.5, include_process: bool = True):
        self.interval = float(interval)
        self.include_process = include_process
        self.samples: list[dict] = []
        self.error = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = None

    def _run(self):
        process = psutil.Process() if self.include_process else None
        start = time.perf_counter()
        try:
            with jtop(interval=self.interval) as jetson:
                while not self._stop.is_set():
                    if not jetson.ok():
                        raise RuntimeError("jtop connection is not OK")
                    sample = extract_sample(jetson, time.perf_counter() - start, process)
                    self.samples.append(sample)
                    self._ready.set()
                    if self._stop.wait(self.interval):
                        break
        except BaseException as exc:
            self.error = repr(exc)
            self._ready.set()

    def start(self, timeout: float = 10.0):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            raise RuntimeError("Timed out waiting for first jtop sample")
        if self.error:
            raise RuntimeError(f"jtop monitor failed: {self.error}")
        if not self.samples or not self.samples[0]["valid"]:
            raise RuntimeError(f"First jtop sample invalid: {self.samples[:1]}")

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10.0)
        if self._thread and self._thread.is_alive():
            raise RuntimeError("jtop monitor thread did not stop")
        if self.error:
            raise RuntimeError(f"jtop monitor failed: {self.error}")


def collect_blocking(sample_count: int | None = None, duration_s: float | None = None, interval: float = 0.5):
    if sample_count is None and duration_s is None:
        raise ValueError("sample_count or duration_s is required")
    rows = []
    start = time.perf_counter()
    with jtop(interval=interval) as jetson:
        while True:
            if not jetson.ok():
                raise RuntimeError("jtop connection is not OK")
            rows.append(extract_sample(jetson, time.perf_counter() - start, None))
            if sample_count is not None and len(rows) >= sample_count:
                break
            if duration_s is not None and time.perf_counter() - start >= duration_s:
                break
            time.sleep(interval)
    return rows
