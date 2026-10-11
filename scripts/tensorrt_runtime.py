#!/usr/bin/env python3
from __future__ import annotations

import ctypes
import ctypes.util
import sys
import time
from pathlib import Path

import numpy as np

TRT_SITE = "/home/jetson/.local/lib/python3.8/site-packages"
if TRT_SITE not in sys.path:
    sys.path.insert(0, TRT_SITE)
import tensorrt as trt


CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2


def _load_cudart():
    candidates = [ctypes.util.find_library("cudart"), "libcudart.so", "/usr/local/cuda/lib64/libcudart.so", "/usr/local/cuda-10.2/targets/aarch64-linux/lib/libcudart.so"]
    last = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return ctypes.CDLL(candidate), candidate
        except OSError as exc:
            last = exc
    raise RuntimeError(f"Unable to load libcudart: {last!r}")


def _configure(cudart):
    cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    cudart.cudaMalloc.restype = ctypes.c_int
    cudart.cudaFree.argtypes = [ctypes.c_void_p]
    cudart.cudaFree.restype = ctypes.c_int
    cudart.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
    cudart.cudaMemcpyAsync.restype = ctypes.c_int
    cudart.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    cudart.cudaStreamCreate.restype = ctypes.c_int
    cudart.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
    cudart.cudaStreamDestroy.restype = ctypes.c_int
    cudart.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
    cudart.cudaStreamSynchronize.restype = ctypes.c_int
    cudart.cudaEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    cudart.cudaEventCreate.restype = ctypes.c_int
    cudart.cudaEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    cudart.cudaEventRecord.restype = ctypes.c_int
    cudart.cudaEventSynchronize.argtypes = [ctypes.c_void_p]
    cudart.cudaEventSynchronize.restype = ctypes.c_int
    cudart.cudaEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p]
    cudart.cudaEventElapsedTime.restype = ctypes.c_int
    cudart.cudaEventDestroy.argtypes = [ctypes.c_void_p]
    cudart.cudaEventDestroy.restype = ctypes.c_int


def _check(code: int, operation: str):
    if int(code) != 0:
        raise RuntimeError(f"{operation} failed with CUDA runtime error {code}")


def _numpy_dtype(trt_dtype):
    """Map TensorRT 8.0 dtypes without its NumPy-1.24-incompatible nptype()."""
    mapping = {
        "DataType.FLOAT": np.float32,
        "DataType.HALF": np.float16,
        "DataType.INT8": np.int8,
        "DataType.INT32": np.int32,
        "DataType.BOOL": np.bool_,
    }
    key = str(trt_dtype)
    if key not in mapping:
        raise TypeError(f"Unsupported TensorRT binding dtype: {key}")
    return np.dtype(mapping[key])


class CudaEventPair:
    def __init__(self, cudart):
        self.cudart = cudart
        self.start = ctypes.c_void_p()
        self.end = ctypes.c_void_p()
        _check(cudart.cudaEventCreate(ctypes.byref(self.start)), "cudaEventCreate(start)")
        _check(cudart.cudaEventCreate(ctypes.byref(self.end)), "cudaEventCreate(end)")

    def measure(self, stream, operation):
        _check(self.cudart.cudaEventRecord(self.start, stream), "cudaEventRecord(start)")
        result = operation()
        _check(self.cudart.cudaEventRecord(self.end, stream), "cudaEventRecord(end)")
        _check(self.cudart.cudaEventSynchronize(self.end), "cudaEventSynchronize(end)")
        elapsed = ctypes.c_float()
        _check(self.cudart.cudaEventElapsedTime(ctypes.byref(elapsed), self.start, self.end), "cudaEventElapsedTime")
        return result, float(elapsed.value)

    def close(self):
        if self.start:
            self.cudart.cudaEventDestroy(self.start)
            self.start = None
        if self.end:
            self.cudart.cudaEventDestroy(self.end)
            self.end = None


class TensorRTRunner:
    """Static-shape TensorRT runner with CUDA-event H2D/engine/D2H timings."""

    def __init__(self, engine_path: str | Path, expected_shape=(1, 3, 640, 960)):
        self.engine_path = Path(engine_path)
        self.expected_shape = tuple(int(x) for x in expected_shape)
        self.cudart, self.cudart_path = _load_cudart()
        _configure(self.cudart)
        self.stream = ctypes.c_void_p()
        _check(self.cudart.cudaStreamCreate(ctypes.byref(self.stream)), "cudaStreamCreate")
        self.events = CudaEventPair(self.cudart)
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError("TensorRT engine deserialization returned None")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("TensorRT execution context creation returned None")
        self.bindings = [0] * self.engine.num_bindings
        self.device_buffers: list[ctypes.c_void_p] = []
        self.host_outputs: dict[int, np.ndarray] = {}
        self.output_indices: list[int] = []
        self.input_index = None
        self.binding_metadata = []
        self._configure_bindings()
        self._allocate_buffers()

    def _configure_bindings(self):
        for index in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(index)
            is_input = bool(self.engine.binding_is_input(index))
            engine_shape = tuple(int(x) for x in self.engine.get_binding_shape(index))
            if is_input:
                if self.input_index is not None:
                    raise RuntimeError("Multiple input bindings are not supported")
                self.input_index = index
                if any(x < 0 for x in engine_shape):
                    if not self.context.set_binding_shape(index, self.expected_shape):
                        raise RuntimeError(f"Failed to set static input shape {self.expected_shape}")
            else:
                self.output_indices.append(index)
            runtime_shape = tuple(int(x) for x in self.context.get_binding_shape(index))
            self.binding_metadata.append({
                "index": index,
                "name": name,
                "is_input": is_input,
                "engine_shape": list(engine_shape),
                "runtime_shape": list(runtime_shape),
                "dtype": str(self.engine.get_binding_dtype(index)),
            })
        if self.input_index is None:
            raise RuntimeError("No TensorRT input binding found")
        actual = tuple(int(x) for x in self.context.get_binding_shape(self.input_index))
        if actual != self.expected_shape:
            raise RuntimeError(f"Engine input shape {actual} does not equal required {self.expected_shape}")

    def _device_alloc(self, nbytes: int, label: str):
        pointer = ctypes.c_void_p()
        _check(self.cudart.cudaMalloc(ctypes.byref(pointer), int(nbytes)), f"cudaMalloc({label})")
        self.device_buffers.append(pointer)
        return pointer

    def _allocate_buffers(self):
        self.input_dtype = _numpy_dtype(self.engine.get_binding_dtype(self.input_index))
        if self.input_dtype != np.dtype(np.float32):
            raise RuntimeError(f"Strict FP32 input required; engine binding is {self.input_dtype}")
        input_bytes = int(np.prod(self.expected_shape)) * self.input_dtype.itemsize
        self.device_input = self._device_alloc(input_bytes, "input")
        self.input_nbytes = input_bytes
        self.bindings[self.input_index] = int(self.device_input.value)
        for index in self.output_indices:
            shape = tuple(int(x) for x in self.context.get_binding_shape(index))
            if any(x < 0 for x in shape):
                raise RuntimeError(f"Unresolved output shape for {self.engine.get_binding_name(index)}: {shape}")
            dtype = _numpy_dtype(self.engine.get_binding_dtype(index))
            host = np.empty(shape, dtype=dtype)
            device = self._device_alloc(host.nbytes, f"output:{self.engine.get_binding_name(index)}")
            self.bindings[index] = int(device.value)
            self.host_outputs[index] = host

    def synchronize(self):
        _check(self.cudart.cudaStreamSynchronize(self.stream), "cudaStreamSynchronize")

    def run_timed(self, input_array: np.ndarray):
        if input_array.dtype != self.input_dtype or tuple(input_array.shape) != self.expected_shape or not input_array.flags.c_contiguous:
            raise ValueError(f"Input must be contiguous {self.input_dtype} {self.expected_shape}; got {input_array.dtype} {input_array.shape} contiguous={input_array.flags.c_contiguous}")
        wall_start = time.perf_counter_ns()
        _, h2d_ms = self.events.measure(
            self.stream,
            lambda: _check(self.cudart.cudaMemcpyAsync(self.device_input, ctypes.c_void_p(input_array.ctypes.data), input_array.nbytes, CUDA_MEMCPY_HOST_TO_DEVICE, self.stream), "cudaMemcpyAsync H2D"),
        )

        def execute():
            ok = self.context.execute_async_v2(self.bindings, int(self.stream.value))
            if not ok:
                raise RuntimeError("TensorRT execute_async_v2 returned False")

        _, engine_ms = self.events.measure(self.stream, execute)

        def copy_outputs():
            for index in self.output_indices:
                host = self.host_outputs[index]
                _check(self.cudart.cudaMemcpyAsync(ctypes.c_void_p(host.ctypes.data), ctypes.c_void_p(self.bindings[index]), host.nbytes, CUDA_MEMCPY_DEVICE_TO_HOST, self.stream), f"cudaMemcpyAsync D2H:{self.engine.get_binding_name(index)}")

        _, d2h_ms = self.events.measure(self.stream, copy_outputs)
        self.synchronize()
        wall_end = time.perf_counter_ns()
        timings = {
            "h2d_ms": h2d_ms,
            "tensorrt_engine_ms": engine_ms,
            "d2h_ms": d2h_ms,
            "transfer_inclusive_inference_ms": (wall_end - wall_start) / 1_000_000.0,
        }
        return [self.host_outputs[i] for i in self.output_indices], timings

    def metadata(self) -> dict:
        return {
            "engine_path": str(self.engine_path.resolve()),
            "tensorrt_version": trt.__version__,
            "cudart_path": self.cudart_path,
            "execution_api": "IExecutionContext.execute_async_v2",
            "cuda_stream": "one package-local runtime stream",
            "timing": "CUDA events around H2D, execute_async_v2, and D2H; each event end synchronized; stream synchronized before return",
            "pinned_memory": False,
            "buffer_allocation": "device buffers allocated once before any warm-up; reusable NumPy host output buffers",
            "binding_metadata": self.binding_metadata,
        }

    def close(self):
        if getattr(self, "events", None):
            self.events.close()
            self.events = None
        for pointer in getattr(self, "device_buffers", []):
            if pointer and pointer.value:
                self.cudart.cudaFree(pointer)
        self.device_buffers = []
        if getattr(self, "stream", None) and self.stream.value:
            self.cudart.cudaStreamDestroy(self.stream)
            self.stream = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
