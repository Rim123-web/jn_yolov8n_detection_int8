# Bottleneck analysis

The largest measured stage is `tensorrt_engine_ms` at 89.451 ms. TensorRT engine execution accounts for 61.10% of mean end-to-end latency; 38.90% is outside the TensorRT engine. NMS alone contributes 2.02%.
