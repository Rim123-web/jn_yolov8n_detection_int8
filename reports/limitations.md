# Limitations

- Jetson Nano reports no native fast-INT8 capability; performance may be worse than FP16.
- TensorRT may retain higher-precision internal layers; per-layer runtime precision was not fully exposed by TensorRT 8.0.1.
- Power TOT is board-level telemetry, not GPU-only power. Idle-corrected energy subtracts the fresh idle baseline and may be sensitive to short-run telemetry cadence.
- Validation labels contain recorded boxes that cross normalized image boundaries; labels are not calibration inputs.
