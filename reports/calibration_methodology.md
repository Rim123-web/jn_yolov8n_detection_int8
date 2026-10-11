# Calibration Methodology

# Methodology

- Task: YOLOv8 multi-class object detection (leccino, varietafrancese, cimadimelfimisto, ulivigrandi, montalbanosecolari, irriguo).
- Source model: `/home/jetson/Downloads/jetson_folder/yolov8n_detection_fp32_benchmark/yolov8n_detection_tensorrt_int8_640x960_batch1_warmup10_20260720_192532/model/oct_albument2_v6_source.pt`; SHA-256 `7c09b3b9866f633c60005b7446f29e9612fae00a8789ab9fe983b32b2f08828d`.
- Shared ONNX: `/home/jetson/Downloads/jetson_folder/yolov8n_detection_fp32_benchmark/yolov8n_detection_tensorrt_int8_640x960_batch1_warmup10_20260720_192532/onnx/oct_albument2_v6_inference_only.onnx`; SHA-256 `6a58292b377d7bdfe208fa23adff170e864d9d8695dba0c7ab4d6e426ff15524`; reused without re-export.
- Static input `[1,3,640,960]`; batch size 1.
- INT8 calibration dataset: `/home/jetson/Downloads/jetson_folder/jetson_valid_imgs`.
- Deterministic calibration subset: 90 images, seed 42, greedy multi-label class/object/aspect coverage.
- Algorithm: TensorRT IInt8EntropyCalibrator2 (Entropy Calibration 2), batch size 1.
- Calibration uses canonical RGB/EXIF/letterbox-114/bilinear/divide-by-255/contiguous NCHW FP32 preprocessing. Labels were used only for audit and representative selection, not entropy calibration.
- Calibration/test independence: 0 exact file, 0 decoded-pixel, 0 unresolved dHash overlap.
- Cache: `/home/jetson/Downloads/jetson_folder/yolov8n_detection_fp32_benchmark/yolov8n_detection_tensorrt_int8_640x960_batch1_warmup10_20260720_192532/calibration/oct_albument2_v6_int8_calibration.cache`; SHA-256 `d31edf77cac862ad164184524127795b23662fc36bff08e2e747fdc3630a14b2`; freshly created, no legacy cache reused.
- Execution: TensorRT INT8-enabled calibrated execution with FP32 input/output bindings and TensorRT-permitted internal higher-precision fallbacks. This is not described as pure or exclusive INT8.
- FP16 builder flag OFF; TF32 OFF; DLA disabled; platform fast INT8 = False.
- Correctness: five backends on the same 20 images; FP32/INT8 and FP16/INT8 paired class agreement both 1.000 and 1.000.
- Test data/order, preprocessing, postprocessing, class-aware NMS (IoU 0.7), confidence 0.25, AP floor 0.001, and IoU matching 0.5 match the accepted FP32/FP16 packages.
- Exactly 10 real-image full-pipeline warm-ups used the same paths as FP32/FP16 and were excluded from all reported results.
- Every one of 192 test images was measured exactly once. CUDA events independently timed H2D, engine, and D2H; CPU monotonic timing covered preprocessing, postprocessing, and E2E.
- jtop/jetson-stats sampled a fresh 60-second idle baseline and the accepted measured interval at 0.5 s. Power TOT energy used trapezoidal integration.


## Selected Class Coverage

- leccino: 8 images, 323 objects
- varietafrancese: 9 images, 318 objects
- cimadimelfimisto: 7 images, 117 objects
- ulivigrandi: 29 images, 310 objects
- montalbanosecolari: 16 images, 110 objects
- irriguo: 21 images, 449 objects
