# Calibration Selection Policy

Select up to 90 valid, non-overlapping images using deterministic greedy multi-label class coverage, object coverage, and aspect-ratio diversity. Ties use Python seed 42; the final execution order is sorted. Labels guide selection/auditing only and are not inputs to TensorRT entropy calibration.
