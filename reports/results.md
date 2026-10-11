# Results

| Metric | Value |
|---|---:|
| mAP50 | 0.977208 |
| mAP50-95 | 0.830817 |
| Precision @0.25 | 0.868574 |
| Recall @0.25 | 0.974423 |
| F1 @0.25 | 0.918459 |
| Engine latency (ms) | 89.451 |
| Engine FPS | 11.179 |
| E2E latency (ms) | 146.395 |
| Wall-clock FPS | 6.197 |
| Mean active power (W) | 6.845 |
| Gross energy/image (J) | 1.104559 |
| Idle-corrected energy/image (J) | 0.698973 |

Recommendation: **FP16**. FP16 remains preferred because INT8 does not simultaneously improve speed and energy while preserving mAP50-95; the Nano reports no native fast-INT8 support.
