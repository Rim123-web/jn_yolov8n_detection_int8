# FP32 / FP16 / INT8 Comparison

| Precision | mAP50 | mAP50-95 | Engine ms | Engine FPS | E2E ms | Gross J/image | Net J/image |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP32 | 0.977236 | 0.830485 | 91.636 | 10.913 | 148.078 | 1.081605 | 0.686215 |
| FP16 | 0.977178 | 0.829597 | 66.306 | 15.082 | 123.110 | 0.816820 | 0.414618 |
| INT8 | 0.977208 | 0.830817 | 89.451 | 11.179 | 146.395 | 1.104559 | 0.698973 |

**Recommendation: FP16.** FP16 remains preferred because INT8 does not simultaneously improve speed and energy while preserving mAP50-95; the Nano reports no native fast-INT8 support.
