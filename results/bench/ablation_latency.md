# Detector latency · 40 val photos, 2 rounds, interleaved

Device: cuda:0. Detector only (tiling, inference, merging); JPEG decoding excluded.

| Configuration | p50 | p95 | mean |
|---|---:|---:|---:|
| A0-baseline-s640-i1280 | 3.44 s | 5.92 s | 3.60 s |
| A1-s640-i640 | 2.99 s | 5.98 s | 3.50 s |
| A5-s1024-i1280 | 1.75 s | 2.80 s | 1.77 s |
| A6-gsd1.35-s640-i640 | 3.52 s | 10.53 s | 4.33 s |
