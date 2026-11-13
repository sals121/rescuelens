# Paired comparisons · val · recall at ≤ 1 false alarms per photo

Candidate minus base, same photos in every bootstrap replicate (2,000).

| Base | Candidate | Criterion | Δ recall (points) [95 % CI] | Replicates where candidate wins |
|---|---|---|---:|---:|
| A5-s1024-i1280 | B1-scale075-s1024-i1280 | IoU ≥ 0.5 | -0.7 [-2.7, +1.1] | 17 % |
| A5-s1024-i1280 | B1-scale075-s1024-i1280 | found | -1.1 [-2.7, +0.0] | 0 % |
| A1-s640-i640 | B2-scale075-s640-i640 | IoU ≥ 0.5 | +1.5 [-1.3, +4.2] | 84 % |
| A1-s640-i640 | B2-scale075-s640-i640 | found | +0.4 [-0.8, +1.7] | 61 % |
| A0-baseline-s640-i1280 | A1-s640-i640 | IoU ≥ 0.5 | +9.1 [+4.8, +13.1] | 100 % |
| A0-baseline-s640-i1280 | A1-s640-i640 | found | +3.6 [+1.7, +5.8] | 100 % |
| A1-s640-i640 | A5-s1024-i1280 | IoU ≥ 0.5 | +0.0 [-2.9, +2.9] | 47 % |
| A1-s640-i640 | A5-s1024-i1280 | found | +0.4 [-0.8, +1.6] | 61 % |
| A1-s640-i640 | A6-gsd1.35-s640-i640 | IoU ≥ 0.5 | +0.4 [-1.3, +2.5] | 55 % |
| A1-s640-i640 | A6-gsd1.35-s640-i640 | found | +0.0 [-1.0, +1.0] | 36 % |
| A1-s640-i640 | A2-s640-i960 | IoU ≥ 0.5 | -2.2 [-4.6, +0.4] | 3 % |
| A1-s640-i640 | A2-s640-i960 | found | -0.7 [-2.3, +1.0] | 14 % |
| A0-baseline-s640-i1280 | A3-s640-i1280-nostd | IoU ≥ 0.5 | -0.4 [-2.9, +1.9] | 31 % |
| A0-baseline-s640-i1280 | A3-s640-i1280-nostd | found | -1.1 [-2.8, +0.4] | 5 % |
| A5-s1024-i1280 | A5h-s1024-i1280-hflip | IoU ≥ 0.5 | +0.4 [-1.5, +2.5] | 54 % |
| A5-s1024-i1280 | A5h-s1024-i1280-hflip | found | +0.4 [+0.0, +1.2] | 63 % |
| A5-s1024-i1280 | T1-wbf-A5+A5flip | IoU ≥ 0.5 | +0.4 [+0.0, +1.2] | 63 % |
| A5-s1024-i1280 | T1-wbf-A5+A5flip | found | +0.0 [-1.0, +1.0] | 34 % |
| A5-s1024-i1280 | E1-wbf-A5+B1 | IoU ≥ 0.5 | +1.1 [+0.0, +2.2] | 95 % |
| A5-s1024-i1280 | E1-wbf-A5+B1 | found | +0.7 [+0.0, +1.8] | 86 % |
| A5-s1024-i1280 | E2-wbf-A5+A1 | IoU ≥ 0.5 | +1.8 [+0.0, +4.2] | 95 % |
| A5-s1024-i1280 | E2-wbf-A5+A1 | found | +1.1 [+0.0, +2.3] | 95 % |
