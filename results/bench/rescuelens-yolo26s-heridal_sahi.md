# Benchmark · rescuelens-yolo26s-heridal_sahi

- **val**: 155 images, 275 people, 0.595 km² (assumed altitude)
- **test**: 101 images, 337 people, 0.461 km² (assumed altitude)

## Val · person found: IoU ≥ 0.1 or a centre inside the other box (operational)

| Rule | Conf | Found | Recall % [95 % CI] | Precision % | False alarms / image | False alarms / km² |
|---|---:|---:|---:|---:|---:|---:|
| product default | 0.400 | 257/275 | 93.5 [90.1, 96.7] | 64.3 [58.5, 70.2] | 0.93 [0.70, 1.16] | 242 [184, 301] |
| max F1 (IoU 0.5) | 0.661 | 231/275 | 84.0 [78.6, 90.0] | 83.8 [78.7, 88.5] | 0.29 [0.18, 0.42] | 76 [48, 109] |
| recall >= 0.90 (found) | 0.534 | 248/275 | 90.2 [85.7, 94.6] | 74.0 [68.3, 79.4] | 0.57 [0.41, 0.74] | 148 [107, 194] |
| recall >= 0.95 (found) | 0.251 | 262/275 | 95.3 [92.8, 97.6] | 53.1 [46.8, 59.4] | 1.52 [1.19, 1.84] | 395 [313, 475] |
| recall >= 0.98 (found) | not reachable on val | | | | | |

## Test · person found: IoU ≥ 0.1 or a centre inside the other box (operational)

| Rule | Conf | Found | Recall % [95 % CI] | Precision % | False alarms / image | False alarms / km² |
|---|---:|---:|---:|---:|---:|---:|
| product default | 0.400 | 290/337 | 86.1 [81.6, 90.9] | 71.1 [66.5, 75.4] | 1.17 [0.93, 1.43] | 256 [204, 313] |
| max F1 (IoU 0.5) | 0.661 | 247/337 | 73.3 [67.5, 79.6] | 87.9 [84.3, 91.4] | 0.34 [0.22, 0.47] | 74 [49, 102] |
| recall >= 0.90 (found) | 0.534 | 276/337 | 81.9 [76.5, 87.7] | 79.4 [75.3, 83.5] | 0.71 [0.53, 0.91] | 156 [117, 198] |
| recall >= 0.95 (found) | 0.251 | 304/337 | 90.2 [86.2, 94.1] | 60.4 [54.6, 65.4] | 1.99 [1.61, 2.39] | 436 [355, 527] |
| recall >= 0.98 (found) | not reachable on val | | | | | |
| YOLO + claude-opus-5-5 (filter) | 0.400 | 287/337 | 85.2 [80.4, 89.9] | 88.3 [84.5, 92.0] | 0.38 [0.24, 0.52] | 82 [53, 115] |
| YOLO + claude-haiku-5-5 (filter) | 0.400 | 253/337 | 75.1 [69.8, 80.5] | 92.0 [88.7, 95.5] | 0.22 [0.11, 0.33] | 48 [25, 72] |

## Val · IoU ≥ 0.5, duplicates count as false positives (literature)

| Rule | Conf | Found | Recall % [95 % CI] | Precision % | False alarms / image | False alarms / km² |
|---|---:|---:|---:|---:|---:|---:|
| product default | 0.400 | 232/275 | 84.4 [79.0, 89.9] | 57.6 [52.0, 62.8] | 1.10 [0.83, 1.38] | 287 [218, 356] |
| max F1 (IoU 0.5) | 0.661 | 214/275 | 77.8 [71.2, 85.0] | 77.0 [71.7, 82.6] | 0.41 [0.26, 0.59] | 108 [69, 152] |
| recall >= 0.90 (found) | 0.534 | 226/275 | 82.2 [76.0, 88.3] | 66.9 [61.3, 72.1] | 0.72 [0.52, 0.94] | 188 [137, 245] |
| recall >= 0.95 (found) | 0.251 | 233/275 | 84.7 [79.5, 90.1] | 46.5 [40.9, 51.9] | 1.73 [1.35, 2.11] | 451 [357, 547] |
| recall >= 0.98 (found) | not reachable on val | | | | | |

## Test · IoU ≥ 0.5, duplicates count as false positives (literature)

| Rule | Conf | Found | Recall % [95 % CI] | Precision % | False alarms / image | False alarms / km² |
|---|---:|---:|---:|---:|---:|---:|
| product default | 0.400 | 273/337 | 81.0 [75.8, 86.8] | 66.7 [62.3, 71.0] | 1.35 [1.09, 1.63] | 295 [238, 362] |
| max F1 (IoU 0.5) | 0.661 | 237/337 | 70.3 [64.6, 76.8] | 84.3 [80.3, 88.5] | 0.44 [0.30, 0.59] | 95 [64, 131] |
| recall >= 0.90 (found) | 0.534 | 261/337 | 77.4 [71.7, 83.6] | 74.8 [70.6, 79.1] | 0.87 [0.66, 1.11] | 191 [145, 244] |
| recall >= 0.95 (found) | 0.251 | 285/337 | 84.6 [79.9, 89.3] | 56.2 [51.0, 61.1] | 2.20 [1.80, 2.63] | 481 [395, 583] |
| recall >= 0.98 (found) | not reachable on val | | | | | |
| YOLO + claude-opus-5-5 (filter) | 0.400 | 270/337 | 80.1 [75.0, 85.5] | 82.8 [78.6, 87.1] | 0.55 [0.39, 0.75] | 121 [85, 165] |
| YOLO + claude-haiku-5-5 (filter) | 0.400 | 239/337 | 70.9 [65.4, 76.8] | 86.6 [82.4, 90.9] | 0.37 [0.23, 0.52] | 80 [49, 116] |

## Cost and latency of the Claude arms (test)

| Arm | Calls | Cost / image | Call p50 / p95 | Image p50 / p95 (serial) |
|---|---:|---:|---:|---:|
| claude-opus-5-5 | 409 | $0.0732 | 4.2 / 8.2 s | 16.6 / 58.5 s |
| claude-haiku-5-5 | 409 | $0.0022 | 3.9 / 5.7 s | 13.9 / 47.5 s |

## Detector latency per image

| Split | Detect p50 / p95 | + JPEG decode p50 / p95 |
|---|---:|---:|
| val | 1.10 / 1.37 s | 1.18 / 1.47 s |
| test | 1.17 / 1.59 s | 1.26 / 1.77 s |
