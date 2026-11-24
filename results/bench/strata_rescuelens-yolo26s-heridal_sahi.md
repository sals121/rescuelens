# Recall by stratum · rescuelens-yolo26s-heridal_sahi · val

## Val · person size, √(w·h) of the annotated box · criterion: IoU ≥ 0.5

| Stratum | People | Photos | Recall % at conf 0.40 | Recall % at any box (conf 0.05) |
|---|---:|---:|---:|---:|
| < 30 px | 1 | 1 | 1/1 · 100.0 [100.0, 100.0] | 1/1 · 100.0 [100.0, 100.0] |
| 30-40 px | 16 | 11 | 11/16 · 68.8 [35.0, 100.0] | 12/16 · 75.0 [40.0, 100.0] |
| 40-50 px | 42 | 28 | 34/42 · 81.0 [64.3, 96.3] | 35/42 · 83.3 [70.7, 96.3] |
| 50-60 px | 57 | 39 | 47/57 · 82.5 [69.4, 94.7] | 48/57 · 84.2 [71.2, 96.3] |
| 60-80 px | 116 | 50 | 104/116 · 89.7 [83.7, 95.5] | 106/116 · 91.4 [85.7, 97.0] |
| 80-120 px | 43 | 29 | 35/43 · 81.4 [68.4, 92.7] | 35/43 · 81.4 [68.4, 92.7] |

## Val · person size, √(w·h) of the annotated box · criterion: person found (IoU ≥ 0.1 or centre inside)

| Stratum | People | Photos | Recall % at conf 0.40 | Recall % at any box (conf 0.05) |
|---|---:|---:|---:|---:|
| < 30 px | 1 | 1 | 1/1 · 100.0 [100.0, 100.0] | 1/1 · 100.0 [100.0, 100.0] |
| 30-40 px | 16 | 11 | 13/16 · 81.2 [62.5, 100.0] | 15/16 · 93.8 [85.0, 100.0] |
| 40-50 px | 42 | 28 | 37/42 · 88.1 [75.0, 100.0] | 39/42 · 92.9 [85.7, 100.0] |
| 50-60 px | 57 | 39 | 53/57 · 93.0 [86.0, 100.0] | 56/57 · 98.2 [94.7, 100.0] |
| 60-80 px | 116 | 50 | 113/116 · 97.4 [94.1, 100.0] | 115/116 · 99.1 [97.2, 100.0] |
| 80-120 px | 43 | 29 | 40/43 · 93.0 [85.2, 100.0] | 40/43 · 93.0 [85.2, 100.0] |

## Val · ground resolution (DJI RelativeAltitude, flat ground; Canon has none) · criterion: person found (IoU ≥ 0.1 or centre inside)

| Stratum | People | Photos | Recall % at conf 0.40 | Recall % at any box (conf 0.05) |
|---|---:|---:|---:|---:|
| < 1.25 cm/px | 6 | 5 | 6/6 · 100.0 [100.0, 100.0] | 6/6 · 100.0 [100.0, 100.0] |
| 1.25-1.5 cm/px | 55 | 36 | 54/55 · 98.2 [94.7, 100.0] | 55/55 · 100.0 [100.0, 100.0] |
| 1.5-1.75 cm/px | 36 | 16 | 34/36 · 94.4 [87.5, 100.0] | 35/36 · 97.2 [92.9, 100.0] |
| 2-2.5 cm/px | 31 | 14 | 31/31 · 100.0 [100.0, 100.0] | 31/31 · 100.0 [100.0, 100.0] |

## Val · camera (EXIF model) · criterion: person found (IoU ≥ 0.1 or centre inside)

| Stratum | People | Photos | Recall % at conf 0.40 | Recall % at any box (conf 0.05) |
|---|---:|---:|---:|---:|
| Canon PowerShot S110 | 122 | 13 | 111/122 · 91.0 [83.6, 97.2] | 116/122 · 95.1 [90.2, 98.6] |
| FC220 | 97 | 57 | 94/97 · 96.9 [93.8, 100.0] | 96/97 · 99.0 [96.9, 100.0] |
| FC300X | 56 | 24 | 52/56 · 92.9 [86.5, 100.0] | 54/56 · 96.4 [91.9, 100.0] |

## Test, described only (annotations + EXIF, no predictions used)

Expected recall (criterion: person found (IoU ≥ 0.1 or centre inside), conf 0.40) if only person size mattered: val recall per size bin, weighted by the group's sizes.

| Camera | People | Median size | Share < 40 px | Median rel. altitude | Expected recall from size |
|---|---:|---:|---:|---:|---:|
| Canon PowerShot S100 | 7 | 53 px | 14 % | n/a | 91.8 % |
| Canon PowerShot S110 | 115 | 65 px | 5 % | n/a | 94.1 % |
| FC220 | 33 | 60 px | 0 % | 45 m | 94.2 % |
| FC300X | 106 | 51 px | 20 % | 51 m | 90.0 % |
| FC550 | 42 | 53 px | 2 % | 101 m | 91.7 % |
| no EXIF | 34 | 53 px | 15 % | n/a | 91.6 % |
