# Dataset: HERIDAL

RescueLens is trained and evaluated on **HERIDAL**, a dataset of aerial images collected for
land search-and-rescue research in Croatia and Bosnia and Herzegovina. Most frames are
4000 × 3000 px nadir views taken from a drone; every person in them is annotated with a box.

| | |
|---|---|
| Authors | D. Božić-Štulić, Ž. Marušić, S. Gotovac — University of Split (FESB) |
| Paper | *Deep Learning Approach in Aerial Imagery for Supporting Land Search and Rescue Missions*, IJCV 127 (2019) |
| Copy used | Aalto University mirror in Pascal VOC format, [Zenodo record 5662351](https://zenodo.org/records/5662351) |
| File | `heridal_keras_retinanet_voc.zip`, 8,311,735,812 bytes, md5 `c2b625ae986898a870b9d8a907aefe45` |
| License | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) |

The original FESB download server did not accept connections when this project was built,
so the Zenodo mirror is the canonical source here. The images and annotations are **not**
redistributed in this repository; `just download-data` fetches them.

## Reproduce

```bash
just download-data   # 8.3 GB, resumes after dropped connections, verifies the md5
just prepare-data    # extract + build tiles (~30 s with 8 worker processes)
```

Both steps are deterministic: the splits come from the archive and negative-tile sampling
is seeded per tile.

## Splits

The archive ships `ImageSets/Main/{train,val,test}.txt`. RescueLens uses them as-is:
`test` is the official HERIDAL test set, and `val` is the mirror's hold-out taken from the
original training images. 37 extra images listed in no split are ignored.

| Split | Images | With people | People | Frames with EXIF GPS |
|---|---:|---:|---:|---:|
| train | 1,391 | 891 | 2,617 | 1,253 |
| val | 155 | 94 | 275 | 144 |
| test | 101 | 101 | 337 | 74 |

Person boxes are small: the median width is 60 px in train and 55 px in test (out of 4,000).
The narrowest annotated person is 19 px wide.

## How the training data is built

A detector that resizes a whole 12 MP frame to its input size shrinks a 55 px person to a few
pixels. At inference RescueLens therefore slices every frame with SAHI (640 px tiles, 20 %
overlap, 48 tiles per 4000 × 3000 frame) and runs YOLO at native resolution on each tile.
Training uses the **same** tiling, so the detector sees people at the scale it will meet in
production.

- Tile origins replicate `sahi.slicing.get_slice_bboxes` exactly (checked in `tests/test_data.py`).
- A box is kept in a tile when at least 50 % of its area is inside, clipped to the tile.
  People cut by a tile border still appear whole in the neighbouring, overlapping tile.
- Every tile containing a person is kept. Person-free tiles are sampled at 5 % for
  train/val, which gives a roughly 1:1 positive/negative mix. Test keeps every tile.

| Split | Tiles with people | Empty tiles | Boxes in tiles |
|---|---:|---:|---:|
| train | 3,266 | 3,277 | 4,358 |
| val | 334 | 337 | 433 |
| test | 435 | 4,381 | 549 |

The 101 test frames are also kept untouched in `data/heridal/test_full/` with their ground
truth (`boxes.json`, pixels). The end-to-end evaluation (YOLO + SAHI ± Claude) runs on
those full frames, not on tiles.

## Known limitations

- Small people are hard to annotate exhaustively. If a person was missed by the annotators,
  a correct detection of them counts as a false positive in the evaluation.
- 561 of the 1,546 train/val images contain no people. They are useful negatives, but
  the scenes, seasons and terrain are all from one region of the Balkans.
- The people were volunteers posing in open terrain, which is easier than real casualties
  under canopy.
