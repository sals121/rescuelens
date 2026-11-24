# bench_small

Twelve HERIDAL test photos, picked with a fixed seed: the fine-tuned detector's real
predictions (YOLO26s + SAHI, score >= 0.05) and the annotated person boxes. Coordinates
only, no images. `expected.json` is the output of

    uv run rescuelens bench score --preds tests/fixtures/bench_small/preds.json \
        --gt tests/fixtures/bench_small/gt.json --write tests/fixtures/bench_small/expected.json

CI runs the same command with `--check` instead of `--write`, so any change to matching,
thresholds or the bootstrap shows up as a diff. Regenerate it only for an intended change.

Annotations derived from HERIDAL (Božić-Štulić, Marušić, Gotovac; CC BY 3.0).
