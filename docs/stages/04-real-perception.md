# Stage 4 — reproducible perception recovery

Status: accepted on the RTX PRO 6000 workstation on 2026-09-18.

## What changed

- Added an idempotent downloader for CHEESE-HIDB, cheese-images, and Food
  Recognition 2022.
- Added a public Hugging Face parquet fallback because the upstream Dataset
  Tools Dropbox archive was disabled during the run.
- Kept only unambiguous single-object fallback rows. The mirror stores ordered
  labels and boxes separately, and visual inspection showed that multi-object
  rows can pair them incorrectly.
- Refined fallback bounding boxes with GrabCut before placing the cutouts in
  Isaac Sim.
- Rendered 3,645 target-domain images, with grouped train/validation/test
  splits so the five views of one source object never cross split boundaries.
- Trained both the 13-class fine-type model and the direct 7-class sorting
  model with inverse-frequency sampling.

## Evidence from this recovery run

The manifest contains 969 distinct groups and no group crosses a split:

| split | images | groups |
|---|---:|---:|
| train | 2,555 | 679 |
| validation | 545 | 145 |
| test | 545 | 145 |

Class totals are 45 blue, 180 fresh, 680 hard, 260 semi-hard, 250 soft,
1,930 non-cheese, and 300 empty-belt renders.

| model | test top-1 | test top-5 | test macro-F1 |
|---|---:|---:|---:|
| 13-class fine type | 0.7872 | 0.9211 | 0.4365 |
| 7-class direct sort | 0.7541 | 0.9725 | 0.5006 |

The fine-type checkpoint exported to ONNX with a measured maximum absolute
PyTorch/ONNX logit difference of `2.15e-06`.

These are synthetic-domain results from the smaller public fallback, not
real-camera accuracy. Rare fine types have very small test support, and
`processed_cheese` has no test example in this grouped split; the per-class
report in `runs/sim_type13/results.json` is the authoritative evidence.

## Artifact boundary

Raw datasets, rendered images, checkpoints, and ONNX tensors remain ignored
because they are reconstructible and exceed GitHub's normal file-size limit.
The repository keeps the source, sidecars, exact metrics, and reproduction
commands. The workstation retains the generated checkpoints for the demo.
