# Stage 5 — production perception integration

Status: accepted on the RTX PRO 6000 workstation on 2026-09-18.

## What changed

- Moved trained inference to a host-side HTTP service because the stock Isaac
  Sim container does not include PyTorch or timm.
- Added a fail-closed client contract. Only `status=ok` with a known bin can
  actuate the robot; malformed, low-confidence, foreign, and empty decisions
  cannot select a destination.
- Kept the fine-type network for the operator-facing cheese name, while a
  direct-bin network owns the safety-critical routing decision.
- Replaced colored model-mode proxy cubes with held-out photographic cutouts
  on the same carrier geometry used by the perception dataset.
- Captured factory-camera adaptation images in bounded Isaac processes. The
  process restart is deliberate: Isaac Sim 6.1 crashed during experimental
  scene teardown after long capture runs.
- Added a reusable checkpoint evaluator for explicit manifest slices.

## Leak-free target-domain adaptation

The capture contains 2,007 images from 669 source objects. Each source object's
group keeps its original split, so alternate views never cross splits:

| split | camera crops |
|---|---:|
| train | 1,407 |
| validation | 300 |
| test | 300 |

The integrity check found 2,007 unique capture IDs, zero missing files, and
zero groups crossing splits. Three views cover scale/rotation variation plus
the exact production carrier geometry.

Generated images, manifests, and checkpoints remain ignored. Reproduce them
on the Isaac workstation with:

```bash
CHEESE_CAPTURE_VIEWS=3 infra/isaac-sim/capture-all-domain.sh

env LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib" \
  .venv/bin/python src/train.py \
  --task sim_bin \
  --manifest data/processed/manifest_factory_adapt.csv \
  --model convnext_base.fb_in22k_ft_in1k_384 \
  --img-size 384 --epochs 20 --batch-size 64 --lr 1e-4 --workers 8 \
  --balanced-sampler --out runs/sim_bin_adapt_v2
```

## Model evidence

The selected checkpoint was chosen on the 300-image target-camera validation
slice by the operational correct-and-confident rate at the fixed 0.55 threshold.

| target-camera validation model | top-1 | macro-F1 | coverage | accuracy when covered | correct and covered |
|---|---:|---:|---:|---:|---:|
| two-view adapted checkpoint | 0.6867 | 0.4030 | 0.8000 | 0.7958 | 0.6367 |
| three-view adapted checkpoint | 0.7433 | 0.4010 | 0.8867 | 0.7857 | 0.6967 |

The selected model's untouched 300-image target-camera test slice measured
0.7433 top-1, 0.3740 macro-F1, 0.8867 coverage, and 0.7932 accuracy on covered
decisions. The combined original-plus-camera test set measured 0.7763 top-1
and 0.5509 macro-F1. These results expose substantial rare-bin weakness; they
are not real-camera or production-plant accuracy.

## Full Isaac run

The deterministic 11-object scenario completed with the production classifier:

| metric | result |
|---|---:|
| camera detection | 11/11 (100%) |
| fine-type classification | 5/11 (45.5%) |
| attempted physical picks | 6/6 succeeded |
| cheese pick coverage | 6/10 (60%) |
| correct-bin placement | 3/10 (30%) |
| safe foreign-object rejection | 1/1 (100%) |
| empty-frame handling | 1/1 (100%) |
| end-to-end object outcome | 4/11 (36.4%) |
| mean cycle time | 12.80 s |

This is a functioning perception-to-action system, not a claim of production
accuracy. The main remaining limitation is rare-bin perception: the controller
and arm completed every attempted move, while the trained router rejected or
misrouted several uncommon cheese types.

The trained-model run is reported separately from the perfect development
classifier run. The latter validates deterministic plumbing and robot control,
not perception quality.

## Artifact boundary

The workstation retains `runs/sim_type13/best.pt` and
`runs/sim_bin_adapt_v2/best.pt`. They are about 335 MB each and are not committed
because the repository does not configure Git LFS. Dataset images, generated
manifests, runtime frames, and machine-readable reports are likewise ignored.
