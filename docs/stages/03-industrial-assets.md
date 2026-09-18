# Stage 3 — industrial receiving modules

## What changed

The six fully saturated open boxes in `sim/sorting_line.py` are replaced by a
shared food-factory receiving module in `sim/usd_kit.py`. Each module now has:

- a neutral removable tote;
- a stainless support frame and adjustable feet;
- a sloped transfer chute with side guides;
- a low-energy rear bumper and tote handle;
- class colour used only on a front band and chute-edge accents; and
- restrained floor markings and smaller existing class signs.

The conveyor layout, output mapping, diverters, inspection geometry and camera
domain are unchanged.

## Static asset validation

`--usd-only` builds the same stage as the sorting demo, exports
`sorting_line.usda`, and exits before Replicator or the classifier. This gives
asset work a deterministic, checkpoint-independent validation path:

```bash
mkdir -p outputs/stage3-preview && chmod 777 outputs/stage3-preview
docker run --rm --gpus all -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -v "$PWD:/workspace:rw" --entrypoint /isaac-sim/python.sh \
  "${ISAAC_SIM_IMAGE:-nvcr.io/nvidia/isaac-sim:6.1.0}" \
  /workspace/sim/sorting_line.py --usd-only --items 1 \
  --out /workspace/outputs/stage3-preview
```

The generated USD and preview outputs remain ignored under `outputs/`.

## Verification

- `python3 -m compileall -q sim`
- `PYTHONPATH=. .venv/bin/python -m pytest -q` — 12 passed
- Isaac Sim 6.1 parsed and constructed all redesigned module prims without a
  Python or USD exception.

The legacy Replicator preview path can stall during renderer pre-warm on this
workstation. That runtime issue is intentionally kept separate from the new
USD-only asset validation path.
