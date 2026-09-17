---
tags: [ops, simulation]
---

# Isaac Sim setup

For an interactive GUI on an SSH/Brev machine, use the reproducible
[`infra/isaac-sim`](../../infra/isaac-sim/README.md) stack. It keeps WebRTC
server-local and publishes a browser desktop over authenticated HTTPS.

## Batch rendering

A LaunchPad stack already runs Isaac Sim containers. **None of it was modified.**
Rendering uses a disposable container instead:

```bash
docker run --rm --gpus all -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -v /home/nvidia/hpe/cheese:/workspace \
  -v /home/nvidia/.cache/ov/hub:/var/cache/hub \
  --entrypoint /isaac-sim/python.sh \
  ${ISAAC_SIM_IMAGE:-nvcr.io/nvidia/isaac-sim:6.1.0} /workspace/sim/<script>.py
```

| detail | value |
|---|---|
| headless startup | ~40 s |
| shader cache mount | `/var/cache/hub` — skip it and every run recompiles |
| GPU access | both H100s visible, CDI |

> [!warning] The container runs as a non-root user
> Output directories must exist with `chmod 777` **before** the run, or writes fail with
> `PermissionError` after the 40 s startup.

## Headless API shape

```python
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})
# only AFTER this line can omni.* and pxr be imported
import omni.usd, omni.replicator.core as rep
from pxr import UsdGeom, UsdShade, UsdLux, Sdf, Gf
...
app.close()
```

> [!bug] The first frames fail
> `Error while stepping, renderer failed to advance to the scheduled frame` for the first
> several steps while the engine warms up. Fixed by burning 8 throwaway steps, then
> retrying each real frame up to 3 times.

> [!note] DLSS-RR is unsupported on H100 NVL
> A warning appears each run; raytracing still works, only the denoiser is affected.
