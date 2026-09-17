---
tags: [pipeline, simulation]
script: sim/render_belt.py
engine: Isaac Sim 6.0.1
---

# Isaac Sim rendering

Places each [[Cutouts|cutout]] in a plate on a conveyor and renders it from several
viewpoints, so training data lives in the robot's domain.

> [!note] The belt, the plate and the cutout quad live in `sim/usd_kit.py`
> [[Sorting line demo|The sorting line]] imports the same ones, which is what makes its
> inspection station the domain the model was trained on.

```bash
docker run --rm --gpus "device=0" -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -v /home/nvidia/hpe/cheese:/workspace -v /home/nvidia/.cache/ov/hub:/var/cache/hub \
  --entrypoint /isaac-sim/python.sh ${ISAAC_SIM_IMAGE:-nvcr.io/nvidia/isaac-sim:6.1.0} \
  /workspace/sim/render_belt.py --all --views 5 --min-fill 0.35 --max-ar 3.0 \
    --elev-min 45 --elev-max 80 --shard 0/2 --seed 200 --skip-existing
```

## The scene, built in USD

| element | construction |
|---|---|
| belt | dark `UsdGeom.Cube` slab, 2 metal side rails, 26 transverse slats |
| plate | **96-segment hand-built mesh** — triangle fan + rim ring |
| piece | textured quad, alpha as `opacityThreshold` cutout |
| lighting | `UsdLux.DomeLight` + `UsdLux.SphereLight` key |
| camera | `UsdGeom.Camera`, transform = inverse of a `SetLookAt` view matrix |

> [!tip] Why the plate is a hand-built mesh
> `UsdGeom.Cylinder` tessellates to about ten facets and rendered as a visible
> **decagon** — a hard edge the model would have learned as a background cue rather
> than as cheese.

## Material — alpha cutout without MDL

`UsdPreviewSurface` fed by a `UsdUVTexture`: `rgb` → `diffuseColor`, `a` → `opacity`,
with `opacityThreshold = 0.5`. Standard USD, no OmniPBR/MDL complexity, works in RTX.

## Domain randomisation, per view

| parameter | range |
|---|---|
| piece rotation | 0-360° |
| piece offset | constrained so it never overflows the plate |
| piece size | 35-75% of the usable diameter |
| camera azimuth | 0-360° |
| camera elevation | **45-80°** |
| camera distance | 0.85-1.45 |
| focal length | 20-32 mm |
| key light | position, 12k-60k intensity, warm tint |
| dome light | 120-700 |
| plate / belt shade | randomised greys |

## Output and throughput

12355 images, 3191 pieces. ~0.5 s/image; 9,545 took ~75 min on two GPUs.

Filenames carry everything:
```
bin_hard__00097992f9de7f56__v0.png
╰──┬───╯  ╰───────┬────────╯  ╰┬╯
 label       group key       view
```

> [!bug] The sharding bug — a quarter of the data silently vanished
> The two shards were launched with different `--seed`, and that seed also drove the
> shuffle **before** sharding. Each shard sliced a different ordering: a quarter of the
> pieces rendered twice (colliding filenames, one overwriting the other) and **a quarter
> never rendered**. Expected coverage 3/4 → 1,432; observed 1,435.
>
> Fixes: shuffle seed fixed at `random.Random(1234)`; `--seed` drives only domain
> randomisation; `--skip-existing` for resumable runs; [[Render manifest]] enumerates
> files on disk rather than trusting shard JSONs; and a guard refuses to train if the
> piece count or per-piece view count is off. **Nothing in the metrics would have shown
> this.** See [[Bug log]].

## Known limits

> [!failure] The piece is flat
> A decal with a correct silhouette: no thickness, no self-shadowing, no specular of its
> own. The visible shading comes from the source photograph.

> [!note] A flat decal render ≈ a 2D homography
> Camera angle ≈ `RandomPerspective`, scene ≈ background compositing, lighting ≈
> photometric jitter. Only the cast shadow and plate specular genuinely need a renderer.
> A dataloader-side 2D pipeline would give unlimited variation at zero GPU cost — see
> [[Decision log]]. Isaac becomes indispensable once real 3D assets sit on the belt, and
> `render_belt.py` is ready: only the quad needs to become a mesh.
