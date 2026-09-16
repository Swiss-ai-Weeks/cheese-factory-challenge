---
tags: [integration, simulation, demo]
---

# Sorting line demo

A full sorting line in Isaac Sim, **driven by the model**: a piece arrives on the belt,
stops under the inspection camera, `sim_type13` says what it is, and the diverter for
that class lifts and pushes it onto its own lane. One lane per output class.

```
                    ┌── BIN_HARD      ┌── BIN_SOFT      ┌── BIN_BLUE
─── [camera] ───────┴────┬────────────┴────┬────────────┴────┬───────
                         └── BIN_SEMI_HARD └── BIN_FRESH     └── REJECT
```

> [!important] Nothing is scripted
> The lane a piece takes is whatever `CheeseSorter` returns for the frame the camera
> just took. Feed the model a different piece and the line sends it somewhere else —
> including into `REJECT`, which is where the three non-`ok` statuses go
> (see [[Output contract]]).

## The pieces

| file | runs | role |
|---|---|---|
| `sim/sort_server.py` | host, `.venv` | the model behind a 40-line HTTP service |
| `sim/sorting_line.py` | Isaac Sim container | the scene, the animation, the rendering |
| `sim/pick_demo_pieces.py` | host, `.venv` | picks the cheeses that ride the belt |
| `sim/make_labels.py` | host, `.venv` | the bin signs (Isaac Sim cannot draw text) |
| `sim/make_video.py` | host, `.venv` | HUD overlay + H.264 |
| `sim/usd_kit.py` | container | belt, plate, cutout quad — shared with `render_belt.py` |

> [!note] Why the model lives outside the container
> The `isaac-sim:6.0.1` image has neither torch nor timm, and adding them to it would
> fork a 32 GB image for one dependency. The scene POSTs the camera frame to the host
> instead. That split is also the one the real line has: camera and PLC on one side,
> perception on the other.

## The inspection station is the training domain

The model was trained on renders of a plate on a belt, camera 0.85–1.45 m away,
elevation 45–80°, focal 20–32 mm (see [[Isaac Sim rendering]]). The inspection station
reuses **the same geometry** — same `Tapis`, same plate, same cutout quad — and parks
the piece at the exact origin those renders used, with the camera at 1.15 m, 62°,
26 mm. Everything else in the scene (diverters, lanes, bins) is kept out of that frame.

That is why the rest of the line has no side rails past `x = 0.7`: the training belt
has them, so the **inspection section keeps them**, and the sorting section drops them
so the lanes can meet the main belt.

> [!warning] It is still not a validation
> Eleven pieces chosen for a demo measure nothing. `pick_demo_pieces.py` keeps cheeses
> the model already handles on training-domain renders, so the video shows the line
> working, not the model's error rate. The numbers that count are in
> [[Model comparison]], and what they do not cover is in
> [[What the numbers do not measure]].

## Running it

```bash
.venv/bin/python sim/make_labels.py
.venv/bin/python sim/pick_demo_pieces.py --per-bin 2 --negatives 1
.venv/bin/python sim/sort_server.py &            # port 8765

mkdir -p sim/render/{frames,insp} && chmod 777 sim/render sim/render/*
docker run --rm --gpus '"device=1"' --network host -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -v $PWD:/workspace -v ~/.cache/ov/hub:/var/cache/hub \
  --entrypoint /isaac-sim/python.sh nvcr.io/nvidia/isaac-sim:6.0.1 \
  /workspace/sim/sorting_line.py --out /workspace/sim/render

.venv/bin/python sim/make_video.py
```

`--preview` renders one overview frame, one inspection frame and one classification,
then exits — the fast loop for reframing the camera without re-rendering a minute of
video.

> [!tip] `--network host` is not optional
> Without it the container cannot reach the model on `127.0.0.1:8765`, and every piece
> falls through to `REJECT` with `status = "erreur"`.

## What the video shows

![[sorting-line.png]]

`sim/cheese_sorting.mp4` — 40 s, 1280×720, 24 fps. Eleven pieces (two per bin plus one
`not_cheese`), 839 rendered frames, 35 s of line time.

| on the belt | the model says | lane | confidence |
|---|---|---|---|
| blue mould cheese | blue mould cheese | `bin_blue` | 0.918 |
| cream cheese | cream cheese | `bin_fresh` | 0.942 |
| blue mould cheese | blue mould cheese | `bin_blue` | 0.905 |
| cottage cheese | cottage cheese | `bin_fresh` | 0.935 |
| emmental cheese | emmental cheese | `bin_hard` | 0.926 |
| raclette cheese | raclette cheese | `bin_semi_hard` | 0.921 |
| hard cheese | hard cheese | `bin_hard` | 0.920 |
| raclette cheese | raclette cheese | `bin_semi_hard` | 0.921 |
| processed cheese | processed cheese | `bin_soft` | 0.933 |
| taboulé with couscous | **not cheese** | `reject` | 0.868 |
| soft cheese | soft cheese | `bin_soft` | 0.923 |

Eleven out of eleven landed in the expected bin, the couscous salad included — it never
reaches a cheese bin because `status = "not_cheese"` carries `bin = None`, and the line
sends every non-`ok` status to `REJECT`.

| | |
|---|---|
| model latency | 29–48 ms per frame |
| round trip, scene → host → scene | 50–75 ms |
| rendering | 0.71 s per frame (1280×720 + 768² inspection, 6 subframes) |
| whole render | 839 frames in ~10 min on one H100 NVL |

The HUD is drawn afterwards from `timeline.jsonl`, which carries one record per frame —
so the overlay can be redesigned without re-rendering a single frame.

## Design notes

- **The belt stops for the camera.** The conveyor advances until the next piece is
  exactly at the station, holds `--dwell` seconds, then resumes. An indexing conveyor
  is what a real inspection station does, and it also gives the viewer time to read the
  decision.
- **Pieces detach onto their lane** when they reach their diverter, then run on their
  own clock — a lane keeps moving while the main belt is stopped for the next piece.
- **The diverter blade is parallel to its lane**, so a piece sliding along it is pushed
  out. It rises with a 0.1 s time constant when the model picks that lane and drops
  back afterwards, which is what makes the decision visible in the wide shot.
- **`group` discipline does not apply here** — these pieces come from the training
  manifest on purpose. See [[Splits and data leakage]] for when it does.

## Related

- [[Output contract]] — the `status`/`bin` contract the line obeys
- [[Inference API]] — `CheeseSorter`, the class the server wraps
- [[Isaac Sim rendering]] — the renders the inspection station imitates
- [[Isaac Sim setup]] — the container, the shader cache, the 40 s startup
