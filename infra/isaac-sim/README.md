# Isaac Sim remote GUI

This stack runs NVIDIA Isaac Sim 6.1 headlessly and exposes its GUI through a
browser desktop. The WebRTC connection stays entirely on the GPU machine;
noVNC carries the resulting desktop over one HTTP/WebSocket port. This avoids
the UDP and high-port restrictions commonly encountered on SSH, Brev, VPN, and
corporate networks.

## Requirements

- Linux host with a supported NVIDIA RTX GPU and driver
- Docker with the NVIDIA Container Toolkit
- Access to `nvcr.io/nvidia/isaac-sim:6.1.0`

## Environment check

Run the non-destructive preflight before starting a demo:

```bash
infra/isaac-sim/check-environment.sh
```

It validates the GPU, Docker daemon and Compose plugin, NVIDIA container runtime,
Git checkout, pinned image and cache location. It does not require services to be
running. Documentation MCP and H200/Qwen health are informational because neither
is part of the runtime. `check-environment.sh model` additionally requires the
Python environment and verifies both ignored checkpoints against committed hashes.

The default image is Isaac Sim 6.1.0 pinned by registry digest in
`config/runtime-provenance.json`. Override it explicitly only when intentionally
testing another build:

```bash
export ISAAC_SIM_IMAGE=nvcr.io/nvidia/isaac-sim:<tag>
```

## Start

```bash
infra/isaac-sim/factory-demo.sh launch showcase
```

The command performs preflight, safely prepares persistent cache paths, launches all
three services and verifies the live runtime identity. The first Isaac Sim startup
may take several minutes while the image, shaders and extensions are cached.

Open `http://127.0.0.1:6080` on the host. For Brev, publish the desktop as an
authenticated HTTPS endpoint:

```bash
brev ports create <instance-name> 6080 --protocol http
```

Open the HTTPS endpoint printed by Brev. It redirects to noVNC and connects
automatically. An SSH-only alternative is to run this on your local computer:

```bash
ssh -N -L 6080:127.0.0.1:6080 <ssh-host>
```

Then open `http://localhost:6080` locally.

## Services and ports

| Service | Port | Exposure |
|---|---:|---|
| Isaac Sim signaling | 49100/TCP | server-local |
| Isaac Sim media | 47998/UDP | server-local |
| NVIDIA web viewer | 8210/TCP | server-local |
| noVNC desktop | 6080/TCP | publish through authenticated HTTPS |

Isaac Sim supports one WebRTC viewer at a time. If a forcibly terminated viewer
leaves the stream busy, restart the stack together:

```bash
docker compose -p isim stop remote-desktop web-viewer isaac-sim
docker compose -p isim up -d
```

The cache and user data live under `ISAAC_SIM_DATA`; recreating the containers
does not remove them.

## Factory demo launchers

The supported operator entry point is:

```bash
infra/isaac-sim/factory-demo.sh help
infra/isaac-sim/factory-demo.sh launch showcase
```

It exposes only allowlisted actions and uses the canonical runtime launcher for
start, replay and recovery. See [`docs/demo_runbook.md`](../../docs/demo_runbook.md).

Use the repository launchers instead of invoking the evaluation container next
to the persistent stream:

```bash
# browser-streamed trained-model demo; starts host perception when needed
infra/isaac-sim/run-gui.sh model

# judge-facing deterministic motion showcase; camera detection/localization,
# scripted ground-truth routing, visibly labelled as non-model evidence
infra/isaac-sim/run-gui.sh showcase

# bounded disposable evaluation; restores the stream afterward
infra/isaac-sim/run-evaluation.sh model 11

# stop the stream and only the sorter process started by this repository
infra/isaac-sim/stop-gui.sh
```

The H200/Qwen endpoint is optional and is not used by the runtime factory.
