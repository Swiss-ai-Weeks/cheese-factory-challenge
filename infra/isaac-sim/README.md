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

It treats GPU, Docker, running services, MCP, model endpoint and the Python
environment as required. Missing datasets, raw renders and the ignored trained
checkpoint are reported as warnings so the development integration harness can
still be tested honestly.

The default image is `nvcr.io/nvidia/isaac-sim:6.1.0`. Override it explicitly
when reproducing an older result:

```bash
export ISAAC_SIM_IMAGE=nvcr.io/nvidia/isaac-sim:<tag>
```

## Start

```bash
cd infra/isaac-sim
cp .env.example .env
mkdir -p /home/ubuntu/docker/isaac-sim/{cache/main,cache/computecache,cache/kit,config,data,logs,pkg}
sudo chown -R 1234:1234 /home/ubuntu/docker/isaac-sim
docker compose -p isim up -d --build
docker compose -p isim ps
```

All three services should become healthy. The first Isaac Sim startup may take
several minutes while shaders and extensions are cached.

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
