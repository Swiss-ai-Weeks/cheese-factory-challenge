---
tags: [ops]
---

# Jupyter web app

The notebook lives at `/opt/nvidia/launchpad/jupyter-notebook/hpe.ipynb`, served by the
`jupyter-notebook` container on port 8888 under `/notebook/`.

- `hpe.ipynb` — English, 48 cells, with dropdown selectors
- `hpe_fr.ipynb` — French, kept for reference

## The constraint

The container mounts only `/opt/nvidia/launchpad/jupyter-notebook` (as `/workspace`) and
cannot see the project. Two workarounds:

**Packages**, installed into the container venv from the host:
```bash
docker exec jupyter-notebook /opt/venv/bin/pip install \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124
docker exec jupyter-notebook /opt/venv/bin/pip install \
    timm pillow numpy pandas scikit-learn matplotlib ipywidgets
```
`/opt/venv` runs JupyterLab, so it **is** the default `python3` kernel. Both H100s reach
the container through CDI.

**The project**, copied into the shared area:
```bash
./sync_jupyter.sh          # -> /workspace/cheese, ~3.5 GB
```
A copy, not a mount — different filesystems, so hard links are impossible. Raw `sim/out`
PNGs are excluded (3.6 GB, unused).

> [!warning] Two limitations
> - `/workspace/cheese` is a **snapshot**: re-run `sync_jupyter.sh` after retraining or
>   the notebook reads stale checkpoints
> - the `pip install` survives `docker restart` but **not** a `docker compose up` that
>   recreates the container
>
> Both disappear if the project is mounted in the compose file:
> ```yaml
>     volumes:
>       - /opt/nvidia/launchpad/jupyter-notebook:/workspace
>       - /home/nvidia/hpe/cheese:/home/nvidia/hpe/cheese
> ```
