# Workstation operations and Git workflow

Commands in this guide assume the project workstation and repository path used
for the accepted integration work:

```text
host alias: cheesefactory
repository: /home/ubuntu/cheese-factory-challenge
branch: codex/hackathon-integration
GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB
Isaac image: nvcr.io/nvidia/isaac-sim:6.1.0
```

## Connect

With the Brev CLI:

```bash
brev login
brev shell cheesefactory
```

With the generated SSH alias:

```bash
ssh cheesefactory
```

From Windows OpenSSH, if a stale multiplexed connection produces
`getsockname failed: Not a socket`, bypass the control socket:

```powershell
ssh -S none cheesefactory
```

Then enter the repository:

```bash
cd /home/ubuntu/cheese-factory-challenge
git status --short --branch
git log -5 --oneline --decorate
```

## Preflight and status

```bash
infra/isaac-sim/factory-demo.sh preflight
infra/isaac-sim/factory-demo.sh status
nvidia-smi
docker ps
```

The normal GUI stack has three containers:

- `isim-isaac-sim-1` — Isaac Sim;
- `isim-web-viewer-1` — WebRTC browser client;
- `isim-remote-desktop-1` — Chromium/noVNC wrapper.

`isaacsim-mcp` is a separate documentation/search server. It is not the
simulation controller.

## Run the factory

Recommended deterministic presentation:

```bash
infra/isaac-sim/factory-demo.sh showcase
```

Honest trained-model path:

```bash
infra/isaac-sim/factory-demo.sh model
```

Replay, recover, inspect and stop:

```bash
infra/isaac-sim/factory-demo.sh replay
infra/isaac-sim/factory-demo.sh recover showcase
infra/isaac-sim/factory-demo.sh status
infra/isaac-sim/factory-demo.sh stop
```

Use `showcase` to debug motion and presentation. Use `model` to judge actual
autonomy. Do not silently substitute one for the other.

## Open the browser desktop

Forward the noVNC port from your local computer:

```bash
ssh -N -L 6080:127.0.0.1:6080 cheesefactory
```

Then open <http://127.0.0.1:6080/vnc.html?autoconnect=1&resize=scale>.

If local port 6080 is busy:

```bash
ssh -N -L 16080:127.0.0.1:6080 cheesefactory
```

Open <http://127.0.0.1:16080/vnc.html?autoconnect=1&resize=scale> instead.

Keep the tunnel terminal open. Only one WebRTC viewer is supported at a time.

## Logs and results

```bash
tail -f outputs/factory/gui.log
cat outputs/factory/runtime-status.json
python -m json.tool outputs/factory/results.json | less
ls -lah outputs/factory/frames
docker compose -p isim -f infra/isaac-sim/docker-compose.yml logs --tail=200
```

The most useful failure evidence is the first exception in `gui.log`, the
runtime status, the affected annotated frame, and the corresponding record in
`results.json`.

## Tests outside Isaac

Use the repository virtual environment when it exists:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q sim src tests
git diff --check
```

The pure tests intentionally avoid importing Isaac. They validate configuration,
geometry, contracts, state transitions, data splits and promotion policy.

## Bounded Isaac validation

```bash
infra/isaac-sim/run-evaluation.sh showcase 1
infra/isaac-sim/run-evaluation.sh showcase 11
infra/isaac-sim/run-evaluation.sh model 11
```

Use the one-object run only as a smoke test. Any claim about the scenario needs
the complete run and inspected evidence.

## Edit the scene layout

```bash
cp sim/factory/scene_layout.json sim/factory/scene_layout.local.json
export CHEESE_FACTORY_SCENE_LAYOUT=sim/factory/scene_layout.local.json
infra/isaac-sim/factory-demo.sh recover showcase
```

After visual experimentation, copy only accepted values into the checked-in
layout. Do not commit `scene_layout.local.json`; add it to your personal exclude
file if needed:

```bash
printf '%s\n' 'sim/factory/scene_layout.local.json' >> .git/info/exclude
```

## Git workflow

Never develop directly on the integration branch. Start from its latest remote
commit:

```bash
git fetch origin
git switch codex/hackathon-integration
git pull --ff-only origin codex/hackathon-integration
git switch -c codex/<short-feature-name>
```

Inspect before staging:

```bash
git status --short
git diff --stat
git diff
git diff --check
```

Stage only the intended files and inspect the staged result:

```bash
git add path/to/file1 path/to/file2
git diff --cached --stat
git diff --cached
```

Commit and publish:

```bash
git commit -m "feat: describe the change"
git push -u origin HEAD
```

Open a pull request from the feature branch into
`codex/hackathon-integration`. Include:

- why the change is needed;
- files and runtime modes affected;
- exact validation commands;
- screenshots for visual/physics work;
- metrics before and after;
- known limitations and rollback instructions.

After merging:

```bash
git switch codex/hackathon-integration
git pull --ff-only origin codex/hackathon-integration
git branch -d codex/<short-feature-name>
```

Do not force-push shared branches, commit generated datasets/checkpoints, or
discard unrecognized workstation changes.

## Dataset and model commands

The full data path is expensive and should not be run casually:

```bash
.venv/bin/python src/download_datasets.py
.venv/bin/python src/normalize.py
.venv/bin/python src/cutouts.py
```

Canonical-camera and Replicator workflows:

```bash
infra/isaac-sim/capture-all-domain.sh
infra/isaac-sim/replicator-smoke.sh
.venv/bin/python src/audit_camera_dataset.py
.venv/bin/python src/audit_replicator_dataset.py --help
```

Training options change over time; inspect them before launching a GPU job:

```bash
.venv/bin/python src/train.py --help
.venv/bin/python src/evaluate_checkpoint.py --help
.venv/bin/python src/check_candidate_gate.py --help
```

Record the command, Git SHA, manifest hashes, seed, checkpoint output and GPU in
the experiment report. Never overwrite the production checkpoint manually.
