# Cheese factory browser export

**[Open the hosted replay](https://adiy.ch/cheese)** ·
**[Isaac Web Exporter source](https://github.com/gr9aditya/isaac-web-exporter)** ·
**[Download export.zip](../export.zip)**

## What the exporter makes possible

Isaac Web Exporter packages supported Isaac Sim geometry and recorded motion
for interactive browser playback. The recipient can review a run without
installing Isaac Sim or connecting to a running simulation workstation. Unlike
a fixed-angle video, the browser experience supports camera exploration, object
inspection, timeline seeking and guided captions. Static HTTP hosting is enough
to serve the exported package.

Simulation and capture still happen in Isaac. Playback requires a browser with
WebGL support and enough graphics resources for the scene, but not an RTX GPU,
Docker, Isaac Sim or an inference service. The exporter is a separate project
by Aditya Deshpande, not an NVIDIA product or a built-in Isaac WebGL runtime.

## Included sample

The root-level `export.zip` is an unchanged copy of the supplied exporter
artifact `sample-exports/cheese-factory/package.zip`, renamed for this repository.
It contains the full-visual cheese-factory recording: factory scene, robot
geometry, embedded imagery, recorded motion and guided run outcomes.

| Property | Included sample |
| --- | --- |
| Mode | Recorded playback |
| Source application | Isaac Sim 6.1.0 |
| Duration | 72.783 seconds |
| Scenario | 11-item camera/model run |
| Guided chapters | 12: preparation plus one for each item |
| Original end-to-end outcomes | 4/11 successful, with other outcomes preserved |
| ZIP size | 8,858,099 bytes |
| Archive entries | 19 |

The displayed classifications and decisions are recorded outcomes, not fresh
browser inference. In particular, seeking or restarting the replay does not
rerun the classifier, change a decision or test a new robot trajectory. Do not
present this model-run sample as the separate 11/11 scripted showcase.

## Run locally

Download `export.zip` using GitHub's download/raw action, or obtain it from a
clone of this repository. With Python 3 installed, run these commands from the
directory containing the archive:

```bash
python -m zipfile -e export.zip export
python -m http.server 8000 --bind 127.0.0.1 --directory export
```

On Windows, use `py -3` instead of `python` if that is your Python launcher.
Open <http://127.0.0.1:8000/> and press Start. Leave the terminal running during
playback and press Ctrl+C when finished. Use a different free port if 8000 is
already occupied. The extracted `/export/` directory is ignored by this repo.

Serve the files over HTTP rather than double-clicking `index.html`: browser
restrictions on local `file://` resource loading can prevent the player working.
For deployment, upload the extracted directory to a static web host while
preserving its relative file paths. The hosted example is
<https://adiy.ch/cheese>.

## Player controls

- Start, Pause and Restart control recorded playback.
- The timeline and frame controls seek through the captured run.
- Drag to orbit, right-drag to pan, and scroll to zoom.
- Camera views and object selection help inspect the cell from different angles.
- Guided captions explain the item, prediction, destination and observed outcome.

## Package contents and boundaries

- `scene.glb`: exported geometry, materials, embedded images and animation.
- `index.html` and `assets/`: the bundled browser player.
- `manifest.json`: timing, source application and asset integrity metadata.
- `scene-map.json`: object identities and scene metadata.
- `experience.json`: guided chapters and presentation context.
- `compatibility-report.json`: capture and conversion findings.
- `schemas/`, `customization/` and `THIRD_PARTY_LICENSES/`: format definitions,
  optional customization examples and bundled dependency notices.

The package does not include the source project, model checkpoints, raw camera
frames or source USD. It does not execute live Isaac physics, Python controllers,
ROS, sensors or model inference. Export support is bounded by the exporter and
the source scene; arbitrary materials, deformables or project behavior should
not be assumed to transfer. Review the compatibility report for this sample
and the exporter repository for the supported scope.

## Integrity and provenance

The archive's SHA-256 is:

```text
901918691a9e452e930620c6593eb0f5cd341c82e856b2a487221579a6590fb7
```

Check it on Linux with `sha256sum export.zip`, or on PowerShell with:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath .\export.zip
```

The bundled manifest identifies `scene.glb` with SHA-256:

```text
4dc7507a4c99beaf75b1ef22e3150ba57cc26b1fe0ccdc74d5524816b08ab88d
```

The supplied exporter sample documentation records a passing v1 package check
and a browser check covering animation, transport controls, camera navigation,
guided chapters and local-only resource loading. Those checks validate the
sample's playback, not classifier quality or physical safety. The factory's
model evaluation remains documented in [evaluation_results.md](evaluation_results.md).

The project owner supplied this archive for publication on 24 September 2026.
Preserve bundled dependency notices. Inclusion here does not grant new rights
to third-party source assets; review relevant asset terms before reusing or
redistributing them independently.
