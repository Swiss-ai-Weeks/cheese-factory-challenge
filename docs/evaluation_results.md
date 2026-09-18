# Factory evaluation results

Machine-readable run results are generated at `outputs/factory/results.json`
and intentionally excluded from Git. They must be reported separately from the
offline classifier metrics in `runs/`.

The deterministic scenario contains ten cheese objects (two destinations per
bin), one foreign object, and one empty-belt interval. The report measures:

- foreground detection success;
- fine-type classification correctness;
- physical pick success;
- correct-bin placement;
- complete camera-to-placement success;
- mean cycle time.

## Verified run

The deterministic scenario completed on 2026-09-16 with Isaac Sim
`6.1.0-rc.26`. These are integration results from the explicitly labeled
rendered-pixel development classifier, not trained-model accuracy:

| Metric | Result |
|---|---:|
| detection success | 11/11 (100%) |
| fine-type classification | 11/11 (100%) |
| physical pick success | 10/10 (100%) |
| correct-bin placement | 10/10 (100%) |
| safe foreign-object rejection | 1/1 (100%) |
| empty-frame handling | 1/1 (100%) |
| end-to-end object outcome | 11/11 (100%) |
| mean cycle time | 14.70 s |

All ten cheese proxies reached their classifier-selected bins. The foreign
object was detected and rejected without a pick, and the empty interval caused
no action. The complete machine-readable report and annotated evidence are in
the ignored `outputs/factory/` directory on the evaluation workstation.

Runs made with the development classifier carry
`development_classifier: true`; they validate the camera-to-controller
integration only and must never be presented as trained-model accuracy.

## Judge-facing deterministic showcase

After the canonical cell redesign, the old pixel-color development classifier became
lighting-sensitive. It is retained for diagnostic compatibility, but the reliable live
demonstration now has a separate `showcase` mode. The mode uses the rendered camera for
foreground detection and calibrated pick localization, then supplies the route from the
known scripted scenario so judges can observe every physical destination.

The 11-object acceptance run completed on 2026-09-18:

| Metric | Result |
|---|---:|
| detection success | 11/11 (100%) |
| physical cheese pick/place | 10/10 (100%) |
| correct scripted destination | 10/10 (100%) |
| scripted foreign-object rejection | 1/1 (100%) |
| empty-frame handling | 1/1 (100%) |
| end-to-end scripted outcome | 11/11 (100%) |
| mean cycle time | 15.61 s |

Every artifact carries `classifier_mode: showcase`,
`showcase_ground_truth_routing: true`, and `trained_model: false`. These results prove
the camera/localization/manipulation integration and presentation sequence; they are not
classifier accuracy. Trained-model evidence remains in the next section.

## Trained-model run

The production path was evaluated on 2026-09-18 with the host-side
fine-type/direct-bin service and held-out photographic cutouts. Unlike the
development result above, these numbers include actual model errors:

| Metric | Result |
|---|---:|
| detection success | 11/11 (100%) |
| fine-type classification | 5/11 (45.5%) |
| successful picks among attempts | 6/6 (100%) |
| cheese pick coverage | 6/10 (60%) |
| correct-bin placement | 3/10 (30%) |
| safe foreign-object rejection | 1/1 (100%) |
| empty-frame handling | 1/1 (100%) |
| end-to-end object outcome | 4/11 (36.4%) |
| mean cycle time | 12.80 s |

The robot-control path was reliable when actuated; perception of rare cheese
families is the limiting factor. See
[`stages/05-production-perception.md`](stages/05-production-perception.md) for
the leak-free adaptation protocol and offline metrics.

## Stage 8 validation-only result

Stage 8 evaluated controlled rare-bin training changes against the immutable
300-image target-camera validation slice. None passed the predeclared yield,
coverage, and foreign-object safety gate. The accepted checkpoint remains
`runs/sim_bin_adapt_v2/best.pt`; its validation correct-and-confident rate is
0.6967, macro-F1 is 0.4010, coverage is 0.8867, and foreign safe rejection is
0.9425 at threshold 0.55.

Because there was no validation winner, Stage 8 did not inspect the target test
split and did not rerun either the one-object smoke or the 11-object Isaac
scenario. The trained-model integration results above are therefore unchanged,
not superseded. See
[`stages/08-rare-bin-perception.md`](stages/08-rare-bin-perception.md) for the
complete experiment table, selection criterion, per-class limitations, and
reproduction commands.

## Stage 9 targeted-data result

Stage 9 added a fail-closed capture filter for split and bin selection, then
tested two validation-only data changes. Neither candidate passed the unchanged
Stage 8 gate, so production still uses `runs/sim_bin_adapt_v2/best.pt`.

The additional-view candidate scored 0.6433 correct-and-confident, 0.8333
coverage, and 0.8736 foreign safe rejection. A second candidate added 36
training-only, model-curated blue-cheese photographs to the original three-view
baseline; it scored 0.6633 correct-and-confident, 0.8300 coverage, and 0.9253
foreign safe rejection. Both left blue recall at 0/3.

The locked target test split and deterministic Isaac scenario were not run.
The production integration results above remain unchanged. See
[`stages/09-targeted-data.md`](stages/09-targeted-data.md) for the capture
integrity evidence, experiment details, and diagnosis.
