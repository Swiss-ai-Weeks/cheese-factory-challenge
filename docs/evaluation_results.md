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
