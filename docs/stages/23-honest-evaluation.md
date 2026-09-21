# P2.2 — Honest end-to-end evaluation

## Outcome

The project now publishes four distinct evidence layers instead of combining them into
one flattering accuracy number:

1. held-out fine-type classification;
2. held-out direct-bin routing;
3. manipulation success conditional on an attempted pick;
4. complete camera-to-destination outcomes in the deterministic Isaac scenario.

`src/build_evaluation_report.py` regenerates both `docs/evaluation_results.md` and
`docs/evidence/p22-evaluation-summary.json`. The report derives every live count from
per-item records and every offline aggregate from committed result JSON. It also stores
SHA-256 hashes of each input and both production checkpoints.

## Reproducibility and uncertainty

Runtime evidence schema 2 records the source commit, runtime ID, deterministic seed,
scenario, factory-config hash, object limit and digest-pinned Isaac image. Both fresh
11-item runs were executed from `4bc39ee3e7fe766f636e9e7321f6010cf167497a`.

Every binomial proportion includes a two-sided 95% Wilson interval. These intervals are
wide for the 11-item integration scenario and intentionally expose how little can be
inferred from a perfect showcase. No interval is reported for legacy macro-F1 because
the committed training artifacts contain aggregate class reports rather than the
per-sample predictions needed for defensible resampling.

## Accepted findings

- Fine-type held-out test: 429/545 top-1 (78.7%), macro-F1 0.4365.
- Direct-bin held-out test: 656/845 top-1 (77.6%), macro-F1 0.5509.
- Trained-model Isaac run: 4/11 full-loop success; 3/10 correct route decisions;
  4/10 cheese pick coverage; 4/4 successful manipulations given an attempt.
- Scripted showcase: 11/11 full-loop outcomes and 10/10 manipulations, explicitly
  labelled as scripted routing rather than model accuracy.

This is evidence of a functioning simulated perception-to-action loop. It is not a
claim of real-world accuracy, industrial reliability or safety certification.

## Verification

The evidence-producing commit passed shell syntax, Python compilation and the complete
test suite before both bounded runs. The public Markdown and summary JSON were then
regenerated from the saved raw artifacts without manually entering metric values.
