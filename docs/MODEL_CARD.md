# Model card — Cheese Factory perception

## Model package

The production decision is intentionally split across two ConvNeXt-Base classifiers at
384 px. `sim_type13` supplies an operator-facing fine type across 11 cheese types plus
`empty` and `not_cheese`. `sim_bin_adapt_v2` supplies the authoritative five-bin route
plus the two reject classes. The immutable artifact hashes and training provenance are
in `config/runtime-provenance.json`; weights are not in Git because each file is about
350 MB and the repository does not use Git LFS.

## Intended use

- Research and hackathon demonstration of a simulated perception-to-action loop.
- Classification of the project's normalized photographic/Isaac-rendered cheese domain.
- Fail-closed route suggestions to the simulated factory controller.

Not intended for food-safety decisions, quality inspection, autonomous deployment near
people, commercial variety identification, or operation of a physical robot without a
separate safety-rated system and real-domain validation.

## Data and training

- Architecture: `convnext_base.fb_in22k_ft_in1k_384`.
- Training seed: 42; 20 epochs; 384 px; balanced sampler.
- Fine-type manifest: `data/processed/manifest_sim.csv`.
- Routing manifest: `data/processed/manifest_factory_adapt.csv`.
- Sources include Food Recognition 2022 plus normalized public cheese datasets. Object
  groups, rather than rendered images, own the train/validation/test split to prevent
  alternate views of one source object leaking across splits.
- Factory adaptation images are synthetic camera renders. They are not photographs from
  a physical production line.

## Accepted metrics

The generated `docs/evaluation_results.md` is authoritative. At the accepted artifact:

- fine type: 429/545 held-out top-1 (78.7%), macro-F1 0.4365;
- direct-bin routing: 656/845 held-out top-1 (77.6%), macro-F1 0.5509;
- deterministic 11-object Isaac model run: 3/10 correct routes, 4/10 pick coverage,
  4/4 successful manipulations given an attempt, and 4/11 full-loop outcomes.

The report includes 95% Wilson intervals and input hashes. Showcase results are excluded
from model quality because their routes come from scenario ground truth.

## Limitations and risk controls

Rare fresh, soft, semi-hard and blue classes have small support and weak recall. Domain
shift between photographic cutouts, synthetic renders and a real factory is unmeasured.
The live 11-object scenario is far too small for a reliability claim. Confidence is not
calibrated as a probability of safety.

The controller therefore rejects uncertain, foreign, stale, malformed and mismatched
decisions; it never substitutes showcase labels in model mode. Timing and correlation
metadata bind each action to one camera frame and item. These are software safeguards in
a simulation, not safety certification.

## Evaluation and update policy

Any candidate must pass the frozen validation gate in `config/p14_candidate_gate.json`.
The test split and final Isaac scenario remain untouched until a candidate passes every
aggregate, rare-bin, coverage and foreign-object threshold. Rejected experiments are
documented; production stays on the hashes in `runtime-provenance.json`.
