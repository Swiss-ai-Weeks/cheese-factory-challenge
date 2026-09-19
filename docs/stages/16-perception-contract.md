# Stage 16 — authoritative perception contract

## Problem

Model mode combined a fine-type classifier with a target-domain direct-bin classifier.
The direct-bin output controlled the arm, while the unrelated fine-type output was
displayed. A response could therefore say `blue_mould_cheese` and command `bin_hard`,
or display a cheese type while silently rejecting it, without recording the conflict.

## Decision policy

Contract version 2 names the policy `route_authoritative_fail_closed_v1` and preserves
both raw outputs.

| Route result | Fine type | Actuator decision |
|---|---|---|
| below threshold | any | `uncertain`; no pick |
| `empty` / `not_cheese` | any | route rejection is authoritative; no pick; disagreement retained |
| cheese bin | type implies same bin | `ok`; route is actionable |
| cheese bin | type implies another/no bin | `uncertain`; safe hold; no pick |

Every response now carries `route_label`, `type_implied_bin`, `decision_policy`,
`agreement`, and `decision_reason`. The Isaac client validates these fields and rejects
an actionable contradiction. Per-item evaluation records preserve them, and the HUD
shows `SAFE HOLD · MODEL DISAGREEMENT` when the models conflict.

The launcher also detects an old sorter contract. It restarts a repository-owned stale
service and refuses an incompatible external service instead of feeding an ambiguous
decision to the controller.

## Verification

- all five bins have matching route/type policy tests
- `empty`, `not_cheese`, `uncertain`, cross-bin conflict, unknown route, and malicious
  actionable-conflict responses have fail-safe tests
- 58 repository tests passed; Python compilation, shell syntax, and diff checks passed
- a live one-object trained-model run started contract version 2, predicted
  `hard_cheese` / `bin_hard` with raw route `bin_hard`, recorded `models_agree`, and
  completed the physical pick/place successfully
- runtime: `9a256cd65cda-20260919T091934Z-2627672-24653`; inference confidence 0.9143,
  measured request latency 19.86 ms, and 21.77 s physical cycle time

The live smoke is one simulated item and is integration evidence, not a new classifier
accuracy claim. Existing held-out evaluation remains the source for model-quality
claims.
