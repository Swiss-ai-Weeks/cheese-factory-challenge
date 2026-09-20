# Stage 20 — camera-to-control timing

P1.5 closes the asynchronous safety gap between a camera observation and robot motion.
Every sortable frame now has one immutable correlation envelope:

- object ID and monotonically increasing observation sequence;
- UUID request ID;
- capture time in epoch and monotonic clocks;
- SHA-256 of the exact crop sent for inference.

The model service requires those values as HTTP headers, records when it received the
request and made the decision, and echoes the complete envelope. The factory validates
the result after inference and again immediately before starting the controller. The
second check is important: a decision can be valid when received and expire while the
runtime is preparing to move.

## Fail-closed behavior

`perception.decision_max_age_s` is the single decision deadline and is five seconds in
the canonical configuration. The client also caps its network timeout to that deadline.
The arm remains inhibited while the state machine is `CLASSIFYING`. Missing metadata,
malformed values, a wrong object/sequence/request/frame, inconsistent timestamps,
future timestamps, or an expired response transitions to `RECOVERY`; no arm target is
issued. The runtime records a structured perception fault, annotates the evidence frame
and moves the item to the simulated reject/quarantine path.

The runtime status remains healthy during a recoverable object-level fault. Fault and
deadline details are exposed in `runtime-status.json`, `results.json`, annotated frames
and the live HUD. Aggregate output adds `perception_faults`,
`mean_decision_age_ms` and `max_authorization_age_ms`.

## Verification

The isolated feature commit was validated on the RTX PRO 6000 workstation before
integration:

- 83 unit tests passed, including delayed, missing, malformed, mismatched and
  future-dated inference responses;
- Python compilation, shell syntax and `git diff --check` passed;
- a one-object deterministic showcase run completed with zero perception faults,
  0.475 ms decision age and 70.204 ms authorization age;
- a one-object trained-model run completed through the HTTP service with zero
  perception faults, 19.691 ms reported inference latency, 57.103 ms decision age and
  125.118 ms authorization age;
- the trained-model record contains the same item ID, sequence, request UUID, frame hash
  and ordered timestamps from observation through completed action;
- the restored WebRTC view was inspected and visibly showed `Timing: awaiting camera
  observation` and `Safety: ARM INHIBITED` before a decision.

The two one-object runs are contract and integration smoke tests, not classifier-quality
benchmarks. Showcase timing is local scripted timing, and the trained-model sample size
is one. The timing envelope protects against accidental stale or mismatched local
responses; it does not cryptographically authenticate the localhost service.
