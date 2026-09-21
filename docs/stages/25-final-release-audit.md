# P2.4 — Final release and branch audit

## Result

The final automated release gate passed all 19 checks on integration commit
`cdcc2bbcfae48d02d205f019141b13d8f12b7958`. The complete machine-readable result is
[`p24-final-audit.json`](../evidence/p24-final-audit.json).

## Branch integrity

- `work/perception` remains unchanged at
  `fe70eab58185881190fc3ddc257f0933159384ed` and is an ancestor of the integration.
- `work/isaac-factory` remains unchanged at
  `0f144ae65c23b13e3b92e0cf29d61ee71680122b` and is an ancestor of the integration.
- The audited local and remote integration SHAs matched exactly.
- No tracked path matched raw/processed datasets, local outputs, `.venv`, PyTorch
  checkpoints or ONNX exports. The largest tracked blob was the pre-existing 47.3 MB
  demonstration video, below GitHub's 100 MB single-file limit.

## Regression gate

- 111 tests passed.
- Python compilation passed for `src`, `sim` and `tests`.
- Every top-level Isaac infrastructure shell script passed `bash -n`.
- `git diff --check` passed and the workstation tree was clean after generated P1.4
  candidate directories were explicitly ignored.
- Judge guide, architecture, model card and evaluation report were present.
- The continuous demo video and visually reviewed preview matched their accepted
  SHA-256 hashes.

## Live gate

The canonical showcase reported the exact audited commit and a valid running phase. All
three services—Isaac Sim, WebRTC viewer and remote desktop—were `running/healthy`; the
noVNC endpoint returned HTTP 200. The digest-pinned Isaac 6.1 container was in use.

The preview and continuous capture were visually inspected during P2.3. They show the
factory overview, HUD and physical arm progression without relying on logs alone.

## Final evidence boundary

“Hackathon-ready” means the simulated system is reproducible, understandable, live,
audited and honestly evaluated. It does not mean the classifier is production-grade,
the emergency control is safety-rated, or the system has been validated on a physical
factory line. Model mode currently completes 4/11 deterministic full-loop outcomes;
showcase mode completes 11/11 by using explicitly labelled scripted routes.
