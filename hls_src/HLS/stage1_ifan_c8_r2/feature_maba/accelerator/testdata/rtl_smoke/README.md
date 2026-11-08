# Curated RTL smoke fixture

This small, self-contained dataset lets a clean Linux checkout run C simulation, HLS synthesis,
and RTL co-simulation without access to the original training checkpoint or scene exports.

- 10 six-frame test transactions; the default RTL run uses the first three (frozen, zero, repeat).
- Packed 864-element FP32 weights, inputs, and software-reference outputs.
- `manifest.json` records every file's byte length and SHA-256 digest; `run.py` verifies them before a run.
- Total binary and manifest payload is under 1 MiB.

The fixture is for module-level functional verification. It does not replace the full 84-case native
regression or provide new scene-level accuracy evidence.
