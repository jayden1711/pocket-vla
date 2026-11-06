# pocket-vla: Claude rules

Phone-run VLA driving a 2-wheel robot: ESP32-S3 BLE motor firmware + iPhone app (Core ML). Plan: PLAN.md. State: HANDOFF.md.

- **Main copy and session: the Mac** (`~/Downloads/projects/pocket-vla`). Core ML conversion, Xcode, the iPhone
  and USB flashing all need it. If GPU training ever moves to the NUC, that clone follows `~/projects/NUC_RULES.md`
  (gpu-run, results policy).
- **Public repo, Apache-2.0.** Code, firmware, configs and docs only. Benchmarks, conversion logs, phone numbers,
  videos and write-ups stay in git-ignored `results/` until the owner approves. No numbers in README/PLAN/HANDOFF.
- **Nothing goes public without the owner's go-ahead**: pushes, releases, videos.
- **Licenses:** firmware deps Apache/BSD/MIT only (ESP-IDF + NimBLE, not the LGPL Arduino core). Every model
  and source is pinned in THIRD_PARTY.md. Distilling from Llama-2-derived OmniVLA (full) is an open decision;
  ask before using it.
- **Hardware:** flashing, motor tests and anything that moves the robot only on the owner's request. No
  `.mlpackage`, checkpoints or weights in git (training/out/ is ignored).
