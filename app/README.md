# app/

## Bench (Step 0 phone benchmark)

SwiftUI app that times the Step 0 Core ML models on the iPhone and runs the offline speech check.
Results are JSON files in the app's `Documents/results/` (Finder > iPhone > Files > VLA Bench). Copy them
to the Mac's git-ignored `results/step0/phone/`; never commit them.

What it measures, per model and compute-unit setting:
- load time, first prediction, then median / p95 / max latency and Hz over N runs (fixed synthetic inputs);
- a 10-minute sustained run with `ProcessInfo.thermalState` and `phys_footprint` logged on every prediction;
- the Step 0 pass bar: p95 over the last minute of a >= 10-min run <= 1 s and <= 40% of the chunk duration
  (chunk defaults: SmolVLA 5.0 s, student 1.6 s; editable in the app);
- offline speech: `SFSpeechRecognizer` with `requiresOnDeviceRecognition = true`; the result records whether
  any network path was up (must be off: airplane mode, Bluetooth may stay on).

Note: on the Mac, `phys_footprint` stays small with the Neural Engine because the ANE's copy of the weights
is not charged to the app process. Treat the memory column as the app's own footprint, not total model memory.

### Build

Needs full Xcode (Command Line Tools are not enough) and [XcodeGen](https://github.com/yonaskolb/XcodeGen).

```
brew install xcodegen
training/.venv/bin/python training/export/export_student.py     # writes training/out/student_*.mlpackage
cd app/Bench && xcodegen generate && open Bench.xcodeproj
```

In Xcode: pick your team under Signing & Capabilities, select the iPhone, Run. Models found in
`training/out/` are compiled into the app; others can be dropped into the app's Documents folder.

Run order for Step 0: plug in, screen brightness fixed, Low Power Mode off, phone at room temperature.
1. Each model: "Run 50 predictions" with CPU_AND_NE, ALL, CPU_AND_GPU.
2. Best setting per model: "Sustained 10 min".
3. Airplane mode on, Bluetooth on: "Record 5 s, on-device only".

### Check without a phone

The bench core also builds for macOS (proxy numbers only):

```
swiftc -O -o /tmp/bench_smoke app/Bench/Sources/ModelBench.swift app/Bench/Sources/System.swift app/Bench/MacSmoke/main.swift
/tmp/bench_smoke training/out/student_policy_fp16.mlpackage 50 CPU_AND_NE
```
