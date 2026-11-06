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
- offline speech: `SFSpeechRecognizer` with `requiresOnDeviceRecognition = true`. The network path
  (`NWPathMonitor`) is sampled at recognition start, every 250 ms while it runs, on every path change and at
  the end; each sample (time, status, interface types, isExpensive/isConstrained) is saved in the result as
  `networkSamples`. `offlineCheck.offline` is true only if every sample is `unsatisfied`, a path was known at
  every sample, start and end samples exist and no gap exceeds 1 s; otherwise `offlineCheck.reasons` says why
  (airplane mode on, Bluetooth may stay on).

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
3. Airplane mode on, Bluetooth on, Wi-Fi off (iOS may remember Wi-Fi on in airplane mode [verify]): "Record 5 s,
   on-device only". The status line must say "offline check PASSED"; a FAILED result is kept but is not an
   offline test.

### Check without a phone

The bench core also builds for macOS (proxy numbers only):

```
swiftc -O -o /tmp/bench_smoke app/Bench/Sources/ModelBench.swift app/Bench/Sources/System.swift app/Bench/MacSmoke/main.swift
/tmp/bench_smoke training/out/student_policy_fp16.mlpackage 50 CPU_AND_NE
```

The speech test's network sampler and offline pass/fail rule also have a macOS test (synthetic series plus
~1.3 s of the real `NWPathMonitor` on the Mac; an online Mac gives a live verdict of "not offline", which is expected):

```
make app-nettest
```
