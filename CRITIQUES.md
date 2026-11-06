# CRITIQUES

Criticisms a skeptical expert would raise about this repo, and the answer to each. Every item gets exactly one
of three answers:

- **Addressed**: the work already handles it; the pointer says where.
- **Experiment**: the extra run that would handle it; whether it was run here, and where its output goes.
- **Limitation**: an honest limit, stated up front, which we do not claim past.

This file has no result numbers. Numbers live in git-ignored `results/` (local until the owner approves
publishing); each pointer below names the file. Every result file records date, git commit, command line, tool
versions, seeds and settings, and is produced by one `make` target (see [Makefile](Makefile)).

Current state, in one line: nothing has run on an iPhone; nothing has driven a robot; the student model has
never been trained on real data. No simulation is used anywhere in this repo.

## A. Phone performance

**A1. Mac numbers are not phone numbers.** Limitation. All latencies so far come from an Apple M3 Mac running the
same `.mlpackage` (`results/step0/mac_proxy/`). An M-series Mac differs from any iPhone in Neural Engine
generation, clocks, memory bandwidth, OS and thermal envelope. Mac proxy numbers will not be quoted as phone
numbers or used to claim a phone frame rate. Every Mac proxy file carries `"note": "Mac proxy only; NOT phone
numbers"` ([eval/step0/bench_mac.py](eval/step0/bench_mac.py)). The experiment that replaces them is A2.

**A2. No on-device measurement exists.** Experiment (not run: needs Xcode, signing and the owner's iPhone).
[app/Bench](app/Bench) measures load time, first prediction, median / p95 / max over N runs, and a 10-minute
sustained run on the phone; how to run it is in [app/README.md](app/README.md). Output goes to
`results/step0/phone/`. Until those files exist, no phone performance claim is made.

**A3. Which iPhone?** Limitation until A2. Neural Engine performance varies by chip generation. The phone
result records the hardware identifier (`device` in each app JSON, [ModelBench.swift](app/Bench/Sources/ModelBench.swift));
any post names the exact model and iOS version and does not generalise to other iPhones.

**A4. Phones throttle; a short burst says nothing about sustained driving.** Experiment (built, not run). The app's
10-minute sustained mode logs `thermalState` and latency on every prediction and judges the pass bar on the last
60 s only ([ModelBench.swift](app/Bench/Sources/ModelBench.swift), `runBench`). Output: `results/step0/phone/*_sustained_*.json`.

**A5. Memory numbers understate model memory.** Limitation. `phys_footprint` (app) and process RSS (Mac) do not
include the Neural Engine's copy of the weights. Stated in [app/README.md](app/README.md) and next to the field
in the Mac JSON (`peak_rss_mb_process_all_units`). Reported as app footprint, never as total model memory.

**A6. Model-only latency, not end-to-end.** Limitation for Step 0. Benchmarks time `prediction()` on fixed
synthetic tensors: no camera capture, image resize, tokenisation, chunk scheduling or BLE. End-to-end
frame-to-motor latency is a Step 5 experiment (capture timestamps are already part of the app loop design in
[PLAN.md](PLAN.md)); it is not measured yet.

**A7. Synthetic inputs.** Limitation. Inputs are seeded uniform noise (floats) and all-ones (int tokens); recorded
in each result's `settings`. Core ML shapes are static, so latency should not depend on values, but this has not
been checked against real camera frames. Experiment (cheap, after the app runs): bench on recorded frames.

**A8. Run-to-run variance, cherry-picking.** Addressed (from this commit on). `bench_mac.py` runs `--trials`
independent trials (fresh model load each, compute units interleaved within each trial so drift hits all of
them), keeps every raw latency and reports median, p5/p95, min/max, std per trial plus the spread of the
per-trial medians. Each run writes a new timestamped file; nothing is overwritten. `make bench-mac`.
The older single-trial files `results/step0/mac_proxy/<model>.json` predate this change and carry no commit or
date; they are kept unedited and superseded by `<model>_<date>.json`. The app stores every sample of a run.

**A9. One host, one toolchain.** Limitation. One Mac, one macOS, coremltools 9.0 with torch 2.15 (coremltools
warns it is tested up to torch 2.7). Versions are in every result file; the exact environment is
[training/requirements-mac.lock](training/requirements-mac.lock).

## B. The student model

**B1. The student is untested: random weights, no task performance.** Limitation. The 76M student
([training/models/student.py](training/models/student.py)) has only been exported and timed with random weights.
Its results show that it converts to Core ML and how long a forward pass takes on the Mac; they show nothing
about whether it can follow instructions or drive. No accuracy, success-rate or "works" claim is made for it.
The experiment that addresses it is training/distillation on this robot's demos and closed-loop evaluation
(PLAN Steps 3 to 5), not run.

**B2. Can this architecture learn at all?** Experiment (cheap, run here; SYNTHETIC data). `make overfit-student`
([training/tests/overfit_student.py](training/tests/overfit_student.py)) trains the student from scratch on a tiny
synthetic set in which each image appears with two instructions and two targets, so an instruction-blind model
cannot beat a stated floor. Three seeds at each of two learning rates; every seed's full loss curve is kept.
Output: `results/student/overfit_<date>.json`, with a reading in `results/student/NOTES.md`. This only tests
trainability and that the language path carries information; it is not a driving or generalisation result and
will not be presented as one. Outcome: training as configured is not reliable (most runs did not fit the
synthetic set; the tanh output head saturates early). So no claim is made that the student trains as-is.
Follow-up experiment (not run): lr warmup, no tanh during training, or a smaller head init, on the same seeds,
reporting every run.

**B3. Unit tests are shape tests.** Addressed, within scope. [training/tests/test_student.py](training/tests/test_student.py)
checks output shape and range, parameter count, that padded tokens are ignored, and that the split (text +
policy) export equals the full model. These are correctness checks of the code, not of behaviour.

**B4. Does the Core ML model compute the same thing as PyTorch?** Addressed for the student; SmolVLA see C3.
Export scripts write a `parity_vs_torch` block (max and mean abs error vs the fp32 torch output, its scale,
finiteness) for both the CPU path and the Neural Engine path into `results/step0/conversion/<name>.json`, and
keep every attempt under `results/step0/conversion/history/` (`make export-student`). Each Core ML run happens in
a fresh torch-free process ([training/export/parity_ne.py](training/export/parity_ne.py) explains why).
Limitations: one fixed seeded input, not a distribution of real inputs; random weights; and no pass/fail
tolerance is defined. Whether the fp16 error matters for control can only be judged on the trained model in
closed loop, so "matches PyTorch" is not claimed beyond what the logged errors show.

**B5. Random-weight latency may not equal trained-model latency.** Limitation. For dense fp16 weights the compute
is the same; once trained weights are quantised or pruned (PLAN Step 4) latency changes, so the model is
re-benchmarked on the phone after training. Random-weight numbers are labelled as such.

**B6. Small model, little data: generalisation.** Experiment (not run). Success rate per instruction with
denominators, including held-out objects and at least one room not in the training set (PLAN Steps 3 and 5). The
demo room is a training room; this is stated wherever the demo is shown.

**B7. Direct regression averages multimodal actions.** Limitation / ablation not run. The student regresses one
16x2 chunk (decision recorded in HANDOFF.md). If demos are multimodal (go left or right around an obstacle),
regression can average them. Ablation against a flow-matching head is possible later; not done.

**B8. The pass bar depends on the chosen chunk length.** Limitation. The Step 0 bar (p95 <= 1 s and <= 40% of the
chunk duration) gets easier with longer chunks. Chunk lengths were fixed before any phone run (student 16 steps,
SmolVLA 50 steps, both at 10 Hz) and the result files record `chunkSeconds`; latency is always reported in ms
as well, independent of the bar.

## C. Comparison with SmolVLA

**C1. Any student-vs-SmolVLA comparison must be like-for-like.** Limitation, plus a rule. The two models differ in
almost everything that sets latency: image size (224 vs 512), action chunk (16x2 vs 50x32), action head (one
forward pass vs 10 flow-matching steps), parameter count, and weights (random vs pretrained). Therefore:
- No "N times faster than SmolVLA" claim. A side-by-side table, if shown, lists all of the above per row, the
  same host, Core ML version, precision and compute unit, and the same harness (`make bench-mac`, later app/Bench).
- Each bench file records the model's input/output shapes (`io` in the Mac JSON, `settings.modelInputs` /
  `modelOutputs` in the app JSON) so a reader can check what was compared.
- No quality comparison is possible: SmolVLA is pretrained on arm data; the student is untrained.
Experiments that would make a fairer latency comparison (not run): a SmolVLA flow-step sweep
(`export_smolvla.py --steps 1..10`) to separate head cost from backbone cost; latency per second of action
horizon at matched control rate. A fair *quality* comparison needs both models trained on the same demos and
scored on the same closed-loop tasks.

**C2. SmolVLA is an arm policy.** Limitation. `lerobot/smolvla_base` predicts 50-step chunks of a 32-dim
(padded) action space pretrained on arm datasets; a 2-wheel drive head discards most of that pretraining.
Listed under open decisions in [HANDOFF.md](HANDOFF.md). SmolVLA here is a size/latency reference, not a
proposed driving policy as-is.

**C3. The conversion patches could change SmolVLA's behaviour.** Addressed, logged from this commit on. Five
conversion fixes are listed in [PLAN.md](PLAN.md). `export_smolvla.py` now writes the static-position patch
difference vs the unpatched model and Core ML vs torch parity (CPU and Neural Engine) into
`results/step0/conversion/smolvla_base_s10_l16_fp16.json` (`make export-smolvla`). Limitation: one input.

## D. Robot and firmware

**D1. The firmware has never run on hardware.** Limitation. Host unit tests pass (`make firmware-test`,
[firmware/test/test_control.c](firmware/test/test_control.c)) and it builds for ESP32-S3 (`make firmware-build`).
Watchdog, slew limit and stop behaviour are verified only in the host tests. Hardware tests happen only on the
owner's request.

**D2. "A phone VLA driving a real robot" is not shown yet.** Limitation. The claim is not made until a closed-loop
run exists (PLAN Steps 5 and 6). No robot, no closed loop and no demos exist today.

**D3. Demo videos can be cherry-picked.** Experiment (not run). Report every trial with its outcome and the
success rate with denominators; keep full attempt logs in `results/`; film the airplane-mode toggle on camera.

**D4. "Offline" must be proven, not asserted.** Addressed for speech, experiment for the demo. The speech test sets
`requiresOnDeviceRecognition` and records whether any network path was up
([SpeechTest.swift](app/Bench/Sources/SpeechTest.swift)). The full demo needs the airplane-mode-on,
Bluetooth-on state shown on camera (PLAN Step 6).

**D5. Safety: anyone nearby can connect over BLE.** Limitation. No pairing or authentication yet (PLAN Risks);
bonding/allowlist before any demo is an open decision. The safety stop lives on the ESP32 (watchdog, disconnect
stop), not on the phone.

## E. Novelty, prior work, disclosure

**E1. "First VLA on a phone."** Addressed. Not claimed; PLAN fixes the wording ("a phone VLA driving a real
robot") and names BitVLA as an earlier phone VLA. Prior work is listed in [README.md](README.md); items marked
[verify] still need the owner to check the source.

**E2. Teacher licensing.** Limitation / open decision. Full OmniVLA is derived from Llama 2 (not Apache/BSD/MIT).
Distilling from it is not done without the owner's decision ([THIRD_PARTY.md](THIRD_PARTY.md)).

**E3. Conflict of interest.** Addressed by disclosure in [README.md](README.md): the author was an NVIDIA intern
until Nov 13, 2026 and founded Quatern. Training is planned on an NVIDIA RTX GPU.

## F. Reproducibility

**F1. Can every number be regenerated with one command?** Addressed for what exists. [Makefile](Makefile) has
one target per result family; README lists them. Phone results come from app/Bench by hand (app/README.md run
order); that is a manual procedure, not one command.

**F2. Older result files lack provenance.** Limitation. Files written before this commit have tool versions but no
date, git commit or seeds. They are kept as-is; rerunning the `make` target produces files with full provenance.

**F3. Failed conversion attempts.** Partly addressed. From this commit on, every conversion attempt (including
failures, with traceback) is kept under `results/step0/conversion/history/`. The failed SmolVLA attempts that led
to the five fixes were overwritten before this change; only their descriptions in PLAN.md remain.

**F4. Unpinned environment.** Addressed. Model revisions are pinned in THIRD_PARTY.md and recorded in the
conversion logs; the Python environment is frozen in [training/requirements-mac.lock](training/requirements-mac.lock);
ESP-IDF is pinned by container tag.
