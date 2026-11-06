# CRITIQUES

Criticisms a skeptical expert would raise about this repo, and the answer to each. Every item gets exactly one
of three answers:

- **Addressed**: the work already handles it; the pointer says where.
- **Experiment**: the extra run that would handle it; whether it was run here, and where its output goes.
- **Limitation**: an honest limit, stated up front, which we do not claim past.

This file has no result numbers. Numbers live in git-ignored `results/` (local until the owner approves
publishing); each pointer below names the file. Every result file written by the current scripts records date,
git commit, command line, tool versions, seeds and settings, and is produced by one `make` target (see
[Makefile](Makefile)); older files without this are listed under F2.

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

**A8b. Background load on the Mac contaminates timings.** Addressed in the harness from this commit on; earlier
files limited. Overlapping benches, exports or training share the Neural Engine and CPU. `bench_mac.py` now
refuses to start while one of this repo's own Python jobs is running (`bench_mac.py`, the export scripts,
`parity_ne.py`, `overfit_student.py`; matched by name in `ps`), unless `--allow-concurrent`, which leaves
a flagged record. It does not detect other heavy work (the MacSmoke bench binary, Xcode builds, other apps);
for those the load average it writes into `provenance` and after every trial is the only trace. Files written
before this change do not have these fields. `results/step0/mac_proxy/NOTES.md` lists the files known to have
overlapped another bench: the SmolVLA multi-trial file and, per the reviewer pass below, every student
multi-trial file too. They are kept but not used.

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
synthetic set; per-run results in the files above). So no claim is made that the student trains as-is.
Saturation of the tanh output head was measured in the reviewer pass (2 of the 6 runs, first 40 steps,
`results/student/tanh_probe_<date>.json`, reading in `results/student/NOTES.md`): in the failed run probed,
every pre-tanh output saturates within the first few steps after a large first-step gradient (no clipping,
no warmup) and the gradient then vanishes; the run that fit saturated only partly. The probe also reproduces
the logged loss curves of those two runs with committed code. Follow-up experiment (not run): lr warmup,
gradient clipping, no tanh during training, or a smaller head init, on the same seeds, reporting every run.

**B3. Unit tests are shape tests.** Addressed, within scope. [training/tests/test_student.py](training/tests/test_student.py)
checks output shape and range, parameter count, that padded tokens are ignored, and that the split torch
modules (text encoder, then policy) give the same output as the full torch model. It does not test the
exported Core ML packages; those are covered only by the parity check in B4. These are correctness checks of
the code, not of behaviour.

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
- Parameter counts are per row and split by what runs per frame: the student's total includes a text
  embedding table and text encoder that run once per instruction, so its per-frame (policy) count is smaller
  than its total (`params` in the conversion logs).
- Status: no clean latency file exists for either model. The student's multi-trial files were written before
  the overlap guard and overlapped each other (an n 20 and an n 50 series ran at the same time) and, for one
  file, the SmolVLA bench; SmolVLA has only the older single-trial file (fewer predictions, no provenance) and a
  multi-trial attempt that overlapped another bench (see `results/step0/mac_proxy/NOTES.md`). The clean run
  is `make bench-mac N=50 TRIALS=5` (all four models), one model at a time, on an otherwise idle Mac. Until it
  exists, no side-by-side Mac table is shown.
Experiments that would make a fairer latency comparison (not run): a SmolVLA flow-step sweep
(`export_smolvla.py --steps 1..10`) to separate head cost from backbone cost; latency per second of action
horizon at matched control rate. A fair *quality* comparison needs both models trained on the same demos and
scored on the same closed-loop tasks.

**C2. SmolVLA is an arm policy.** Limitation. `lerobot/smolvla_base` predicts 50-step chunks of a 32-dim
(padded) action space pretrained on arm datasets; a 2-wheel drive head discards most of that pretraining.
Listed under open decisions in [HANDOFF.md](HANDOFF.md). SmolVLA here is a size/latency reference, not a
proposed driving policy as-is.

**C3. The conversion patches could change SmolVLA's behaviour.** Limitation (the check is logged; agreement is
not shown). Five conversion fixes are listed in [PLAN.md](PLAN.md). `export_smolvla.py` writes
`static_position_patch_max_diff`, which is the torch output before vs after all four torch-side patches
(vision positions, vision mask, RoPE, attention), not only the static-position one; the fifth fix
(`reciprocal`) is a converter override that this diff cannot see. It also writes Core ML vs torch parity (CPU
and Neural Engine paths) into `results/step0/conversion/smolvla_base_s10_l16_fp16.json` (`make export-smolvla`).
The torch-side patch diff is zero for the one input. The Core ML vs torch error, however, is a sizeable
fraction of the output scale on both paths (larger on the Neural Engine path): this is not shown to be fp16
noise, and it is unknown whether it comes from fp16 accumulation over the 10 unrolled Euler steps, the
attention-mask change, or the conversion. So "the Core ML SmolVLA computes what SmolVLA computes" is not
claimed. Experiment (not run; an export takes tens of minutes and must not overlap a bench): fp32 Core ML
export parity, parity per stage (prefix vs each Euler step), and several inputs. One input only.

## D. Robot and firmware

**D1. The firmware has never run on hardware.** Limitation. Host unit tests pass (`make firmware-test`,
[firmware/test/test_control.c](firmware/test/test_control.c)) and it builds for ESP32-S3 (`make firmware-build`).
Watchdog, slew limit and stop behaviour are verified only in the host tests. Hardware tests happen only on the
owner's request.

**D2. "A phone VLA driving a real robot" is not shown yet.** Limitation. The claim is not made until a closed-loop
run exists (PLAN Steps 5 and 6). No robot, no closed loop and no demos exist today.

**D3. Demo videos can be cherry-picked.** Experiment (not run). Report every trial with its outcome and the
success rate with denominators; keep full attempt logs in `results/`; film the airplane-mode toggle on camera.

**D4. "Offline" must be proven, not asserted.** Experiment, for speech and for the demo; the speech code is
ready, the run is not done. The speech test sets `requiresOnDeviceRecognition` and samples the network path
(`NWPathMonitor`) during recognition: at recognition start, every 250 ms while it runs, on every path change,
and at the end. Each sample records time, path status, interface types (wifi/cellular/wired/loopback/other),
isExpensive and isConstrained; the full series is saved in the result JSON (`networkSamples`). The result fails
the offline check (`offlineCheck.offline == false`, with `reasons`) if any sample shows a usable path
(`satisfied`, `requiresConnection` or an unknown status), if a sample was taken before the first path update,
if the start or end sample is missing, or if two samples are more than 1 s apart
([NetworkSampler.swift](app/Bench/Sources/NetworkSampler.swift),
[SpeechTest.swift](app/Bench/Sources/SpeechTest.swift)). The sampler and the rule are tested on macOS with
synthetic series and a short live run (`make app-nettest`, [NetTest](app/Bench/NetTest/main.swift)).
Experiment: in airplane mode with Bluetooth on, run "Record 5 s, on-device only" on the iPhone
(app/README.md); output goes to `results/step0/phone/speech_*.json`. Not run (needs Xcode and the iPhone).
Limitations: a path sample shows that no network path existed, not that no packet was sent; samples are
250 ms apart, so a path that comes and goes between ticks is seen only through the path-change event, if NWPathMonitor reports it [verify]. The full
demo needs the airplane-mode-on, Bluetooth-on state shown on camera (PLAN Step 6).

**D5. Safety: anyone nearby can connect over BLE.** Limitation. No pairing or authentication yet (PLAN Risks);
bonding/allowlist before any demo is an open decision. The safety stop lives on the ESP32 (watchdog, disconnect
stop), not on the phone.

## E. Novelty, prior work, disclosure

**E1. "First VLA on a phone."** Addressed. Not claimed; PLAN fixes the wording ("a phone VLA driving a real
robot") and names BitVLA as an earlier phone VLA. Prior work is listed in [README.md](README.md); items marked
[verify] still need the owner to check the source.

**E2. Teacher licensing.** Limitation / open decision. Full OmniVLA is derived from Llama 2 (not Apache/BSD/MIT).
Distilling from it is not done without the owner's decision ([THIRD_PARTY.md](THIRD_PARTY.md)).

**E3. Conflict of interest.** Addressed by disclosure in [README.md](README.md): the author's NVIDIA internship
runs until Nov 13, 2026, and the author founded Quatern. Training is planned on an NVIDIA RTX GPU. (README says
"was an NVIDIA intern until Nov 13, 2026", which is only accurate from that date on; see the reviewer pass.)

## F. Reproducibility

**F1. Can every number be regenerated with one command?** Addressed for what exists. [Makefile](Makefile) has
one target per result family; README lists them. Phone results come from app/Bench by hand (app/README.md run
order); that is a manual procedure, not one command.

**F2. Older result files lack provenance.** Limitation. Files written before this commit have tool versions but no
date, git commit or seeds. They are kept as-is; rerunning the `make` target produces files with full provenance.

**F3. Failed conversion attempts.** Limitation for the past; addressed from this commit on. From this commit on, every conversion attempt (including
failures, with traceback) is kept under `results/step0/conversion/history/`. The failed SmolVLA attempts that led
to the five fixes were overwritten before this change; only their descriptions in PLAN.md remain.

**F4. Unpinned environment.** Addressed. Model revisions are pinned in THIRD_PARTY.md and recorded in the
conversion logs; the Python environment is frozen in [training/requirements-mac.lock](training/requirements-mac.lock);
ESP-IDF is pinned by container tag.

## Reviewer pass (2026-11-06)

A fresh, skeptical read before anything is posted: CLAUDE.md, README, PLAN, HANDOFF, THIRD_PARTY, this file,
the code, and every file in the git-ignored `results/` (there is no `reports/`). Read-only scripts used:
`results/_scratch/review_summary.py` and `results/_scratch/review_numbers.py`. No bench was started: a SmolVLA
bench from an earlier session (PID 82289) was still running for the whole pass, and it was left alone.
Checks run: `make test` (student unit tests and firmware host tests pass), plus one cheap experiment (R3 below).

### R1. Claims that do not trace to a results file, or exceed the evidence

| Where | Claim | Problem | Action |
|---|---|---|---|
| CRITIQUES C1 (old), `results/step0/mac_proxy/NOTES.md` | student multi-trial Mac files are usable ("use the n 50 files") | All of them overlapped another bench: start stamps and file times show an n 20 and an n 50 series running at the same time, and one file also overlapped PID 82289. Written before the overlap guard, so no load fields. | Fixed in C1 and A8b; correction appended to the NOTES file. Clean rerun needed for every model. |
| CRITIQUES C3 (old) | "Addressed" | The logged Core ML vs torch error for SmolVLA is a sizeable fraction of the output scale on both paths; nothing says so. The "static-position patch" diff covers all four torch-side patches. | Reclassified as Limitation with the outcome and an experiment. |
| CRITIQUES B3 (old) | "the split (text + policy) export equals the full model" | The test compares torch modules, not exported packages. | Fixed. |
| CRITIQUES header | "Every result file records date, git commit, ..." | The older single-trial files do not (F2 says so). | Fixed: "written by the current scripts". |
| CRITIQUES A8b | bench refuses while "another bench/export/training job" runs | Only this repo's named Python scripts are detected. | Fixed. |
| CRITIQUES D4 | speech test "records whether any network path was up" | Sampled once, at save time after recognition. | Fixed. Later (branch d4-network): sampled during recognition, see D4. |
| CRITIQUES E3, README Disclosure | "was an NVIDIA intern until Nov 13, 2026" | As of this pass (2026-11-06) the internship is ongoing; "was" is only accurate after Nov 13. | E3 reworded; README left for the owner (suggest "the author's NVIDIA internship ends/ended Nov 13, 2026"). |
| PLAN Step 0, HANDOFF Done | student policy part "entirely" / "all ops" on the Neural Engine in the Mac compute plan | `ops_off_neural_engine` in `results/step0/conversion/student_policy_fp16.json` lists a few small ops (mask building) placed on the CPU; and a compute plan is Core ML's preferred-device estimate, not proof of where ops ran. | Owner: reword to "almost all estimated cost on the Neural Engine in the Mac compute plan". See R2.4. |
| HANDOFF Done | "Core ML matches torch within fp16 noise" (student) | No tolerance is defined (B4 says so). | Owner: reword to "max/mean abs error logged in results/step0/conversion/". |
| HANDOFF Done | MacSmoke bench "agrees with the Python proxy" | No MacSmoke output exists in `results/`. | Owner: drop, or rerun MacSmoke and save its output under `results/step0/mac_proxy/`. |
| `training/models/student.py` docstring | "on the phone one pass is ~10x cheaper than 10 Euler steps" | A number with no measurement, made before any phone run; for a SmolVLA-style model the prefix (vision + VLM) runs once and only the action expert repeats per step, so the ratio is not the step count. | Owner: remove the ratio; the flow-step sweep in C1 would measure it. |
| PLAN, HANDOFF | student "converted with no patches" | True for model code; the `reciprocal` converter override in `common.py` is registered on import for every export, the student's included. | Minor; owner may add "(no model-code patches)". |
| PLAN, B8 | SmolVLA chunk = 5.0 s "at 10 Hz" | 10 Hz is this project's control rate, not necessarily the rate SmolVLA was trained at [verify]. The SmolVLA pass bar depends on it (B8). | State it as an assumption wherever the SmolVLA bar is shown. |

Claims that do trace: firmware constants in PLAN match `firmware/main/control.h` and `board.h`; the
NOT-phone note is in every Mac file; parameter counts and model revisions are in the conversion logs; the
overfit outcome in B2 matches `results/student/overfit_*.json` and its NOTES; the SmolVLA torch-side patch
diff is as logged (plausibly exact: the attention change only moves a power-of-two scale and swaps one large
negative mask value for another).

### R2. Criticisms that were missing, with answers

**R2.1 No clean Mac latency file exists for any model.** Limitation now; experiment pending. See C1/A8b.
Run on an idle Mac, one model at a time: `make bench-mac N=50 TRIALS=5`. Not run here (PID 82289 running).

**R2.2 SmolVLA Core ML output differs from torch by more than fp16 noise.** Limitation; see C3 for the
experiment. Any SmolVLA latency number is for a Core ML graph whose agreement with SmolVLA is unproven.

**R2.3 Bench files do not identify the exact model file benched.** Limitation; cheap code fix not done.
`bench_mac.py` records the package path as given (relative, no working directory) and no hash, and two
checkouts (main and this worktree) each have a `training/out/`. A reader cannot tie a latency file to a
conversion log. Fix: record the absolute path, a hash of the package's `weight.bin`, and the conversion
log's `date_utc` read from the package's metadata or the matching log.

**R2.4 "Runs on the Neural Engine" is a plan, not a measurement, and the phone may plan differently.**
Limitation. `inspect_package` reads `MLComputePlan` preferred devices on the Mac. A preferred device is not
proof of runtime dispatch, and an iPhone's ANE generation can plan differently. The app does not record a
compute plan. Experiment (not run): log the compute plan on the phone in app/Bench, and compare CPU_AND_NE
with CPU_ONLY latency there (if they match, the ANE is not being used).

**R2.5 Small-sample tails and best-unit selection.** Rule. p95 from one trial of a few dozen predictions
rests on a handful of samples; quote pooled p95 with its `n`, and the median of trial medians with the min
and max of trial medians. Report every compute unit measured, not only the fastest; choosing the best unit
after seeing results is stated as such.

**R2.6 "Hz" is not a control rate.** Limitation. `hz` in the older Mac files and in the app's `Stats` is
1000 / median model latency. It ignores capture, preprocessing, BLE and chunk scheduling (A6) and must not be
quoted as the robot's control rate.

**R2.7 Sustained phone run is not the real thermal load.** Limitation. The 10-minute mode runs predictions
back to back with no camera, ISP, BLE or chunk scheduling. Inference duty is higher than in the real loop,
but camera and radio heat are missing, so it is not a bound either way. Experiment (Step 5): log
`thermalState` and latency inside the closed-loop app.

**R2.8 Parameter count framing.** Rule, added to C1. The student's total includes a text embedding table and
encoder that run once per instruction; the per-frame count is smaller. Whether SmolVLA's logged total counts
VLM layers that the policy drops is not checked [verify].

**R2.9 SmolVLA exported in one configuration.** Limitation. One camera at 512x512, 16 instruction tokens, zero
state, noise supplied as an input. SmolVLA is commonly run with more than one camera [verify]; latency here is
for this configuration only and must be labelled so.

**R2.10 Speed limit and physical safety.** Limitation / open decision. PLAN caps speed at about 0.3 m/s in
Step 5, but the firmware only caps PWM at boot (open loop, not a speed). A skeptic will ask for the speed cap
on the ESP32 (encoder-based) and for a physical stop (switch or bumper) before any closed-loop run.

**R2.11 Reaction time with open-loop chunks.** Limitation. Between inferences the robot executes a chunk
open-loop; time to react to a new obstacle is up to the inference period plus latency, and is not part of the
Step 0 pass bar. Experiment (Step 5): measure stop latency to a suddenly placed obstacle, every trial reported.

**R2.12 Overlap guard scope.** Limitation, now stated in A8b.

**R2.13 Mac power and power mode.** Limitation, minor. Bench files record `pmset -g batt` (power source and
charge state) but not macOS Low Power Mode. Add `pmset -g` `lowpowermode` to `provenance` before the clean
rerun.

**R2.14 ESP-IDF pinned by tag.** Limitation, minor (F4). `espressif/idf:v5.4.1` is a tag, which can be
re-pushed; pin the image digest in the Makefile and THIRD_PARTY.md.

**R2.15 Prior work that a reader will expect.** Limitation until the owner checks the sources; all [verify]:
ACT (Zhao et al., 2023, action chunking; the student head is described as "ACT-style" but ACT is not cited);
pi0 (Black et al., 2024, flow-matching action expert that SmolVLA follows); OpenVLA (Kim et al., 2024, base of
full OmniVLA); GNM / ViNT / NoMaD (Shah, Sridhar et al., 2023-2024; navigation foundation models for wheeled
robots, the closest prior work for this task); small or edge VLAs such as TinyVLA, MiniVLA and EdgeVLA;
NVIDIA GR00T N1 (2025; relevant given the disclosure, cited neutrally); Apple's "Deploying Transformers on the
Apple Neural Engine" (2022; fp16 and ANE layout practice). README not edited; owner to add the ones that apply.

**R2.16 Relation between Quatern and this project.** Open decision for the owner. The disclosure says the
author founded Quatern but not whether pocket-vla is a Quatern project or whether Quatern has a commercial
interest in phone VLAs or the hardware. A reader will ask; state it either way in README and any post.

### R3. Experiment run in this pass (cheap)

Tanh saturation probe for B2: `results/_scratch/tanh_probe.py --repo <worktree> --runs 3e-4:0,1e-4:1 --steps 40`
(same data, init and optimiser as `make overfit-student`), output `results/student/tanh_probe_<date>.json`,
reading appended to `results/student/NOTES.md`. Outcome (no numbers here): saturation is measured, not
hypothesised, for the failed run probed; the losses match the logged curves for both runs, so the earlier
overfit file written from a dirty tree is reproduced by committed code for that seed. Only 2 of the 6 runs
were probed, and only their first 40 steps. B2 updated. The probe script lives in git-ignored
`results/_scratch/`; if the owner wants it reproducible from the repo, move it to `training/tests/` and add a
Makefile target.

### R4. Reporting checks

- All attempts and runs: overfit check keeps all six runs with full curves; Mac bench keeps every trial and
  raw latency; contaminated files are kept and marked, not deleted. The guard smoke test in
  `results/_trash/` is described in the mac_proxy NOTES. OK.
- Medians and spreads: present in the current bench files; the old single-trial files have a single median
  and p95 only (F2). No current file is clean (R2.1), so there is nothing to report yet.
- Simulation: stated plainly in README and here (none used). The overfit check is synthetic data, labelled as
  such everywhere it appears. OK.
- Prior work: present with [verify] marks; gaps listed in R2.15.
- Disclosure: in README and E3; tense issue (R1) and Quatern relation (R2.16) open. Any post, the README and
  any comparison involving NVIDIA hardware or models should carry it.
- Neutral wording: no "first", "N times faster" or "works" claims found in tracked files. The
  `student.py` ratio (R1) is the one unmeasured comparative claim.
- Owner-required critiques: Mac proxy is not phone (A1, A2, R2.4): honest. Student untested (B1, B2, R3):
  honest, and the synthetic check failed in most runs. SmolVLA like-for-like (C1): rule is sound, but no clean
  data exists for either side, and the SmolVLA graph's agreement with torch is unproven (C3).

### What this pass changed in this file

- Header: "every result file" limited to files written by the current scripts.
- A8b: guard scope stated; the student files added to the contaminated list.
- B2: tanh saturation changed from "suspected, not measured" to measured (probe), follow-ups updated.
- B3: test compares torch modules, not exported packages.
- C1: status rewritten (no clean file for either model; rerun all four); per-frame parameter rule added.
- C3: reclassified Addressed -> Limitation; patch-diff scope corrected; parity outcome stated without numbers;
  experiment added.
- D4: network state sampled only at save time.
- E3: disclosure tense made accurate on any date.
- F3: "Partly addressed" (not one of the three answers) -> "Limitation for the past; addressed from this
  commit on".
Not changed (left for the owner): README (disclosure tense, prior work, Quatern relation), PLAN and HANDOFF
wording in R1, `student.py` docstring, `bench_mac.py` provenance additions (R2.3, R2.13), Makefile digest pin.
