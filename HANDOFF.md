# HANDOFF (2026-11-06)
Status: CRITIQUES.md, provenance + all-trials logging and Makefile on branch `critiques` (local); still no phone or robot run
Next: Owner reviews CRITIQUES.md and the [verify] refs in README; then Xcode + iPhone Step 0 run (app/README.md)
Updated: 2026-11-06 (daytime run, branch critiques)

## Daytime run 2026-11-06 (branch `critiques`, local commits, not pushed)
- CRITIQUES.md at the repo root: each criticism is answered as addressed / experiment / limitation and points
  to files in results/. Covers the owner's three: Mac proxy is not phone, student untested, SmolVLA like-for-like.
- Provenance on every result: bench_mac.py and the export scripts record UTC date, git commit + dirty flag,
  argv, versions, seeds, settings and model I/O shapes. bench_mac.py now runs `--trials` (default 5), keeps
  every raw latency, reports median/p5/p95/min/max/std per trial and the spread of trial medians, and writes
  a new timestamped file per run (old single-trial files kept unedited). Conversion logs keep every attempt
  under results/step0/conversion/history/ and include Core ML vs torch parity on the CPU and Neural Engine paths.
- SmolVLA export now seeds its example input (it was unseeded) and logs model ID + revision + settings.
- App: BenchResult gains `settings` (runs, warmup, input seed, Low Power Mode, app version, model I/O shapes).
  Type-checks against the macOS SDK; still not built for iOS.
- Makefile: one target per result family (test, export-student, export-smolvla, bench-mac, overfit-student,
  firmware-build). training/requirements-mac.lock = exact frozen venv.
- training/tests/overfit_student.py: SYNTHETIC trainability check (3 seeds x 2 learning rates, all kept).
- README: status line, reproduction commands, prior work ([verify] marks), NVIDIA/Quatern disclosure.
  PLAN: unsourced BitVLA timing replaced by [verify]; links to CRITIQUES.md.
- Runs (outputs in git-ignored results/, read the NOTES there before using the student):
  student re-export with parity (results/step0/conversion/, history/); synthetic overfit check, 6 runs
  (results/student/overfit_*.json + results/student/NOTES.md); SmolVLA re-export and multi-trial Mac bench:
  BENCH_STATUS

## Decisions taken (daytime 2026-11-06)
- Branch: the run brief said `day-1106`, the owner's repo decision said `critiques`; used `critiques` (the
  owner's, more specific). Worktree: moved to ~/Downloads/projects/wt/pocket-vla-critiques (writes under .claude/ prompt).
- Prior work: cited only what looks relevant (SmolVLA, OmniVLA, BitVLA, vla.cpp, MetaMorph, URMA), all
  [verify] since nothing was checked on the web. NOT cited: the MIT weight-space post, PicoGPT, Hirundo /
  CrowdStrike / Booz Allen, RoboStriker (arXiv 2601.22517), Text2Robot: their content and relevance to a phone
  VLA could not be checked without the web. Owner: add any that apply (RoboStriker may belong under CRITIQUES
  D5 if it is about attacks on robot policies).
- Removed the BitVLA phone timing from PLAN (unsourced number), replaced with [verify].
- Disclosure added to README (training is planned on NVIDIA RTX hardware).
- Old result files without provenance left as they are (not edited, not deleted); new runs write new files.
- Overfit check runs two learning rates and reports both instead of tuning to one; CPU for determinism.
- SmolVLA flow-step sweep (CRITIQUES C1 experiment) not run; left for the owner.

## Blocked (daytime)
- ~/projects/bin/make-index does not exist on this Mac (no ~/projects/ at all, so no ~/projects/NUC_RULES.md
  either); index not regenerated.
- Phone/Xcode work: as before, needs Xcode + the owner's iPhone.
- Loading an .mlpackage with CPU_AND_NE inside the export process (torch 2.15 + coremltools 9.0) segfaulted;
  it works in a torch-free process, so parity runs in a subprocess (training/export/parity_ne.py). Another
  reason to try torch 2.7 (coremltools' tested version).

## Done
- Repo skeleton: firmware/, app/, training/, eval/, PLAN.md (licenses, decisions, plan).
- Step 1 firmware (firmware/main/): ESP-IDF + NimBLE, BLE UART-style service, "M <l> <r>\n" / "S\n" commands,
  "E <ms> <encL> <encR> <outL> <outR> <state> <bad>\n" telemetry at 20 Hz, 300 ms watchdog, disconnect stop,
  slew limit with stops exempt, TB6612FNG (or DRV8833) PWM, PCNT quadrature encoders, task-WDT reboot.
  Host unit tests pass; ESP32-S3 build passes with no warnings. Not tested on hardware.
- Step 0 candidate (a) SmolVLA base: converts to Core ML (fp16, iOS18 target) after 5 fixes (listed in PLAN.md).
  Mac M3 proxy bench done; numbers in results/step0/ (local, git-ignored). NOT phone numbers.
- Step 0 candidate (b) 76M student (overnight): training/models/student.py (ViT-S/16 + 2-layer text enc +
  10-layer fusion, 16x2 chunk, direct regression, random weights), training/export/export_student.py writes
  student_full / student_text / student_policy .mlpackages. Converted with no patches, Core ML matches torch
  within fp16 noise, outputs finite. Policy part: all ops on the Neural Engine in the Mac compute plan.
  Mac proxy numbers in results/step0/{conversion,mac_proxy}/student_*.json. training/tests/test_student.py passes.
- app/Bench (overnight): SwiftUI bench app: model picker (bundled from training/out/ or dropped into
  Documents), compute units, N-run median/p95/Hz, 10-min sustained run logging thermalState + phys_footprint per
  prediction, pass-bar check, offline speech test (requiresOnDeviceRecognition, records whether network was up).
  JSON results to the app's Documents/results/. XcodeGen spec in app/Bench/project.yml; how-to in app/README.md.
  Type-checks against the macOS SDK; the bench core also ran on the Mac via app/Bench/MacSmoke against the
  student models and agrees with the Python proxy. NOT built for iOS (no Xcode here).
- Candidate (c) OmniVLA-edge: training/export/export_omnivla_edge.py written (language-goal mode, masked
  branches pruned, compares against upstream forward, plus CLIP B/32 text encoder export). NOT RUN (see below).

## In progress / not started
- OmniVLA-edge export: needs a first run (and probably small fixes).
- iOS benchmark app: needs Xcode to build/sign; never run on a device.
- On-phone numbers: median/p95/Hz/peak memory, 10-min sustained (thermal), offline speech test: none collected yet.
- Pass bar check (p95 after 10 min <= 1 s and <= 40% of chunk duration) cannot be judged until the phone run.

## Resume commands
Shortcut: `make test export-student export-smolvla bench-mac overfit-student` (see Makefile). Long form:
```
cd pocket-vla
uv pip install --python training/.venv/bin/python -r training/requirements-mac.lock   # exact frozen env
uv venv --python 3.12 training/.venv
uv pip install --python training/.venv/bin/python torch coremltools "lerobot[smolvla] @ git+https://github.com/huggingface/lerobot@d490e472a5d1" efficientnet_pytorch==0.7.1
training/.venv/bin/python training/export/export_smolvla.py           # ~6 min, writes training/out/*.mlpackage + results/step0/conversion/*.json
training/.venv/bin/python training/export/export_student.py           # <1 min, student_{full,text,policy}_fp16
training/.venv/bin/python training/tests/test_student.py
training/.venv/bin/python eval/step0/bench_mac.py training/out/student_policy_fp16.mlpackage --n 100
make -C firmware/test                                                  # firmware host tests
cd firmware && docker run --rm -v "$PWD":/project -w /project espressif/idf:v5.4.1 idf.py build
# bench app (needs Xcode): brew install xcodegen && cd app/Bench && xcodegen generate && open Bench.xcodeproj
# OmniVLA-edge (owner approval, see below):
curl -sSL https://codeload.github.com/NHirose/OmniVLA/tar.gz/5182600cb4a9 | tar -xz -C /tmp
training/.venv/bin/python training/export/export_omnivla_edge.py --omnivla-src /tmp/OmniVLA-5182600cb4a9
```
Note: coremltools 9.0 warns torch 2.15 is untested (tested up to 2.7). Conversion worked anyway; pin torch 2.7 if it breaks.

## Next steps
1. Install Xcode + XcodeGen; build app/Bench, set signing team, run on the iPhone (app/README.md has the run order).
2. Run Step 0 on the phone: each model x compute units, 10-min sustained on the best setting, airplane-mode
   speech test. Copy Documents/results/*.json to results/step0/phone/. Fill the Step 0 table in PLAN.md
   (numbers only after you approve), pick model + runtime. Then stop for review.
3. Optionally run the OmniVLA-edge export and add it to the phone bench.

## Decisions taken overnight
- Student action head = direct regression of the 16x2 chunk (one forward pass, tanh to [-1,1]), not flow
  matching. Cheapest on the phone; flow matching can be added if multimodal actions turn out to matter.
- Student text side: SmolVLM2 tokenizer vocab (Apache-2.0, already pinned) so no new tokenizer dependency;
  text encoder depth 2 to land at ~76M total. Text is exported separately so it runs once per instruction.
- Student is plain PyTorch written here (no timm), so there is no new third-party model code to pin.
- Wrote the bench app before attempting OmniVLA-edge, since the phone run is the real Step 0 blocker.
- OmniVLA-edge: the overnight session's safety check blocked running the downloaded upstream model file +
  checkpoint, so the export script is committed untested and marked NOT RUN. The script imports only
  inference/model_omnivla_edge.py (read through: plain torch + efficientnet_pytorch) and loads the
  checkpoint with torch.load(weights_only=True). Run it yourself if you're fine with that.
- Installed efficientnet_pytorch 0.7.1 (Apache-2.0) into training/.venv for that script; added it and the
  CLIP B/32 HF revision to THIRD_PARTY.md / PLAN.md as candidate (c) dependencies.
- Bench app memory is phys_footprint of the app process. On the Mac with the Neural Engine it stays small
  because the ANE's weight copy is not charged to the process; noted in app/README.md so the phone memory
  column isn't misread as total model memory.
- Overnight work was merged into main and pushed by the owner (2026-11-06).

## Open decisions for you
- Exact iPhone model (affects ANE generation / expectations).
- Distilling from full OmniVLA: its backbone is Llama 2 (not Apache/BSD/MIT). OK to use as teacher, or restrict to OmniVLA-edge / other teachers?
- SmolVLA chunk is 50 steps (5 s at 10 Hz) of 32-dim arm actions; a 2-output drive head discards most pretraining (as you noted).
- BLE has no pairing/auth; add bonding before the demo?
- OK to run export_omnivla_edge.py (executes upstream OmniVLA model code, downloads the 434 MB MIT checkpoint)?
- Merge `critiques` into main? Review CRITIQUES.md and the [verify] references in README first.
- Which of the unverified references (MIT weight-space post, PicoGPT, Hirundo / CrowdStrike / Booz Allen,
  RoboStriker, Text2Robot) belong in README?

## Uncommitted (on purpose, gitignored)
- training/.venv, training/out/*.mlpackage, firmware/build, firmware/sdkconfig: all regenerable with the commands above.
  The student .mlpackages are in the worktree's training/out/ (.claude/worktrees/overnight-student/); rerun export_student.py in main after merging.
- Rules in the wrap-up note about robot-arena / robot-framing-test don't apply to this repo (none of that code is here).
