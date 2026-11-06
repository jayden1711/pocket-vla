# HANDOFF (2026-11-06)
Status: Firmware written (not hardware-tested); SmolVLA Core ML export and Mac proxy bench done
Next: Write the 76M student model + Core ML export; then the Xcode bench app
Updated: 2026-11-06

## Done
- Repo skeleton: firmware/, app/ (empty), training/, eval/, PLAN.md (licenses, decisions, plan).
- Step 1 firmware (firmware/main/): ESP-IDF + NimBLE, BLE UART-style service, "M <l> <r>\n" / "S\n" commands,
  "E <ms> <encL> <encR> <outL> <outR> <state> <bad>\n" telemetry at 20 Hz, 300 ms watchdog, disconnect stop,
  slew limit with stops exempt, TB6612FNG (or DRV8833) PWM, PCNT quadrature encoders, task-WDT reboot.
  Host unit tests pass; ESP32-S3 build passes with no warnings. Not tested on hardware.
- Step 0 candidate (a) SmolVLA base: converts to Core ML (fp16, iOS18 target) after 5 fixes (listed in PLAN.md).
  Mac M3 proxy bench done; numbers in results/step0/ (local, git-ignored). NOT phone numbers.

## In progress / not started
- Step 0 candidate (b) 76M student: not written yet (planned: ViT-S image enc + small text enc + 10-layer fusion, chunk 16x2, random weights).
- Candidate (c) OmniVLA-edge (MIT, ~108M, EfficientNet-B0 + FiLM + CLIP text feats): worth adding; nothing written yet.
- iOS benchmark app (app/Bench): not written. Needs full Xcode (only Command Line Tools on this Mac).
- On-phone numbers: median/p95/Hz/peak memory, 10-min sustained (thermal), offline speech test: none collected yet.
- Pass bar check (p95 after 10 min <= 1 s and <= 40% of chunk duration) cannot be judged until the phone run.

## Resume commands
```
cd pocket-vla
uv venv --python 3.12 training/.venv
uv pip install --python training/.venv/bin/python torch coremltools "lerobot[smolvla] @ git+https://github.com/huggingface/lerobot@d490e472a5d1"
training/.venv/bin/python training/export/export_smolvla.py           # ~6 min, writes training/out/*.mlpackage + results/step0/conversion/*.json
training/.venv/bin/python eval/step0/bench_mac.py training/out/smolvla_base_s10_l16_fp16.mlpackage --n 20
make -C firmware/test                                                  # firmware host tests
cd firmware && docker run --rm -v "$PWD":/project -w /project espressif/idf:v5.4.1 idf.py build
```
Note: coremltools 9.0 warns torch 2.15 is untested (tested up to 2.7). Conversion worked anyway; pin torch 2.7 if it breaks.

## Next steps
1. Write training/models/student.py + training/export/export_student.py; convert and Mac-bench.
2. Optionally export OmniVLA-edge with real weights.
3. Install Xcode + XcodeGen; write app/Bench (fixed image + instruction, N runs, 10-min sustained with thermalState log,
   phys_footprint memory, SFSpeechRecognizer requiresOnDeviceRecognition test in airplane mode).
4. Run on the iPhone, fill the Step 0 table in PLAN.md, pick model + runtime. Then stop for review.

## Open decisions for you
- Exact iPhone model (affects ANE generation / expectations).
- Distilling from full OmniVLA: its backbone is Llama 2 (not Apache/BSD/MIT). OK to use as teacher, or restrict to OmniVLA-edge / other teachers?
- SmolVLA chunk is 50 steps (5 s at 10 Hz) of 32-dim arm actions; a 2-output drive head discards most pretraining (as you noted).
- BLE has no pairing/auth; add bonding before the demo?

## Uncommitted (on purpose, gitignored)
- training/.venv, training/out/*.mlpackage (791 MB), firmware/build, firmware/sdkconfig: all regenerable with the commands above.
- Rules in the wrap-up note about robot-arena / robot-framing-test don't apply to this repo (none of that code is here).
