# pocket-vla

A vision-language-action model running on an iPhone drives a 2-wheel robot: ESP32-S3 BLE motor firmware + phone app.

**Status:** nothing has run on an iPhone yet, nothing has driven a robot yet, and the 76M student model has
random weights (never trained). Timings so far are Mac proxy measurements, not phone measurements. No
simulation is used. Known limitations and open criticisms: [CRITIQUES.md](CRITIQUES.md).

- `firmware/`: ESP-IDF + NimBLE motor controller (BLE commands, encoders, watchdog)
- `training/`: student model and Core ML exports
- `eval/`: benchmark scripts (outputs go to git-ignored `results/`)
- `app/`: iOS apps; `app/Bench` is the Step 0 on-phone benchmark ([app/README.md](app/README.md))

Plan and decisions: [PLAN.md](PLAN.md). Licenses of models and tools: [THIRD_PARTY.md](THIRD_PARTY.md).
Code is Apache-2.0 ([LICENSE](LICENSE)).

## Reproducing results

Each result family has one command (see the [Makefile](Makefile)); outputs go to `results/` with date, git
commit, command line, versions, seeds and settings recorded in the file. Environment:
[training/requirements-mac.lock](training/requirements-mac.lock).

```
make test              # student unit tests + firmware host tests
make export-student    # Core ML export + parity check, random-weight student
make export-smolvla    # Core ML export + parity check, SmolVLA base (pinned revision)
make bench-mac         # Mac PROXY latency, all trials kept (not phone numbers)
make overfit-student   # synthetic trainability check (not robot data)
make firmware-build    # ESP32-S3 build (pinned ESP-IDF container)
```

## Prior and related work

Entries marked [verify] have not been checked against the source yet.

- SmolVLA (Hugging Face, 2025): compact VLA, candidate (a) here; `lerobot/smolvla_base`. [verify: paper reference]
- OmniVLA (N. Hirose et al.): omni-modal navigation VLA; OmniVLA-edge is candidate (c), full OmniVLA a possible
  distillation teacher (license open, see THIRD_PARTY.md). [verify: paper reference]
- BitVLA: 1-bit VLA reported to run on an iPhone; the reason this repo does not claim "first VLA on a phone".
  [verify: paper reference and the reported phone timing]
- vla.cpp: on-device VLA inference runtime; related runtime work. [verify: repo, scope]
- MetaMorph (Gupta et al., ICLR 2022) and URMA (Bohlinger et al., 2024): policies that transfer across robot
  morphologies; relevant because SmolVLA is pretrained on arms and this robot is a differential drive.
  [verify: both references]

