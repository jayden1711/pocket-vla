# pocket-vla

A vision-language-action model running on an iPhone drives a 2-wheel robot: ESP32-S3 BLE motor firmware + phone app.

- `firmware/`: ESP-IDF + NimBLE motor controller (BLE commands, encoders, watchdog)
- `training/`: model export (Core ML) and, later, the student model
- `eval/`: benchmark scripts (outputs go to git-ignored `results/`)
- `app/`: iOS app (not written yet)

Plan and decisions: [PLAN.md](PLAN.md). Licenses of models and tools: [THIRD_PARTY.md](THIRD_PARTY.md).
Code is Apache-2.0 ([LICENSE](LICENSE)).
