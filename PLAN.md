# pocket-vla plan

Claim (use verbatim): "a phone VLA driving a real robot." Never "first VLA on a phone" (BitVLA is reported to already run on an iPhone [verify: source and timing]).
Phone: iPhone (exact model TBD). Runtime: Core ML.

## Status
- Step 0: in progress. See HANDOFF.md. SmolVLA and the 76M student convert to Core ML; bench app (app/Bench) written but not built (needs Xcode). Only Mac-proxy timings so far, no phone numbers yet.
- Step 1: firmware written, host-tested, builds for ESP32-S3. Not yet on hardware.
- Steps 2-6: not started (plan below).

Open criticisms and how each is answered (addressed / experiment / limitation): [CRITIQUES.md](CRITIQUES.md).
Result commands: [Makefile](Makefile). Prior work: [README.md](README.md).

## Decisions
- Firmware uses ESP-IDF + NimBLE (both Apache-2.0), NOT the Arduino-ESP32 core (LGPL-2.1, violates license rule).
- Only a full stop ("M 0 0" / "S"), the watchdog and BLE disconnect bypass the slew limit; a one-wheel zero is slewed.
- Watchdog 300 ms, control tick 10 ms, slew 6/tick (0->255 in ~430 ms), boot output cap 120/255.

## Third-party sources (license, pinned commit)
| Source | License | Commit / revision | Use |
|---|---|---|---|
| huggingface/lerobot | Apache-2.0 | d490e472a5d1 | SmolVLA code |
| lerobot/smolvla_base (HF) | Apache-2.0 | 5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6 | candidate (a) |
| HuggingFaceTB/SmolVLM2-500M-Video-Instruct (HF) | Apache-2.0 | 7b375e1b73b11138ff12fe22c8f2822d8fe03467 | SmolVLA backbone/tokenizer |
| huggingface/transformers | Apache-2.0 | pip 5.22.1 | |
| apple/coremltools | BSD-3-Clause | pip 9.0 | conversion |
| espressif/esp-idf | Apache-2.0 | v5.4.1 (docker espressif/idf:v5.4.1) | firmware |
| NimBLE (in ESP-IDF) | Apache-2.0 | bundled with IDF v5.4.1 | BLE |
| NHirose/OmniVLA | MIT (code) | 5182600cb4a9 | distillation teacher (planned) |
| NHirose/omnivla-original (HF) | MIT (card) | e36a84d4923c041149d441f93f3bdb7092bb5f07 | teacher; LICENSE RISK below |
| NHirose/omnivla-edge (HF) | MIT | b1361b7e24f101edea795a98b00a826b61a97394 | possible candidate (c) |
| lukemelas/EfficientNet-PyTorch (efficientnet_pytorch) | Apache-2.0 | pip 0.7.1 | OmniVLA-edge encoder (candidate (c)) |
| openai/clip-vit-base-patch32 (HF) | MIT (OpenAI CLIP) | 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268 | OmniVLA-edge text features (candidate (c)) |

License risks: OmniVLA (full) is fine-tuned from openvla-7b, whose LLM is Llama 2 (Llama 2 Community License, not Apache/BSD/MIT). Distilling from it needs a decision. OmniVLA-edge uses EfficientNet-B0 (efficientnet_pytorch, Apache-2.0) + OpenAI CLIP text features (MIT); verify those pins before use. BDD images in OmniVLA data carry the BDD license.

## Step 0 results so far (Mac M3 PROXY, not phone; student = random weights)
Any student vs SmolVLA latency comparison must list image size, chunk, flow steps, params and weights per row
(CRITIQUES.md C1); no "N times faster" claim.
SmolVLA base, 1 cam 512x512, 16 lang tokens, 10 Euler steps, fp16, chunk 50 (= 5.0 s at 10 Hz).
Numbers are in `results/step0/` (git-ignored) until the owner approves publishing.

Candidate (b) student, random weights (training/models/student.py): ViT-S/16 224x224 + 2-layer text encoder
(SmolVLM2 tokenizer vocab, 16 tokens) + 10-layer 512-d fusion with 16 learned action queries, direct
regression of a 16x2 chunk (= 1.6 s at 10 Hz), ~76M params. Exported as one model and as a text part (once per
instruction) + per-frame policy part. Converted with no patches; Mac compute plan puts the policy part
entirely on the Neural Engine, the text part on the GPU (embedding gather). Numbers in `results/step0/`.

Candidate (c) OmniVLA-edge: export script for language-goal mode written (masked pose/map/goal-image branches
pruned, checked against upstream forward); not run yet.

Conversion issues found and fixed (training/export/): `bucketize` (vision pos ids), `new_ones` (vision patch mask), `reciprocal` on int32 (converter override), in-place slice assign in `apply_rope`, silent fp16 NaN from finfo(float32).min attention mask. Input name `state` is reserved by Core ML -> `robot_state`.

## Later steps
2. App v0: BLE teleop + record frames/commands with capture timestamps.
3. Collect ~50-200 "go to <object>" demos in 2-3 rooms; demo room is a training room.
4. Train/distill on NUC (RTX 5080, Ubuntu 24.04), quantize, convert, re-benchmark on phone.
5. Closed loop, speed capped ~0.3 m/s, success rate per instruction.
6. Film demo: airplane mode on + Bluetooth back on, both shown on camera. No laptop, no cloud.

## App control loop (to build in app/)
Inference thread separate from a 10-20 Hz sender. Frames timestamped at capture; inference uses newest frame. Start next inference at ~half chunk. On new chunk, skip steps for elapsed time since its frame; if elapsed > half chunk duration, discard and fault-stop. Send zero only when a chunk runs out; never repeat last command.

## Hardware (do not order yet)
ESP32-S3 dev board; TB6612FNG or DRV8833; 2WD chassis, N20/TT gear motors ideally with encoders; protected 18650s or BMS pack, 5 V buck, inline fuse, on/off switch; 10k pull-down on driver STBY/nSLEEP; phone mounted low, centered, fixed height/tilt.

## Risks
Conversion is the time sink. Generic VLAs need this robot's own demos. Phones throttle under sustained load. Safety stop lives on the ESP32. BLE has no pairing/auth yet: anyone nearby could connect (add bonding/allowlist before demo).
