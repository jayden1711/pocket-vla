# Third-party models, code and tools

None of these are committed or redistributed here. Models and checkpoints are fetched at the pinned revision;
exported `.mlpackage` files go to git-ignored `training/out/`. Keep this in sync with PLAN.md.

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
