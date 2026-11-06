# One command per result. Outputs go to git-ignored results/ (see CRITIQUES.md for what each one supports).
# Every result file records date, git commit, command line, tool versions, seeds and settings.
#
#   make test                 student unit tests + firmware host tests
#   make export-student       Core ML export of the random-weight student -> results/step0/conversion/student_*.json
#   make export-smolvla       Core ML export of SmolVLA base (pinned revision) -> results/step0/conversion/smolvla_*.json
#   make bench-mac            Mac PROXY latency, all trials kept -> results/step0/mac_proxy/<model>_<date>.json
#   make overfit-student      SYNTHETIC trainability check -> results/student/overfit_<date>.json
#   make firmware-build       ESP32-S3 build in the pinned ESP-IDF container
#   make app-nettest          macOS test of the speech test's network sampler + offline rule (CRITIQUES D4)
#
# Phone numbers do not come from here: they come from app/Bench on the iPhone (app/README.md).

PY      ?= training/.venv/bin/python
OUT     ?= training/out
RESULTS ?= results
TRIALS  ?= 5
N       ?= 50
MODELS  ?= student_policy_fp16 student_text_fp16 student_full_fp16 smolvla_base_s10_l16_fp16

.PHONY: test export-student export-smolvla bench-mac overfit-student firmware-test firmware-build app-nettest

test: firmware-test
	$(PY) training/tests/test_student.py

firmware-test:
	$(MAKE) -C firmware/test

export-student:
	$(PY) training/export/export_student.py --precision fp16

export-smolvla:
	$(PY) training/export/export_smolvla.py --steps 10 --lang 16 --precision fp16

bench-mac:
	for m in $(MODELS); do \
	  $(PY) eval/step0/bench_mac.py $(OUT)/$$m.mlpackage --n $(N) --trials $(TRIALS) \
	    --out-dir $(RESULTS)/step0/mac_proxy || exit 1; \
	done

overfit-student:
	$(PY) training/tests/overfit_student.py --seeds 0,1,2 --steps 300 --lr 3e-4 --out-dir $(RESULTS)/student
	$(PY) training/tests/overfit_student.py --seeds 0,1,2 --steps 300 --lr 1e-4 --out-dir $(RESULTS)/student

firmware-build:
	cd firmware && docker run --rm -v "$$PWD":/project -w /project espressif/idf:v5.4.1 idf.py build

app-nettest:
	mkdir -p $(OUT)
	swiftc -o $(OUT)/nettest app/Bench/Sources/NetworkSampler.swift app/Bench/NetTest/main.swift
	$(OUT)/nettest
