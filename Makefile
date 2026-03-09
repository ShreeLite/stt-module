PYTHON ?= ./.venv/bin/python

.PHONY: test-fast test-full verify-env

test-fast:
	./scripts/test_fast.sh

test-full:
	./scripts/test_full.sh

verify-env:
	$(PYTHON) scripts/verify_stt_environment.py --audio voice-sample.wav --model tiny --device cpu
