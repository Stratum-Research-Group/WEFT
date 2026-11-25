.PHONY: test test-unit test-regression test-cpp formal gate

test: test-unit test-regression test-cpp
	@echo "================================================================="
	@echo "  ALL WEFT TEST SUITES PASSED"
	@echo "================================================================="

test-unit:
	@echo "[TEST] Running Python Core Unit Tests..."
	.venv/bin/pytest tests/test_avx512_gemv.py tests/test_fused_swiglu_avx512.py -q

test-regression:
	@echo "[TEST] Running Regression Guardrails T1-T6..."
	.venv/bin/pytest tests/test_regression_guardrails.py tests/test_packed_vs_dense.py tests/test_gate_and_scaling.py -q

test-cpp:
	@echo "[TEST] Running Native C++ Micro-Kernel Suites..."
	g++ -O3 -mavx512f tests/test_cycle21_kernels.cpp -o tests/test_cycle21_kernels
	./tests/test_cycle21_kernels

formal:
	@echo "[FORMAL] Compiling Lean 4 Mathematical Invariants..."
	lake build FormalSuite

gate:
	@echo "[GATE] Running Full QA Gatekeeper..."
	bash scripts/verify_fast_gate.sh

suite-smoke:
	@echo "[SUITE] Executing WEFT Benchmark & Test Suite (Smoke Tier)..."
	.venv/bin/python run_weft_suite.py --tier smoke --device cpu
