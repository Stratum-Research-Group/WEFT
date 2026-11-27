.PHONY: test test-unit test-regression test-cpp formal gate suite-smoke suite-standard audit matrix lint telemetry search paclet standalone-reader bench-triton eval-quant export-pdfs bind-monograph citation-graph validate-models sign-release verify-vectors package-bundle

test: test-unit test-regression test-cpp
	@echo "================================================================="
	@echo "  ALL WEFT TEST SUITES PASSED"
	@echo "================================================================="

test-unit:
	@echo "[TEST] Running Python Core Unit Tests..."
	.venv/bin/pytest tests/test_avx512_gemv.py tests/test_fused_swiglu_avx512.py tests/test_triton_recurrent_autograd.py -q

test-regression:
	@echo "[TEST] Running Regression Guardrails T1-T6..."
	.venv/bin/pytest tests/test_regression_guardrails.py tests/test_packed_vs_dense.py tests/test_gate_and_scaling.py -q

test-cpp:
	@echo "[TEST] Running Native C++ Micro-Kernel Suites..."
	g++ -O3 -mavx512f tests/test_cycle21_kernels.cpp -o tests/test_cycle21_kernels
	./tests/test_cycle21_kernels
	g++ -O3 -mavx512f tests/test_avx512_recurrent_engine.cpp -o tests/test_avx512_recurrent_engine
	./tests/test_avx512_recurrent_engine

formal:
	@echo "[FORMAL] Compiling Lean 4 Mathematical Invariants..."
	lake build FormalSuite

audit:
	@echo "[AUDIT] Auditing 95 Scientific Manuscripts and Formal Lean 4 Suite..."
	python3 scripts/audit_manuscripts.py
	python3 scripts/audit_lean_axioms.py

matrix:
	@echo "[MATRIX] Validating 1:1 Cross-Verification Parity & Figures..."
	python3 scripts/validate_matrix.py

lint:
	@echo "[LINT] Linting Manuscript Markdown, KaTeX & Link Integrity..."
	python3 scripts/lint_manuscripts.py

telemetry:
	@echo "[TELEMETRY] Rendering Publication Visualizations..."
	.venv/bin/python scripts/generate_telemetry_visualizations.py

search:
	@echo "[SEARCH] Usage: python3 tools/weft_corpus_search.py <query>"
	@python3 tools/weft_corpus_search.py "purkinje"

paclet:
	@echo "[PACLET] Packaging Wolfram WEFT.paclet..."
	tar -czf dist/WEFT-4.0.0.paclet -C wolfram/Paclet/WEFT .

standalone-reader:
	@echo "[READER] Building Standalone Offline Web Reader..."
	python3 scripts/build_standalone_reader.py

bench-triton:
	@echo "[BENCH] Running Triton 3.9 Biophysical Micro-Kernel Benchmarks..."
	.venv/bin/python scripts/bench_triton_biophysical.py

eval-quant:
	@echo "[EVAL] Running Quantization Drift & Cosine Retention Analysis..."
	.venv/bin/python scripts/evaluate_quantization_drift.py

export-pdfs:
	@echo "[PDF] Exporting All 95 Technical Reports to Individual PDFs..."
	.venv/bin/python scripts/export_manuscript_pdf.py --all -j 12

bind-monograph:
	@echo "[BIND] Binding All 95 Reports into Master Monograph Volume IV PDF..."
	python3 scripts/bind_monograph_volume4.py

citation-graph:
	@echo "[GRAPH] Building Co-Citation D3 Network..."
	python3 scripts/build_citation_graph.py

validate-models:
	@echo "[MODELS] Validating Sub-Bit Model Registry & Checkpoint Parity..."
	python3 scripts/validate_model_registry.py

sign-release:
	@echo "[SIGN] Generating Cryptographic SHA-256 and SHA-512 Checksums..."
	python3 scripts/sign_release_artifacts.py

verify-vectors:
	@echo "[VECTORS] Auditing XML and Geometric Fidelity of Vector SVGs..."
	python3 scripts/verify_vector_fidelity.py

package-bundle:
	@echo "[BUNDLE] Creating Master Academic Submission Bundle..."
	python3 scripts/package_distribution_bundle.py

gate:
	@echo "[GATE] Running Full QA Gatekeeper..."
	bash scripts/verify_fast_gate.sh

suite-smoke:
	@echo "[SUITE] Executing WEFT Benchmark & Test Suite (Smoke Tier)..."
	.venv/bin/python run_weft_suite.py --tier smoke --device cpu

suite-standard:
	@echo "[SUITE] Executing WEFT Benchmark & Test Suite (Standard Tier)..."
	.venv/bin/python run_weft_suite.py --tier standard --device cpu

validate-bibtex:
	@echo "[BIBTEX] Validating Master Deduplicated Bibliography..."
	python3 scripts/validate_bibtex_citations.py

profile-triton-sram:
	@echo "[TRITON] Profiling Shared Memory Allocation and Bank Conflicts..."
	python3 scripts/profile_triton_sram.py

generate-lean-badge:
	@echo "[LEAN] Generating Lean 4 Formal Verification Badge..."
	python3 scripts/generate_lean_badge.py

validate-hyperlinks:
	@echo "[LEAN-LINKS] Validating Formal Lean 4 Manuscript Proof Anchors..."
	python3 scripts/validate_lean_hyperlinks.py

export-triton-configs:
	@echo "[TRITON-CONFIGS] Exporting SM90/SM100 Hopper & Blackwell Configurations..."
	python3 scripts/export_triton_autotune_configs.py

check-cpu-vnni:
	@echo "[CPU-VNNI] Checking Native AVX-512 VNNI / VPOPCNTDQ Capabilities..."
	python3 scripts/check_cpu_vnni.py

bench-avx512:
	@echo "[BENCH-AVX512] Benchmarking Native AVX-512 VPOPCNTDQ Sub-Bit Bitboard GEMV..."
	python3 scripts/benchmark_avx512_gemv.py
