# Quickstart Guide: Running Collaborative Research Targets with Dr. Kevin W. McCairn, Ph.D.

**Laboratory:** Motor Systems Neurobiology & Theoretical Neuroscience Laboratory / Stratum Research Group  
**Authors:** Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.  
**Repository:** WEFT Monorepo (`github/master`)

This document provides exact, copy-pasteable terminal instructions to reproduce, simulate, formally verify, and benchmark every neurobiological and motor systems target co-authored with Kevin W. McCairn, Ph.D. (Targets #368 through #385, Papers #177 through #194).

---

## 1. System Requirements & Environment Setup

Ensure the Python virtual environment and native compilers are ready:
```bash
# Verify Python 3.12+ and virtual environment
source .venv/bin/activate

# Verify g++ compiler with AVX2 & FMA capabilities
g++ --version

# Verify Lean 4 (v4.33.1) and Lake build system
elan --version && lake --version
```

---

## 2. Automated Lean 4 Formal Verification Audit

To verify all formal targets (#368 through #385) without speculative axioms or unproven tactics:
```bash
# Run automated verification engine and lake compilation
.venv/bin/python scripts/verify_all_leans.py
```
*Expected Result:*
- `Build completed successfully (385 jobs).`
- `0` unproven obligations across all co-authored neurobiology targets.

---

## 3. Step-by-Step Instructions to Run Each Target

### Cycle v340.46.0: Lateral Corticospinal Tract (LCST) M1 Betz Cell Drive (Target #368, Paper #177)
```bash
# 1. Run sensory-motor fractionation simulation
.venv/bin/python scripts/sim_corticospinal_fractionation.py

# 2. Compile and run AVX2 Betz cell microbenchmark
g++ -O3 -mavx2 -mfma tools/weft_bench_corticospinal_fractionation.cpp -o build/weft_bench_corticospinal_fractionation
./build/weft_bench_corticospinal_fractionation

# 3. Execute Pytest regression tests
.venv/bin/pytest tests/test_corticospinal_fractionation.py
```

---

### Cycle v340.47.0: Anterior Intraparietal (AIP) & F5 Visual Grasp Affordance (Target #369, Paper #178)
```bash
# 1. Run visual grasp preshaping simulation
.venv/bin/python scripts/sim_grasp_affordance.py

# 2. Compile and run AVX2 hand aperture microbenchmark
g++ -O3 -mavx2 -mfma tools/weft_bench_grasp_affordance.cpp -o build/weft_bench_grasp_affordance
./build/weft_bench_grasp_affordance

# 3. Execute Pytest regression tests
.venv/bin/pytest tests/test_grasp_affordance.py
```

---

### Cycle v340.48.0: Spinal Presynaptic PAD & Axo-Axonic Inflow Gating (Target #370, Paper #179)
```bash
# 1. Run presynaptic primary afferent depolarization simulation
.venv/bin/python scripts/sim_spinal_presynaptic_pad.py

# 2. Compile and run AVX2 GABAergic axo-axonic gating microbenchmark
g++ -O3 -mavx2 -mfma tools/weft_bench_spinal_presynaptic_pad.cpp -o build/weft_bench_spinal_presynaptic_pad
./build/weft_bench_spinal_presynaptic_pad

# 3. Execute Pytest regression tests
.venv/bin/pytest tests/test_spinal_presynaptic_pad.py
```

---

### Cycle v340.49.0: SMA Proper & Pre-SMA Motor Sequence Chunking (Target #371, Paper #180)
```bash
# 1. Run sequence chunking and ramp interval timing simulation
.venv/bin/python scripts/sim_sma_motor_sequence.py

# 2. Compile and run AVX2 motor chunking microbenchmark
g++ -O3 -mavx2 -mfma tools/weft_bench_sma_chunking.cpp -o build/weft_bench_sma_chunking
./build/weft_bench_sma_chunking

# 3. Execute Pytest regression tests
.venv/bin/pytest tests/test_sma_chunking.py
```

---

### Cycle v340.50.0: Somatosensory Area 3b/2 Cutaneous Slip & Grip Force (Target #372, Paper #181)
```bash
# 1. Run FA-I / FA-II tactile slip transduction simulation
.venv/bin/python scripts/sim_tactile_slip_grip.py

# 2. Compile and run AVX2 transcortical grip escalation microbenchmark
g++ -O3 -mavx2 -mfma tools/weft_bench_tactile_slip.cpp -o build/weft_bench_tactile_slip
./build/weft_bench_tactile_slip

# 3. Execute Pytest regression tests
.venv/bin/pytest tests/test_tactile_slip.py
```

---

### Cycle v340.51.0: Dentato-Thalamo-Cortical (DTC) Closed-Loop Synchronization (Target #373, Paper #182)
```bash
# 1. Run cerebellar dentate-VL thalamus-M1 phase-locking simulation
.venv/bin/python scripts/sim_dentato_thalamic_sync.py

# 2. Compile and run AVX2 DTC coherence microbenchmark
g++ -O3 -mavx2 -mfma tools/weft_bench_dentato_thalamic_sync.cpp -o build/weft_bench_dentato_thalamic_sync
./build/weft_bench_dentato_thalamic_sync

# 3. Execute Pytest regression tests
.venv/bin/pytest tests/test_dentato_thalamic_sync.py
```

---

### Cycle v340.52.0: Cortico-Subthalamic Hyperdirect Stop-Signal Pathway (Target #374, Paper #183)
```bash
# 1. Run monosynaptic STN excitation and pallidal clamping simulation
.venv/bin/python scripts/sim_subthalamic_hyperdirect_stop.py

# 2. Compile and run AVX2 hyperdirect stopping microbenchmark
g++ -O3 -mavx2 -mfma tools/weft_bench_subthalamic_hyperdirect_stop.cpp -o build/weft_bench_subthalamic_hyperdirect_stop
./build/weft_bench_subthalamic_hyperdirect_stop

# 3. Execute Pytest regression tests
.venv/bin/pytest tests/test_subthalamic_hyperdirect_stop.py
```

---

### Cycle v340.53.0: Posterior Parietal Area 5d / 7a Coordinate Transformations (Target #375, Paper #184)
```bash
# 1. Run eye-position gain fields & hand-centered reach vector simulation
.venv/bin/python scripts/sim_parietal_coordinate_transform.py

# 2. Compile and run AVX2 multimodal coordinate transformation microbenchmark
g++ -O3 -mavx2 -mfma tools/weft_bench_parietal_transform.cpp -o build/weft_bench_parietal_transform
./build/weft_bench_parietal_transform

# 3. Execute Pytest regression tests
.venv/bin/pytest tests/test_parietal_coordinate_transform.py
```

---

### Cycle v340.54.0: Reticulospinal Tract & Pontomedullary Postural Synergies (Target #376, Paper #185)
```bash
# 1. Run anticipatory postural adjustment (APA) center-of-mass simulation
.venv/bin/python scripts/sim_reticulospinal_postural_synergy.py

# 2. Compile and run AVX2 bilateral postural synergy microbenchmark
g++ -O3 -mavx2 -mfma tools/weft_bench_reticulospinal_synergy.cpp -o build/weft_bench_reticulospinal_synergy
./build/weft_bench_reticulospinal_synergy

# 3. Execute Pytest regression tests
.venv/bin/pytest tests/test_reticulospinal_postural_synergies.py
```

---

### Cycle v340.55.0: Vestibulo-Ocular Reflex (VOR) Three-Neuron Arc (Target #377, Paper #186)
```bash
# 1. Run semicircular canal velocity & retinal slip minimization simulation
.venv/bin/python scripts/sim_vor_retinal_slip.py

# 2. Compile and run AVX2 VOR gain microbenchmark
g++ -O3 -mavx2 -mfma tools/weft_bench_vor_gain.cpp -o build/weft_bench_vor_gain
./build/weft_bench_vor_gain

# 3. Execute Pytest regression tests
.venv/bin/pytest tests/test_vor_retinal_slip.py
```

---

### Cycle v340.56.0: Cerebellar Flocculus Purkinje Plasticity (Target #378, Paper #187)
```bash
# 1. Run climbing fiber error signaling & VOR gain adaptation simulation
.venv/bin/python scripts/sim_cerebellar_vor_gain_adaptation.py

# 2. Compile and run AVX2 climbing fiber LTD microbenchmark
g++ -O3 -mavx2 -mfma tools/weft_bench_climbing_fiber_ltd.cpp -o build/weft_bench_climbing_fiber_ltd
./build/weft_bench_climbing_fiber_ltd

# 3. Execute Pytest regression tests
.venv/bin/pytest tests/test_cerebellar_vor_adaptation.py
```

---

### Cycle v340.57.0: Superior Colliculus Saccadic Burst Generator (Target #379, Paper #188)
```bash
# 1. Run omnipause gating & main sequence kinematics simulation
.venv/bin/python scripts/sim_saccadic_burst_gating.py

# 2. Compile and run AVX2 saccadic burst microbenchmark
g++ -O3 -mavx2 -mfma tools/weft_bench_saccadic_burst.cpp -o build/weft_bench_saccadic_burst
./build/weft_bench_saccadic_burst

# 3. Execute Pytest regression tests
.venv/bin/pytest tests/test_saccadic_burst_gating.py
```

---

### Cycle v340.58.0: Horizontal Oculomotor Neural Integrator (Target #380, Paper #189)
```bash
# 1. Run NPH/MVN pulse-step & gaze-drift simulation
.venv/bin/python scripts/sim_neural_integrator_drift.py

# 2. Compile and run AVX2 pulse-step neural integrator microbenchmark
g++ -O3 -mavx2 -mfma tools/weft_bench_neural_integrator.cpp -o build/weft_bench_neural_integrator
./build/weft_bench_neural_integrator

# 3. Execute Pytest regression tests
.venv/bin/pytest tests/test_neural_integrator.py
```

---

### Cycle v340.59.0: Striatal Dopaminergic D1/D2 Opponent Gating (Target #381, Paper #190)
```bash
# 1. Run three-factor RPE plasticity & reversal learning simulation
.venv/bin/python scripts/sim_striatal_dopamine_rpe.py

# 2. Compile and run AVX2 opponent gating microbenchmark
g++ -O3 -mavx2 -mfma tools/weft_bench_striatal_d1_d2.cpp -o build/weft_bench_striatal_d1_d2
./build/weft_bench_striatal_d1_d2

# 3. Execute Pytest regression tests
.venv/bin/pytest tests/test_striatal_dopamine_gating.py
```

---

### Cycle v340.60.0: Substantia Nigra (SNr) Disinhibition & Thalamic Rebound (Target #382, Paper #191)
```bash
# 1. Run striatonigral pause & T-type Ca2+ rebound burst simulation
.venv/bin/python scripts/sim_nigrothalamic_gate.py

# 2. Compile and run AVX2 nigrothalamic disinhibition microbenchmark
g++ -O3 -mavx2 -mfma tools/weft_bench_nigrothalamic_gate.cpp -o build/weft_bench_nigrothalamic_gate
./build/weft_bench_nigrothalamic_gate

# 3. Execute Pytest regression tests
.venv/bin/pytest tests/test_nigrothalamic_gate.py
```

---

### Cycle v340.61.0: Globus Pallidus (GPe) Subpopulations & Beta Resonance (Target #383, Paper #192)
```bash
# 1. Run prototypic vs arkypallidal stop gating & beta resonance simulation
.venv/bin/python scripts/sim_stn_gpe_beta_resonance.py

# 2. Compile and run AVX2 GPe closed-loop resonator microbenchmark
g++ -O3 -mavx2 -mfma tools/weft_bench_stn_gpe_beta.cpp -o build/weft_bench_stn_gpe_beta
./build/weft_bench_stn_gpe_beta

# 3. Execute Pytest regression tests
.venv/bin/pytest tests/test_gpe_subpopulations.py
```

---

### Cycle v340.62.0: Pedunculopontine Nucleus (PPN) & Spinal Locomotor CPG (Target #384, Paper #193)
```bash
# 1. Run PPN locomotor initiation and Freezing of Gait simulation
.venv/bin/python scripts/sim_ppn_locomotor_gating.py

# 2. Compile and run AVX2 PPN locomotor CPG cadence microbenchmark
g++ -O3 -mavx2 -mfma tools/weft_bench_ppn_locomotor_cpg.cpp -o build/weft_bench_ppn_locomotor_cpg
./build/weft_bench_ppn_locomotor_cpg

# 3. Execute Pytest regression tests
.venv/bin/pytest tests/test_ppn_locomotor.py
```

---

### Cycle v340.63.0: Cuneiform Nucleus (CnF) Locomotor Escape & Asymmetric Gait Bifurcations (Target #385, Paper #194)
```bash
# 1. Run CnF escape locomotion and gait bifurcation simulation
.venv/bin/python scripts/sim_cnf_escape_locomotion.py

# 2. Compile and run AVX2 CnF escape kinematics microbenchmark
g++ -O3 -mavx2 -mfma tools/weft_bench_cnf_escape_locomotion.cpp -o build/weft_bench_cnf_escape_locomotion
./build/weft_bench_cnf_escape_locomotion

# 3. Execute Pytest regression tests
.venv/bin/pytest tests/test_cnf_escape_locomotion.py
```

---

## 4. Real-Time Interactive Visual HUDs (WEFT MSS Void)

All interactive React HUDs are committed to `tools/weft-mss-void/src/vs/workbench/contrib/void/browser/react/src/sidebar-tsx/`:
- `WeftLCSTFractionationHUD.tsx` (Digit Fractionation)
- `WeftGraspAffordanceHUD.tsx` (Visual Grasp Preshaping)
- `WeftPresynapticPADHUD.tsx` (Primary Afferent Depolarization)
- `WeftSMAMotorSequenceHUD.tsx` (Sequence Chunking)
- `WeftTactileSlipHUD.tsx` (Tactile Slip & Grip Escalation)
- `WeftDentatoThalamicSyncHUD.tsx` (DTC Closed-Loop Phase Synchronization)
- `WeftHyperdirectStopHUD.tsx` (Cortico-Subthalamic Hyperdirect Stopping)
- `WeftParietalCoordinateHUD.tsx` (Area 5/7 Coordinate Transformations)
- `WeftReticulospinalSynergyHUD.tsx` (Reticulospinal Postural APAs)
- `WeftVORRetinalSlipHUD.tsx` (Vestibulo-Ocular Reflex Optical Stabilization)
- `WeftCerebellarVORAdaptationHUD.tsx` (Cerebellar Purkinje VOR Learning)
- `WeftSaccadicGatingHUD.tsx` (Superior Colliculus Saccadic Burst Generator)
- `WeftNeuralIntegratorHUD.tsx` (Pulse-Step Gaze-Holding Integrator)
- `WeftStriatalDopamineHUD.tsx` (D1/D2 Opponent Action Selection)
- `WeftNigrothalamicDisinhibitionHUD.tsx` (SNr Disinhibition & Thalamic Rebound)
- `WeftGPeArkypallidalHUD.tsx` (GPe Prototypic vs Arkypallidal Stopping & Beta Resonance)
- `WeftPPNLocomotorHUD.tsx` (PPN Locomotor CPG Phase Gating & Freezing of Gait)
- `WeftCnFEscapeHUD.tsx` (CnF High-Speed Locomotor Escape & Gallop Bifurcation)
