# Quickstart Guide: Running Collaborative Research Targets with Dr. Kevin W. McCairn, Ph.D.

**Laboratory:** Motor Systems Neurobiology & Theoretical Neuroscience Laboratory / Stratum Research Group  
**Authors:** Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.  
**Repository:** WEFT Monorepo (`github/master`)

This document provides exact, copy-pasteable terminal instructions to reproduce, simulate, formally verify, and benchmark every neurobiological and motor systems target co-authored with Kevin W. McCairn, Ph.D. (Targets #368 through #415, Papers #177 through #224).

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

To verify all formal targets (#368 through #415) without speculative axioms or unproven tactics:
```bash
# Run automated verification engine and lake compilation
.venv/bin/python scripts/verify_all_leans.py
```
*Expected Result:*
- `Build completed successfully (415 jobs).`
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

### Cycle v340.64.0: Locus Coeruleus Norepinephrine (LC-NE) Gain Modulation & Yerkes-Dodson Arousal Invariants (Target #386, Paper #195)
```bash
# 1. Run LC-NE arousal regime and cortical gain simulation
.venv/bin/python scripts/sim_lc_ne_modulation.py

# 2. Compile and run AVX2 LC-NE cortical gain microbenchmark
g++ -O3 -mavx2 -mfma tools/weft_bench_lc_ne_modulation.cpp -o build/weft_bench_lc_ne_modulation
./build/weft_bench_lc_ne_modulation

# 3. Execute Pytest regression tests
.venv/bin/pytest tests/test_lc_ne_modulation.py
```

---

### Cycle v340.65.0: Dorsal Raphe Serotonin (5-HT) Waiting Patience & Impulsivity Threshold Invariants (Target #387, Paper #196)
```bash
# 1. Run dorsal raphe 5-HT patience and delay discounting simulation
.venv/bin/python scripts/sim_dorsal_raphe_5ht.py

# 2. Compile and run AVX2 5-HT temporal discounting microbenchmark
g++ -O3 -mavx2 -mfma tools/weft_bench_dorsal_raphe_5ht.cpp -o build/weft_bench_dorsal_raphe_5ht
./build/weft_bench_dorsal_raphe_5ht

# 3. Execute Pytest regression tests
.venv/bin/pytest tests/test_dorsal_raphe_5ht.py
```

---

### Cycle v340.66.0: Basal Forebrain Acetylcholine (BF-ACh) Attention Cue Detection & Expected Uncertainty Invariants (Target #388, Paper #197)
```bash
# 1. Run basal forebrain ACh attentional gating simulation
.venv/bin/python scripts/sim_basal_forebrain_ach.py

# 2. Compile and run AVX2 ACh attentional gating microbenchmark
g++ -O3 -mavx2 -mfma tools/weft_bench_basal_forebrain_ach.cpp -o build/weft_bench_basal_forebrain_ach
./build/weft_bench_basal_forebrain_ach

# 3. Execute Pytest regression tests
.venv/bin/pytest tests/test_basal_forebrain_ach.py
```

---

### Cycle v340.67.0: Lateral Habenula (LHb) Anti-Reward Circuitry & Midbrain Dopaminergic Pause Invariants (Target #389, Paper #198)
```bash
# 1. Run lateral habenula negative RPE and dopamine pause simulation
.venv/bin/python scripts/sim_lateral_habenula.py

# 2. Compile and run AVX2 LHb anti-reward microbenchmark
g++ -O3 -mavx2 -mfma tools/weft_bench_lateral_habenula.cpp -o build/weft_bench_lateral_habenula
./build/weft_bench_lateral_habenula

# 3. Execute Pytest regression tests
.venv/bin/pytest tests/test_lateral_habenula.py
```

---

### Cycle v340.68.0: Ventral Tegmental Area (VTA) Dopamine Plasticity & Three-Factor Learning Invariants (Target #390, Paper #199)
```bash
# 1. Run VTA dopamine-gated three-factor synaptic plasticity simulation
.venv/bin/python scripts/sim_vta_dopamine_plasticity.py

# 2. Compile and run AVX2 three-factor synaptic plasticity microbenchmark
g++ -O3 -mavx2 -mfma tools/weft_bench_vta_dopamine_plasticity.cpp -o build/weft_bench_vta_dopamine_plasticity
./build/weft_bench_vta_dopamine_plasticity

# 3. Execute Pytest regression tests
.venv/bin/pytest tests/test_vta_dopamine_plasticity.py
```

---

### Milestone Cycle v340.69.0: Central Neuromodulatory Convergence & Multi-Transmitter Motor Arbitration (Target #391, Milestone Paper #200)
```bash
# 1. Run 5-axis neuromodulatory convergence simulation
.venv/bin/python scripts/sim_neuromodulatory_synthesis.py

# 2. Compile and run AVX2 5-transmitter synthesis microbenchmark
g++ -O3 -mavx2 -mfma tools/weft_bench_neuromodulatory_synthesis.cpp -o build/weft_bench_neuromodulatory_synthesis
./build/weft_bench_neuromodulatory_synthesis

# 3. Execute Pytest regression tests
.venv/bin/pytest tests/test_neuromodulatory_synthesis.py
```

---

### Cycle v340.70.0: Thalamic Reticular Nucleus (TRN) Sensory Attentional Filtering & Burst-Tonic Invariants (Target #392, Paper #201)
```bash
# 1. Run TRN cross-modal attentional filtering simulation
.venv/bin/python scripts/sim_thalamic_reticular_filtering.py

# 2. Compile and run AVX2 TRN cross-modal gating microbenchmark
g++ -O3 -mavx2 -mfma tools/weft_bench_thalamic_reticular_filtering.cpp -o build/weft_bench_thalamic_reticular_filtering
./build/weft_bench_thalamic_reticular_filtering

# 3. Execute Pytest regression tests
.venv/bin/pytest tests/test_thalamic_reticular_filtering.py
```

---

### Cycle v340.71.0: Claustrum Cortical Synchronization & Salience Switching Invariants (Target #393, Paper #202)
```bash
# 1. Run claustrum cortical synchronization and salience switching simulation
.venv/bin/python scripts/sim_claustrum_synchronization.py

# 2. Compile and run AVX2 claustrum salience gating microbenchmark
g++ -O3 -mavx2 -mfma tools/weft_bench_claustrum_synchronization.cpp -o build/weft_bench_claustrum_synchronization
./build/weft_bench_claustrum_synchronization

# 3. Execute Pytest regression tests
.venv/bin/pytest tests/test_claustrum_synchronization.py
```

---

### Cycle v340.72.0: Zona Incerta (ZI) Visceromotor Integration & Binge Feeding Gating Invariants (Target #394, Paper #203)
```bash
# 1. Run Zona Incerta visceromotor integration simulation
.venv/bin/python scripts/sim_zona_incerta.py

# 2. Compile and run AVX2 Zona Incerta visceromotor gate microbenchmark
g++ -O3 -mavx2 -mfma tools/weft_bench_zona_incerta.cpp -o build/weft_bench_zona_incerta
./build/weft_bench_zona_incerta

# 3. Execute Pytest regression tests
.venv/bin/pytest tests/test_zona_incerta.py
```

---

### Cycle v340.73.0: Ventrolateral Periaqueductal Gray (vlPAG) Freezing vs Fight/Flight Action Selection Invariants (Target #395, Paper #204)
```bash
# 1. Run vlPAG freezing vs dlPAG escape simulation
.venv/bin/python scripts/sim_vlpag_survival.py

# 2. Compile and run AVX2 vlPAG defense arbitration microbenchmark
g++ -O3 -mavx2 -mfma tools/weft_bench_vlpag_survival.cpp -o build/weft_bench_vlpag_survival
./build/weft_bench_vlpag_survival

# 3. Execute Pytest regression tests
.venv/bin/pytest tests/test_vlpag_survival.py
```

---

### Cycle v340.74.0: Lateral Hypothalamus (LH) Orexin/Hypocretin Arousal & Muscle Tone Gating Invariants (Target #396, Paper #205)
```bash
# 1. Run LH orexin arousal and cataplexy prevention simulation
.venv/bin/python scripts/sim_lh_orexin.py

# 2. Compile and run AVX2 LH orexin muscle tone microbenchmark
g++ -O3 -mavx2 -mfma tools/weft_bench_lh_orexin.cpp -o build/weft_bench_lh_orexin
./build/weft_bench_lh_orexin

# 3. Execute Pytest regression tests
.venv/bin/pytest tests/test_lh_orexin.py
```

---

### Cycle v340.75.0: Electrophysiological Disease State Prediction: LFP Beta Resonance, PAC & 1/f Spectral Tilt Invariants (Target #397, Paper #206)
```bash
# 1. Run electrophysiological disease state prediction simulation
.venv/bin/python scripts/sim_disease_prediction.py

# 2. Compile and run AVX2 electrophysiological disease prediction microbenchmark
g++ -O3 -mavx2 -mfma tools/weft_bench_disease_prediction.cpp -o build/weft_bench_disease_prediction
./build/weft_bench_disease_prediction

# 3. Execute Pytest regression tests
.venv/bin/pytest tests/test_disease_prediction.py
```

---

### Cycle v340.76.0: Striatal Cholinergic Interneuron (ChI) Hyper-Rebound & Dystonic Co-Contraction Invariants (Target #398, Paper #207)
```bash
# 1. Run striatal cholinergic hyper-rebound dystonia simulation
.venv/bin/python scripts/sim_striatal_cholinergic.py

# 2. Compile and run AVX2 striatal cholinergic microbenchmark
g++ -O3 -mavx2 -mfma tools/weft_bench_striatal_cholinergic.cpp -o build/weft_bench_striatal_cholinergic
./build/weft_bench_striatal_cholinergic

# 3. Execute Pytest regression tests
.venv/bin/pytest tests/test_striatal_cholinergic.py
```

---

### Cycle v340.77.0: Adaptive Closed-Loop DBS (aDBS): Phase-Specific Desynchronization & Beta-Burst Truncation Invariants (Target #399, Paper #208)
```bash
# 1. Run adaptive closed-loop aDBS simulation
.venv/bin/python scripts/sim_adaptive_dbs.py

# 2. Compile and run AVX2 aDBS closed-loop microbenchmark
g++ -O3 -mavx2 -mfma tools/weft_bench_adaptive_dbs.cpp -o build/weft_bench_adaptive_dbs
./build/weft_bench_adaptive_dbs

# 3. Execute Pytest regression tests
.venv/bin/pytest tests/test_adaptive_dbs.py
```

---

### Cycle v340.78.0: Huntington's Disease Striatal D2-SPN Degeneration & Hyperkinetic Chorea Invariants (Milestone Target #400, Paper #209)
```bash
# 1. Run Huntington chorea & indirect pathway collapse simulation
.venv/bin/python scripts/sim_huntington_chorea.py

# 2. Compile and run AVX2 Huntington chorea microbenchmark
g++ -O3 -mavx2 -mfma tools/weft_bench_huntington_chorea.cpp -o build/weft_bench_huntington_chorea
./build/weft_bench_huntington_chorea

# 3. Execute Pytest regression tests
.venv/bin/pytest tests/test_huntington_chorea.py
```

---

### Cycle v340.79.0: Amyotrophic Lateral Sclerosis (ALS) Corticomotoneuronal Excitotoxicity & Renshaw Recurrent Disinhibition Invariants (Target #401, Paper #210)
```bash
# 1. Run ALS excitotoxicity & Renshaw recurrent disinhibition simulation
.venv/bin/python scripts/sim_als_excitotoxicity.py

# 2. Compile and run AVX2 ALS excitotoxicity microbenchmark
g++ -O3 -mavx2 -mfma tools/weft_bench_als_excitotoxicity.cpp -o build/weft_bench_als_excitotoxicity
./build/weft_bench_als_excitotoxicity

# 3. Execute Pytest regression tests
.venv/bin/pytest tests/test_als_excitotoxicity.py
```

---

### Cycle v340.80.0: Essential Tremor (ET) Inferior Olivary Pacemaking & DRT Hyper-Oscillation Invariants (Target #402, Paper #211)
```bash
# 1. Run Essential Tremor inferior olivary pacemaking simulation
.venv/bin/python scripts/sim_essential_tremor.py

# 2. Compile and run AVX2 Essential Tremor microbenchmark
g++ -O3 -mavx2 -mfma tools/weft_bench_essential_tremor.cpp -o build/weft_bench_essential_tremor
./build/weft_bench_essential_tremor

# 3. Execute Pytest regression tests
.venv/bin/pytest tests/test_essential_tremor.py
```

---

### Cycle v340.81.0: Alzheimer’s Disease Corticocortical Gamma Uncoupling & Parvalbumin Hypoactivity Invariants (Target #403, Paper #212)
```bash
# 1. Run Alzheimer corticocortical gamma & PV hypoactivity simulation
.venv/bin/python scripts/sim_alzheimer_gamma.py

# 2. Compile and run AVX2 Alzheimer gamma microbenchmark
g++ -O3 -mavx2 -mfma tools/weft_bench_alzheimer_gamma.cpp -o build/weft_bench_alzheimer_gamma
./build/weft_bench_alzheimer_gamma

# 3. Execute Pytest regression tests
.venv/bin/pytest tests/test_alzheimer_gamma.py
```

---

### Cycle v340.82.0: Paroxysmal Dyskinesia & Channelopathy: Cerebellar-Basal Ganglia Crosstalk Invariants (Target #404, Paper #213)
```bash
# 1. Run Paroxysmal Dyskinesia cerebellar-striatal simulation
.venv/bin/python scripts/sim_paroxysmal_dyskinesia.py

# 2. Compile and run AVX2 Paroxysmal Dyskinesia microbenchmark
g++ -O3 -mavx2 -mfma tools/weft_bench_paroxysmal_dyskinesia.cpp -o build/weft_bench_paroxysmal_dyskinesia
./build/weft_bench_paroxysmal_dyskinesia

# 3. Execute Pytest regression tests
.venv/bin/pytest tests/test_paroxysmal_dyskinesia.py
```

---

### Cycle v340.83.0: Unified Closed-Loop Neuromodulation Engine: Multi-Modal Biomarker Fusion Invariants (Target #405, Paper #214)
```bash
# 1. Run Unified Closed-Loop Neuromodulation Engine simulation
.venv/bin/python scripts/sim_unified_neuromodulation.py

# 2. Compile and run AVX2 Unified Neuromodulation microbenchmark
g++ -O3 -mavx2 -mfma tools/weft_bench_unified_neuromodulation.cpp -o build/weft_bench_unified_neuromodulation
./build/weft_bench_unified_neuromodulation

# 3. Execute Pytest regression tests
.venv/bin/pytest tests/test_unified_neuromodulation.py
```

---

### Cycle v340.84.0: Fibrinogen-SHED-CM Molecular Interactions & Anti-Amyloid Fibrin Shielding Invariants (Target #406, Paper #215)
```bash
# 1. Run Fibrinogen-SHED-CM molecular docking and shielding simulation
.venv/bin/python scripts/sim_fibrinogen_shed.py

# 2. Compile and run AVX2 Fibrinogen-SHED microbenchmark
g++ -O3 -mavx2 -mfma tools/weft_bench_fibrinogen_shed.cpp -o build/weft_bench_fibrinogen_shed
./build/weft_bench_fibrinogen_shed

# 3. Execute Pytest regression tests
.venv/bin/pytest tests/test_fibrinogen_shed.py
```

---

### Cycle v340.85.0: Blood-Brain Barrier (BBB) Endothelial Tight-Junction Restoration & Pericyte Tie2 Signaling Invariants (Target #407, Paper #216)
```bash
# 1. Run BBB tight-junction and Tie2 restoration simulation
.venv/bin/python scripts/sim_bbb_tight_junction.py

# 2. Compile and run AVX2 BBB tight-junction microbenchmark
g++ -O3 -mavx2 -mfma tools/weft_bench_bbb_tight_junction.cpp -o build/weft_bench_bbb_tight_junction
./build/weft_bench_bbb_tight_junction

# 3. Execute Pytest regression tests
.venv/bin/pytest tests/test_bbb_tight_junction.py
```

---

### Cycle v340.86.0: Systemic Vascular Microclotting Under SARS-CoV-2 S1 Spike & SHED-CM Secretome Invariants (Target #408, Paper #217)
```bash
# 1. Run SARS-CoV-2 S1 microclotting and SHED rescue simulation
.venv/bin/python scripts/sim_s1_microclotting.py

# 2. Compile and run AVX2 S1 microclotting microbenchmark
g++ -O3 -mavx2 -mfma tools/weft_bench_s1_microclotting.cpp -o build/weft_bench_s1_microclotting
./build/weft_bench_s1_microclotting

# 3. Execute Pytest regression tests
.venv/bin/pytest tests/test_s1_microclotting.py
```

---

### Cycle v340.87.0: Microglial Mac-1 (CD11b/CD18) Cryptic Epitope Neutralization & NLRP3 Inflammasome Gating Invariants (Target #409, Paper #218)
```bash
# 1. Run microglial Mac-1 and NLRP3 inflammasome simulation
.venv/bin/python scripts/sim_microglial_mac1_inflammasome.py

# 2. Compile and run AVX2 microglial inflammasome microbenchmark
g++ -O3 -mavx2 -mfma tools/weft_bench_microglial_mac1_inflammasome.cpp -o build/weft_bench_microglial_mac1_inflammasome
./build/weft_bench_microglial_mac1_inflammasome

# 3. Execute Pytest regression tests
.venv/bin/pytest tests/test_microglial_mac1_inflammasome.py
```

---

### Cycle v340.88.0: Glymphatic Convective Flux & Aquaporin-4 (AQP4) Polarized Perivascular Clearance Invariants (Target #410, Paper #219)
```bash
# 1. Run Glymphatic flux and AQP4 clearance simulation
.venv/bin/python scripts/sim_glymphatic_aqp4.py

# 2. Compile and run AVX2 Glymphatic AQP4 microbenchmark
g++ -O3 -mavx2 -mfma tools/weft_bench_glymphatic_aqp4.cpp -o build/weft_bench_glymphatic_aqp4
./build/weft_bench_glymphatic_aqp4

# 3. Execute Pytest regression tests
.venv/bin/pytest tests/test_glymphatic_aqp4.py
```

---

### Cycle v340.89.0: Neurovascular Coupling & BK_Ca Vasomotor Invariants (Target #411, Paper #220)
```bash
# 1. Run neurovascular coupling simulation
.venv/bin/python scripts/sim_neurovascular_coupling.py

# 2. Compile and run AVX2 neurovascular microbenchmark
g++ -O3 -mavx2 -mfma tools/weft_bench_neurovascular_coupling.cpp -o build/weft_bench_neurovascular_coupling
./build/weft_bench_neurovascular_coupling

# 3. Execute Pytest regression tests
.venv/bin/pytest tests/test_neurovascular_coupling.py
```

---

### Cycle v340.90.0: Pericyte Capillary Constriction & Ischemic No-Reflow Invariants (Target #412, Paper #221)
```bash
# 1. Run pericyte capillary constriction and no-reflow simulation
.venv/bin/python scripts/sim_pericyte_no_reflow.py

# 2. Compile and run AVX2 pericyte microbenchmark
g++ -O3 -mavx2 -mfma tools/weft_bench_pericyte_no_reflow.cpp -o build/weft_bench_pericyte_no_reflow
./build/weft_bench_pericyte_no_reflow

# 3. Execute Pytest regression tests
.venv/bin/pytest tests/test_pericyte_no_reflow.py
```

---

### Cycle v340.91.0: Endothelial Shear-Stress Mechanotransduction & eNOS Phosphorylation Invariants (Target #413, Paper #222)
```bash
# 1. Run endothelial shear-stress simulation
.venv/bin/python scripts/sim_shear_stress.py

# 2. Compile and run AVX2 shear-stress microbenchmark
g++ -O3 -mavx2 -mfma tools/weft_bench_shear_stress.cpp -o build/weft_bench_shear_stress
./build/weft_bench_shear_stress

# 3. Execute Pytest regression tests
.venv/bin/pytest tests/test_shear_stress.py
```

---

### Cycle v340.92.0: OPC Differentiation & Saltatory Conduction Velocity Invariants (Target #414, Paper #223)
```bash
# 1. Run OPC differentiation simulation
.venv/bin/python scripts/sim_opc_differentiation.py

# 2. Compile and run AVX2 OPC microbenchmark
g++ -O3 -mavx2 -mfma tools/weft_bench_opc_differentiation.cpp -o build/weft_bench_opc_differentiation
./build/weft_bench_opc_differentiation

# 3. Execute Pytest regression tests
.venv/bin/pytest tests/test_opc_differentiation.py
```

---

### Cycle v340.93.0: Presynaptic Mitochondrial Fission-Fusion Balance & Synaptosomal ATP Invariants (Target #415, Paper #224)
```bash
# 1. Run presynaptic mitochondrial dynamics simulation
.venv/bin/python scripts/sim_mitochondrial_dynamics.py

# 2. Compile and run AVX2 mitochondrial microbenchmark
g++ -O3 -mavx2 -mfma tools/weft_bench_mitochondrial_dynamics.cpp -o build/weft_bench_mitochondrial_dynamics
./build/weft_bench_mitochondrial_dynamics

# 3. Execute Pytest regression tests
.venv/bin/pytest tests/test_mitochondrial_dynamics.py
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
- `WeftLCNEModulationHUD.tsx` (Locus Coeruleus Noradrenergic Gain & Yerkes-Dodson Arousal)
- `WeftDorsalRaphe5HTHUD.tsx` (Dorsal Raphe Serotonin Waiting Patience & Impulsivity Threshold)
- `WeftBasalForebrainAChHUD.tsx` (Basal Forebrain Cholinergic Attentional Gating & Cue Detection)
- `WeftLateralHabenulaHUD.tsx` (Lateral Habenula Anti-Reward Signaling & Dopaminergic Pause)
- `WeftVTADopaminePlasticityHUD.tsx` (VTA Dopaminergic Three-Factor Synaptic Plasticity & Trace Consolidation)
- `WeftNeuromodulatorySynthesisHUD.tsx` (Grand 5-Axis Neuromodulatory Convergence & Multi-Transmitter Synthesis)
- `WeftThalamicReticularHUD.tsx` (Thalamic Reticular Cross-Modal Attentional Filter & Burst-Tonic Gate)
- `WeftClaustrumSynchronizationHUD.tsx` (Claustrum Cortical Synchronization & Salience Switching Gate)
- `WeftZonaIncertaHUD.tsx` (Zona Incerta Visceromotor Gate & Binge Feeding Attractor)
- `WeftVLPAGSurvivalHUD.tsx` (vlPAG Freezing vs dlPAG Flight Escape Survival Gate)
- `WeftLHOrexinArousalHUD.tsx` (LH Orexin Arousal & Muscle Tone Stabilizer)
- `WeftElectrophysiologicalDiseasePredictionHUD.tsx` (LFP Beta Power, PAC & 1/f Spectral Tilt Predictive Biomarker Gate)
- `WeftStriatalCholinergicDystoniaHUD.tsx` (ChI Hyper-Rebound & Agonist-Antagonist Co-Contraction Gate)
- `WeftAdaptiveDBSClosedLoopHUD.tsx` (aDBS Phase-Specific Desynchronization & Beta Burst Truncation Gate)
- `WeftHuntingtonChoreaHUD.tsx` (Striatal D2-SPN Collapse, GPe Disinhibition & Chorea Intrusion Gate)
- `WeftAlsCorticomotoneuronalHUD.tsx` (Astrocytic EAAT2 Loss, Motoneuronal Runaway Firing & Renshaw Recurrent Disinhibition Gate)
- `WeftEssentialTremorHUD.tsx` (Inferior Olivary Pacemaking, DRT Hyper-Oscillation & Hand Tremor Gate)
- `WeftAlzheimerGammaHUD.tsx` (PV Interneuron Nav1.1 Hypofunction, Cortical Gamma Uncoupling & Sensory Rescue Gate)
- `WeftParoxysmalDyskinesiaHUD.tsx` (PRRT2/PNKD SNARE Channelopathy, Cerebellar-Striatal Crosstalk & Carbamazepine Gate)
- `WeftUnifiedNeuromodulationHUD.tsx` (Multi-Modal Biomarker Fusion, Closed-Loop Proportional DBS & Autonomous Rescue Gate)
- `WeftFibrinogenShedHUD.tsx` (SHED-CM Chaperone Shielding, Fibrinogen-Aβ Epitope Masking & Mac-1 Suppression Gate)
- `WeftBbbTightJunctionHUD.tsx` (Pericyte-Endothelial Tie2 Phosphorylation, Claudin-5/Occludin Assembly & TEER Restoration Gate)
- `WeftS1MicroclottingHUD.tsx` (SARS-CoV-2 S1 Spike-Fibrinogen Amyloid Microclots, ULVWF Multimerization & SHED Rescue Gate)
- `WeftMicroglialMac1InflammasomeHUD.tsx` (Microglial Mac-1 Ligation, NLRP3 ASC Speck Assembly & SHED miR-124 M2 Resolution Gate)
- `WeftGlymphaticAqp4HUD.tsx` (Astrocytic Endfoot AQP4 Polarized Water Transport, NREM Vasomotion & Glymphatic Flush Gate)
- `WeftNeurovascularCouplingHUD.tsx` (Astrocytic Endfoot Ca2+ Transients, BK_Ca/Kir2.1 Vasomotion & Functional Hyperemia Gate)
- `WeftPericyteNoReflowHUD.tsx` (Capillary Pericyte Contraction, RBC Entrapment & SHED Tie2/cAMP Reflow Rescue Gate)
- `WeftShearStressMechanotransductionHUD.tsx` (Endothelial Glycocalyx Shear Sensing, eNOS Ser1177 Phos & NO Output Gate)
- `WeftOpcDifferentiationHUD.tsx` (OPC Migration, GPR17 Downregulation, MBP Compaction & Saltatory Velocity Gate)
- `WeftPresynapticMitochondriaHUD.tsx` (Drp1/Mfn2 Ratio, Delta Psi_m, Synaptosomal ATP & mPTP Gate)
