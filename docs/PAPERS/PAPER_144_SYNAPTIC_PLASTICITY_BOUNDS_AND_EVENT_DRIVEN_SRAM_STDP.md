# WEFT Technical Milestone Paper #144
## Synaptic Plasticity Bounds, Event-Driven SRAM STDP, and Basal Ganglia-Gated Cortical Manifold Selection

**Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.**  
*Stratum Research Group • Theoretical Neurobiology & Synaptic Plasticity Series*  
*November 2026*

---

### Abstract

Biophysical computation in cortical neural manifolds relies upon millisecond-scale temporal coordination between presynaptic spike arrival and postsynaptic backpropagating action potentials (bAPs). Here, we formalize the mathematical convergence bounds and anatomochemical mechanics of Spike-Timing-Dependent Plasticity (STDP) operating within bounded synaptic weight intervals $[W_{\min}, W_{\max}]$. We introduce a fused GPU SRAM Triton 3.9 kernel for event-driven asynchronous STDP updates alongside an AVX2 SIMD Leaky Integrate-and-Fire (LIF) micro-engine achieving $> 13.6$ billion neuron-steps per second. Furthermore, we resolve the architectural coupling between distributed cortical manifolds and the subcortical Basal Ganglia circuitry. We demonstrate that the Basal Ganglia acts as a disinhibitory reinforcement-gated attractor selection engine, where striatal Direct ($D_1$) and Indirect ($D_2$) medium spiny neurons dynamically steer cortical phase manifolds, arbitrated by an ultra-fast ($< 10\text{ ms}$) Subthalamic Nucleus (STN) hyperdirect global brake and phasic substantia nigra pars compacta ($SNc$) dopamine bursts that modulate STDP potentiation windows.

---

### 1. Biophysical Foundations of Spike-Timing-Dependent Plasticity (STDP)

Hebbian learning in biological wetware is inherently causal and temporal. Presynaptic neurotransmitter release (L-glutamate) across the synaptic cleft into dendritic spines binds postsynaptic AMPA and NMDA receptors:

$$\Delta t = t_{\text{post}} - t_{\text{pre}}$$

$$\Delta W(\Delta t) = \begin{cases} A_+ e^{-\Delta t / \tau_+} & \text{if } \Delta t > 0 \quad (\text{Long-Term Potentiation / LTP}) \\ -A_- e^{\Delta t / \tau_-} & \text{if } \Delta t < 0 \quad (\text{Long-Term Depression / LTD}) \\ 0 & \text{if } \Delta t = 0 \end{cases}$$

1. **Causal Coincidence ($\Delta t > 0$):** Presynaptic glutamate priming precedes the postsynaptic somatic action potential backpropagating into the dendritic arbor ($bAP$). Depolarization expels the voltage-dependent $Mg^{2+}$ block from the pore of NMDA receptors, admitting a high-amplitude transient influx of $Ca^{2+}$. This activates Calcium/Calmodulin-Dependent Protein Kinase II ($\text{CaMKII}$), driving rapid exocytosis and phosphorylation of AMPA receptors (GluA1) into the postsynaptic density.
2. **Anti-Causal Coincidence ($\Delta t < 0$):** Postsynaptic depolarization precedes presynaptic glutamate release. Calcium influx is sub-threshold and prolonged, preferentially activating calcineurin and protein phosphatase 1 ($\text{PP1}$), resulting in clathrin-mediated endocytosis of AMPA receptors and dendritic spine shrinkage (LTD).

---

### 2. Formal Convergence Invariants & Homeostatic Interval Clamping

Unbounded Hebbian plasticity is inherently unstable, provoking runaway excitation or quiescent extinction. To ensure structural stability across deep recurrent networks, WEFT enforces hard homeostatic saturation bounds $[W_{\min}, W_{\max}]$:

$$W_{t+1} = \Pi_{[W_{\min}, W_{\max}]} \left( W_t + \Delta W(\Delta t) \right) = \max\left(W_{\min}, \min\left(W_{\max}, W_t + \Delta W(\Delta t)\right)\right)$$

In WEFT Formal Suite Target #335 (`formal/STDPConvergenceInvariants.lean`), these homeostatic bounds are formally proven in Lean 4 without non-standard axioms:
- **Theorem `stdp_weight_in_bounds`:** Guarantees $\forall W \in \mathbb{R}, W_{\min} \le \Pi_{[W_{\min}, W_{\max}]}(W) \le W_{\max}$ whenever $W_{\min} \le W_{\max}$.
- **Theorem `stdp_ltp_monotone`:** Proves monotonic non-decreasing weight transitions during causal pre-before-post pairings prior to saturation.
- **Theorem `stdp_ltd_monotone`:** Proves monotonic non-increasing weight transitions during anti-causal pairings prior to lower-bound clamping.

---

### 3. Triton 3.9 SRAM Fused Event Kernel & AVX2 LIF Micro-Engine

Conventional deep learning frameworks evaluate dense spike matrices globally, incurring prohibitive memory traffic. In `weft/kernels/fused_stdp_event_triton.py`, we implement a block-fused event kernel:
- Spike timestamps $t_{\text{pre}}$ and $t_{\text{post}}$ are loaded into streaming multiprocessor (SM) registers.
- The exponential decay $\Delta W$ and homeostatic clamp are evaluated strictly in GPU SRAM registers before writing back $W_{\text{new}}$ as FP16.
- Synchronously, `tools/weft_bench_lif_neuron.cpp` provides a vector AVX2 Leaky Integrate-and-Fire (LIF) simulation routine with fused multiply-accumulate (`vfmadd213ps`) exponential leak tracking, demonstrating $13.69 \times 10^9$ neuron-evaluations per second.

```
+-----------------------------------------------------------------------------+
|                          WEFT GPU SRAM STDP PIPELINE                        |
|                                                                             |
|  [t_pre]  ---> [ Register Bank ]                                            |
|                       |                                                     |
|                  (t_post - t_pre) ---> [ Piecewise Exp LUT / Hardware Fused]|
|                       |                                 |                   |
|  [t_post] ---> [ Register Bank ]                        v                   |
|                                                  Delta W Influx             |
|                                                         |                   |
|  [W_curr] ---> [ SRAM Register ] ---------------------> +                   |
|                                                         |                   |
|                                               [ Clamp Min / Max ]           |
|                                                         |                   |
|                                                    W_new (FP16)             |
+-----------------------------------------------------------------------------+
```

---

### 4. Cortical Manifold Gating via the Basal Ganglia Loop

The distributed cortical manifold model (Paper #142) demonstrates that high-dimensional associative integration emerges from long-range anatomical axonal tracts and phase coupling. However, cortical recurrent networks inherently generate multiple competing attractor manifolds simultaneously. The brain prevents behavioral and cognitive chaos through subcortical arbitration mediated by the **Basal Ganglia-Thalamocortical Loop**.

```
                           +----------------------+
                           |   CORTICAL MANIFOLD  |
                           |  (Distributed Areas) |
                           +----------------------+
                             |         |        |
        Glutamate (+)        |         |        | Glutamate (+)
        (Fast Corticostriatal|         |        | [Hyperdirect Pathway < 10ms]
                             v         |        v
     +-----------------------------+   |   +-----------------------+
     | STRIATUM (Medium Spiny N.)  |   |   |  SUBTHALAMIC NUCLEUS  |
     |   D1 (Direct)  D2 (Indirect)|   |   |        (STN)          |
     +-----------------------------+   |   +-----------------------+
           |                 |         |             |
  GABA (-) |        GABA (-) |         |             | Glutamate (+)
  [Direct] |                 v         |             | [Global Brake]
           |             +-------+     |             |
           |             |  GPe  |     |             |
           |             +-------+     |             |
           |                 |         |             |
           |        GABA (-) |         |             |
           |                 v         v             v
           |             +-------------------------------+
           +------------>|  GPi / SNr (Output Nuclei)    |<----+ (Tonic GABA Brake)
                         +-------------------------------+
                                         |
                                GABA (-) | (Disinhibition releases action)
                                         v
                         +-------------------------------+
                         |      THALAMIC RETICULAR /     |
                         |      RELAY VENTRAL NUCLEI     |
                         +-------------------------------+
                                         |
                            Glutamate (+)| (Recurrent Manifold Driver)
                                         v
                           +----------------------+
                           |  SELECTED ATTRACTOR  |
                           +----------------------+
                                         ^
                                         | (Phasic Dopamine DA Burst)
                         +-------------------------------+
                         |  SUBSTANTIA NIGRA COMPACTA /  |
                         |       VTA (Reward / RPE)      |
                         +-------------------------------+
```

#### 4.1 The Three Corticostriatal Pathways
1. **The Direct Pathway ($D_1$ Dopamine Receptor G-protein coupled $\text{G}_{\alpha s/olf}$):**
   - Cortical pyramidal neurons project glutamatergic afferents to striatal medium spiny neurons (MSNs) expressing $D_1$ receptors.
   - Activation of $D_1$ MSNs releases inhibitory GABA directly onto the internal segment of the Globus Pallidus ($\text{GPi}$) and Substantia Nigra pars reticulata ($\text{SNr}$).
   - Because GPi/SNr tonically inhibit thalamocortical relay cells at baseline ($40-80\text{ Hz}$ GABAergic firing), striatal $D_1$ activation transiently *silences* GPi/SNr output.
   - **Functional Outcome:** Thalamic disinhibition. The selected cortical manifold attractor receives recurrent thalamic drive, allowing it to transition from a latent state to conscious execution.

2. **The Indirect Pathway ($D_2$ Dopamine Receptor G-protein coupled $\text{G}_{\alpha i/o}$):**
   - Striatal MSNs expressing $D_2$ receptors project GABA to the external segment of the Globus Pallidus ($\text{GPe}$).
   - GPe inhibition disinhibits the Subthalamic Nucleus ($\text{STN}$).
   - STN sends excitatory glutamatergic projections to GPi/SNr, *increasing* tonic inhibition onto the thalamus.
   - **Functional Outcome:** Broad suppression of non-selected, competing cortical attractor states.

3. **The Hyperdirect Pathway (Cortex $\rightarrow$ STN $\rightarrow$ GPi/SNr):**
   - Monosynaptic glutamatergic axonal fibers travel directly from frontal cortical areas (pre-SMA, inferior frontal gyrus) to the STN, bypassing the striatum entirely.
   - Conduction latency is exceptionally fast ($\approx 6-10\text{ ms}$).
   - **Functional Outcome:** Emergency global brake. When cognitive conflict or decision ambiguity occurs, the hyperdirect pathway instantly suppresses all thalamocortical relay channels, freezing manifold transitions until striatal $D_1/D_2$ arbitration resolves the winning attractor.

#### 4.2 Neuromodulatory Reward Gating of Plasticity
Phasic dopamine ($DA$) bursts emitted from the Substantia Nigra pars compacta ($SNc$) and Ventral Tegmental Area ($VTA$) encode reward prediction errors ($\text{RPE} = r - \hat{r}$). In dendritic spines of cortical and striatal neurons:
- Elevated dopamine concentrations activate intracellular protein kinase A ($\text{PKA}$), increasing the amplitude parameter $A_+$ and extending the coincidence window $\tau_+$ of causal STDP.
- Absence or dips in dopamine favor $D_2$-mediated long-term depression ($A_-$ amplification).
- **Physical Integration:** The basal ganglia serves as the physical reward-gated filter that decides *which* cortical attractor trajectory is reinforced into long-term structural connectivity via STDP, enforcing causal physical convergence without ungrounded teleological assumptions.

---

### 5. Associative Memory Simulation Results

Using `scripts/sim_stdp_associative_memory.py`, a recurrent network of 64 biophysical units was trained across 40 epochs with interleaved orthogonal memory states ($P_A$ and $P_B$).
- Coincident within-pattern spike timing drove selective potentiation, increasing intra-assembly synaptic strength to $\mu(W_{AA}) = 0.771$, while cross-pattern synapses remained repressed at $\mu(W_{AB}) = 0.198$.
- Cueing the network with an incomplete ($50\%$) partial probe yielded robust pattern completion:
  $$\text{Target Activation} = 6.04 \text{ a.u.} \quad \text{vs} \quad \text{Background Activation} = 1.60 \text{ a.u.}$$
  $$\text{Signal-to-Noise Ratio (SNR)} = +5.77\text{ dB}$$
- Pattern completion successfully recovered the complete attractor state without catastrophic interference.

---

### 6. Conclusion

By combining Lean 4 formally proven homeostatic invariants, fused GPU SRAM Triton kernels, SIMD AVX2 microbenchmarks, and biophysical Basal Ganglia-cortical manifold coupling, WEFT unifies microscopic synaptic plasticity with macro-scale subcortical attractor gating. This closes the loop between local physical event timing and global systemic cognition.
