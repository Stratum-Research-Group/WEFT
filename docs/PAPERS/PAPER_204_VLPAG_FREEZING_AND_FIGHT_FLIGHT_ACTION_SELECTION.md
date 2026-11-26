# Paper #204: Ventrolateral Periaqueductal Gray (vlPAG) Freezing vs Fight/Flight Action Selection Invariants

**Authors:** Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.  
**Series:** Synaptic & Neurobiological Circuit Verification Series  
**Target:** Lean 4 Formal Verification Target #395 (`FormalSuite.VLPAGSurvivalInvariants`)  
**Repository:** WEFT Engine Monorepo  
**Date:** November 2026  

---

## Abstract

The periaqueductal gray (PAG) of the midbrain serves as an evolutionarily conserved coordinator of innate defensive behaviors. Anatomically and functionally partitioned into longitudinal columns, the ventrolateral PAG (vlPAG) mediates passive defensive coping characterized by behavioral immobility (freezing) and parasympathetic/bradycardic responses, whereas the lateral and dorsolateral PAG (lPAG/dlPAG) orchestrate active fight-or-flight escape responses accompanied by vigorous motor output and sympathetic tachycardia. In this work, we formalize the synaptic logic of threat assessment, looming approach kinetics, and mutually inhibitory collateral interactions that govern the state transition between immobility and explosive escape. We specify the system within Lean 4, proving twelve machine-checked theorems establishing non-negative firing rates, strict bounded maximums, threat-distance bifurcation, and mutual exclusivity between freezing and escape attractors with zero axioms and zero sorries. Furthermore, we develop an AVX2/FMA SIMD vectorization implementation operating at 1.77 Gelem/s and author a fused GPU tensor kernel in Triton 3.9.

---

## 1. Neuroanatomical Organization & Columnar Defensive Control

### 1.1 Columnar Partitioning of the Midbrain PAG
The midbrain periaqueductal gray surrounds the cerebral aqueduct and is organized into functionally segregated longitudinal columns (Bandler & Shipley 1994; Carrive 1993):
1. **Ventrolateral Column (vlPAG):** Receives afferents from the central amygdala (CeA) and prefrontal infralimbic cortex. Activation of excitatory projection neurons in the vlPAG promotes tonic immobility (conditioned freezing), antinociception, and bradycardia.
2. **Lateral and Dorsolateral Columns (lPAG / dlPAG):** Receive direct monosynaptic sensory afferents from the deep layers of the superior colliculus signaling rapidly expanding visual looming stimuli. Activation of dlPAG neurons triggers explosive running, jumping, and defensive escalation.

### 1.2 Looming Threat Kinetics and Sensory Triggers
Visual looming stimuli—characterized by non-linear optical expansion simulating an approaching predator—are transduced by retinal ganglion cells projecting to the superior colliculus (Yilmaz & Meister 2013). When threat proximity is distant or looming velocity is negligible, sustained threat presence activates CeA-to-vlPAG circuits, locking the animal into postural immobility. However, when looming velocity exceeds an acute behavioral threshold, collicular efferents drive dlPAG excitation, overwhelming vlPAG activity.

### 1.3 Collateral Cross-Inhibition & Attractor Switching
Intracolumnar and intercolumnar GABAergic interneurons mediate collateral cross-inhibition between the vlPAG and dlPAG (Tovote et al. 2016; Lefler et al. 2020). This reciprocal inhibition creates a bistable dynamical switch. Under intermediate threat distances, both attractors compete, but once looming reaches critical proximity, the dlPAG strongly suppresses vlPAG firing, ensuring unambiguous motor output and preventing behavioral paralysis during imminent danger.

---

## 2. Mathematical Formalization & Invariants

Let $d \in [0, 1]$ denote normalized threat distance ($0 = \text{contact}$, $1 = \text{distant}$), $l \in [0, 1]$ represent looming approach speed, and $c \in [0, 1]$ represent baseline fear/corticosterone tone. In fixed-point integer scaling ($\times 100$):

### 2.1 Drive Formulations
The raw drive to the ventrolateral (freezing) and dorsolateral (escape) columns is defined as:
$$\text{Drive}_{\text{vl}}(d, l, c) = 300 + \frac{35 \times \text{clamp}(d, 0, 100) \times (100 - \text{clamp}(l, 0, 100))}{100} + 15 \times c$$
$$\text{Drive}_{\text{dl}}(d, l) = 200 + 60 \times (100 - \text{clamp}(d, 0, 100)) + 40 \times \text{clamp}(l, 0, 100)$$

### 2.2 Collateral Cross-Inhibition
$$\text{Rate}_{\text{vl}} = \text{clamp}\left(\text{Drive}_{\text{vl}} - \frac{45 \times \text{Drive}_{\text{dl}}}{100}, 0, 5000\right)$$
$$\text{Rate}_{\text{dl}} = \text{clamp}\left(\text{Drive}_{\text{dl}} - \frac{20 \times \text{Drive}_{\text{vl}}}{100}, 0, 8000\right)$$

### 2.3 Lean 4 Formally Verified Theorems
Our Lean 4 verification (`formal/VLPAGSurvivalInvariants.lean`) establishes twelve machine-checked proofs:
1. `raw_vlpag_non_negative`: Raw vlPAG drive is strictly non-negative ($\ge 0$).
2. `raw_dlpag_non_negative`: Raw dlPAG drive is strictly non-negative ($\ge 0$).
3. `effective_vlpag_bounded`: Effective vlPAG rate is bounded within $[0, 5000]$ ($50.0\text{ Hz}$).
4. `effective_dlpag_bounded`: Effective dlPAG rate is bounded within $[0, 8000]$ ($80.0\text{ Hz}$).
5. `sympathetic_tone_bounded`: Sympathetic tone is bounded within $[0, 1000]$.
6. `distant_threat_triggers_freezing`: Distant threat with elevated fear tone guarantees freezing activation ($\ge 20.0\text{ Hz}$).
7. `imminent_looming_triggers_escape`: Immediate looming proximity guarantees explosive flight escape ($\ge 35.0\text{ Hz}$).
8. `imminent_escape_cross_inhibition_strength`: dlPAG firing exerts strong inhibitory clamping ($\ge 40.0\text{ Hz}$).
9. `imminent_threat_suppresses_freezing`: Looming strike completely suppresses freezing mode to false.
10. `raw_dlpag_monotonic_looming`: dlPAG drive scales monotonically with looming speed.
11. `high_flight_elevates_sympathetic_tone`: Explosive flight drives sympathetic tone past the tachycardia threshold ($\ge 480$).
12. `quiescent_sympathetic_tone_calm`: Quiescent baseline maintains low autonomic tone ($\le 40$).

---

## 3. High-Performance Hardware Realization

### 3.1 AVX2/FMA Native Vectorized SIMD Microbenchmark
We implemented an 8-lane AVX2/FMA SIMD benchmark in C++ (`tools/weft_bench_vlpag_survival.cpp`):
- Fused multiply-add registers process raw drive computations and reciprocal collateral inhibition simultaneously.
- Single-cycle `_mm256_min_ps` and `_mm256_max_ps` guarantee branchless clamping.
- **Microbenchmark Performance:** Achieved **1,771.7 Melem/s** (1.772 Gelem/s) at **0.564 ms** per $10^6$ operations.

### 3.2 Fused Triton 3.9 GPU Kernel
We authored `weft/kernels/fused_vlpag_survival_triton.py` to enable batched survival action selection over large agent populations on GPU tensors with coalesced 1024-thread blocks.

---

## 4. Empirical Simulation Results

The reference physiological simulation (`scripts/sim_vlpag_survival.py`) demonstrated:
- **Baseline Safe State ($t = 0.4\text{ s}$):** Threat distance $= 0.95$, Looming $= 0.05$, vlPAG $= 34.44\text{ Hz}$, dlPAG $= 0.00\text{ Hz}$, Sympathetic $= 6.9\%$.
- **Distant Threat Immobility ($t = 0.9\text{ s}$):** Threat distance $= 0.75$, Looming $= 0.10$, vlPAG $= 26.93\text{ Hz}$ ($> 20\text{ Hz}$ threshold), dlPAG $= 13.72\text{ Hz}$, Sympathetic $= 21.9\%$, Freezing Probability $= 100.0\%$.
- **Looming Strike Attack ($t = 1.6\text{ s}$):** Threat distance $= 0.15$, Looming $= 0.90$, vlPAG silenced to $0.00\text{ Hz}$, dlPAG exploded to $80.00\text{ Hz}$ ($> 35\text{ Hz}$ threshold), Sympathetic Tone $= 96.0\%$, Escape Probability $= 100.0\%$.

---

## 5. References

1. Bandler, R., & Shipley, M. T. (1994). Columnar organization in the midbrain periaqueductal gray: modules for emotional expression? *Trends in Neurosciences*, 17(9), 379-389.
2. Carrive, P. (1993). The periaqueductal gray and defensive behavior: functional representation and neuronal organization. *Behavioural Brain Research*, 58(1-2), 27-47.
3. Tovote, P., Esposito, M. S., Botta, P., Chaudun, F., Fadok, J. P., Markovic, M., ... & Luthi, A. (2016). Midbrain circuits for conditioned inhibition of movement. *Nature*, 534(7606), 206-212.
4. Lefler, Y., Campagner, D., & Branco, T. (2020). The role of the periaqueductal gray in visually guided escape behavior. *Current Opinion in Neurobiology*, 60, 115-121.
5. Yilmaz, M., & Meister, M. (2013). Rapid innate defensive responses of mice to looming visual stimuli. *Current Biology*, 23(20), 2011-2015.
6. Evans, D. A., Stempel, A. V., Vale, R., Ruehle, S., Lefler, Y., & Branco, T. (2018). A synaptic threshold mechanism for computing escape decisions. *Nature*, 558(7711), 590-594.
7. De Oca, B. M., DeCola, J. P., Maren, S., & Fanselow, M. S. (1998). Distinct neurotoxic lesions of the periaqueductal gray selectively attenuate conditioned freezing and unconditioned panic-like responses. *Journal of Neuroscience*, 18(9), 3426-3432.
8. McCairn, K. W., & Turner, R. S. (2009). Pallidal and cortical synchronization during motor disruption and behavioral arrest. *Journal of Neurophysiology*, 101(4), 1961-1975.
9. McCairn, K. W., Iriki, A., & Isoda, M. (2013). Global basal ganglia and cortical synchrony during hyperkinetic motor disorders. *The Journal of Neuroscience*, 33(31), 12525-12536.
10. Mitrofanis, J. (2005). Some certainty about the mysterious zona incerta? *Progress in Neurobiology*, 75(2), 163-183.
11. Zhang, X., & van den Pol, A. N. (2017). Rapid binge-like eating and body weight gain driven by zona incerta GABA neuron stimulation. *Science*, 356(6340), 853-859.
12. Zhao, Z. D., Chen, Z., Xiang, X., Hu, M. N., Xie, H., Jia, X., ... & Shen, W. L. (2019). Zona incerta GABAergic neurons integrate prey-related sensory signals and induce predatory hunting to optimize survival. *Neuron*, 103(4), 717-730.
13. Crick, F., & Koch, C. (2005). What is the function of the claustrum? *Philosophical Transactions of the Royal Society B: Biological Sciences*, 360(1458), 1271-1279.
14. Atlan, G., Terem, A., Peretz-Rivlin, N., Sehrawat, K., Gonzales, B. J., Pozner, G., ... & Citri, A. (2018). The claustrum supports resilience to distraction. *Current Biology*, 28(17), 2752-2762.
15. Narikiyo, K., Mizuguchi, R., Ajima, A., Shiozaki, M., Hamanaka, H., Rouiller, E. M., & Rockland, K. S. (2020). The claustrum coordinates cortical slow-wave activity. *Nature Neuroscience*, 23(6), 741-753.
16. Buzsaki, G., & Draguhn, A. (2004). Neuronal oscillations in cortical networks. *Science*, 304(5679), 1926-1929.
17. Fries, P. (2005). A mechanism for cognitive dynamics: neuronal communication through coherence. *Trends in Cognitive Sciences*, 9(10), 474-480.
18. Brown, P. (2003). Oscillatory nature of human basal ganglia activity: relationship to normal and abnormal movement. *Movement Disorders*, 18(4), 357-363.
19. Halassa, M. M., & Acsady, L. (2016). Thalamic inhibition: architecture and function. *Neuron*, 91(6), 1187-1202.
20. Pinault, D. (2004). The thalamic reticular nucleus: structure, function and concept. *Brain Research Reviews*, 46(1), 1-31.
21. Salinas, E., & Sejnowski, T. J. (2001). Correlated neuronal activity and the flow of neural information. *Nature Reviews Neuroscience*, 2(8), 539-550.
22. Wang, X. J. (2010). Neurophysiological and computational principles of cortical slow and gamma rhythms. *Physiological Reviews*, 90(3), 1195-1268.
