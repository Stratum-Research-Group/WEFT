#!/usr/bin/env python3
"""
Target #401: Amyotrophic Lateral Sclerosis (ALS) Corticomotoneuronal Excitotoxicity & Renshaw Recurrent Disinhibition
Collaborative Research Target with Kevin W. McCairn, Ph.D.
Motor Systems Neurobiology & Theoretical Neuroscience Laboratory / Stratum Research Group

Simulates:
1. Healthy motoneuron homeostasis (EAAT2 100%, Renshaw 100%, Glutamate 1.2 uM, Rate 8 Hz, Fasciculations 0.1 Hz).
2. Astrocytic EAAT2 loss & spinal Renshaw interneuron demise (Glutamate surges to 16.5 uM, Motoneurons fire at 65.9 Hz, Fasciculations at 4.7 Hz).
3. Spasticity and excitotoxic motoneuronal death.
4. Dual neuroprotective rescue with Riluzole (glutamate release blocker) and glycinergic agonist (restoring recurrent inhibition).
"""

import numpy as np
import sys


def run_als_excitotoxicity_simulation():
    dt = 0.001
    t_total = 3.0
    time = np.arange(0, t_total, dt)
    n_steps = len(time)

    eaat2 = np.ones(n_steps)
    renshaw = np.ones(n_steps)
    riluzole = np.zeros(n_steps)
    glycine = np.zeros(n_steps)

    glu_conc = np.zeros(n_steps)
    moto_rate = np.zeros(n_steps)
    ren_gain = np.zeros(n_steps)
    fasc_freq = np.zeros(n_steps)

    for i, t in enumerate(time):
        if t < 0.8:
            # Healthy
            e = 1.0
            r = 1.0
            ril = 0.0
            gly = 0.0
        elif t < 2.0:
            # ALS neurodegeneration
            e = 0.15
            r = 0.20
            ril = 0.0
            gly = 0.0
        else:
            # Dual therapy rescue
            e = 0.15
            r = 0.20
            ril = 0.85
            gly = 0.80

        eaat2[i] = e
        renshaw[i] = r
        riluzole[i] = ril
        glycine[i] = gly

        # Synaptic glutamate
        glu = 1.2 + 18.0 * (1.0 - e) * (1.0 - 0.70 * ril)
        glu_conc[i] = glu

        # Renshaw gain
        gain = min(1.0, 0.80 * r + 0.50 * gly)
        ren_gain[i] = gain

        # Motoneuron rate
        raw_rate = 12.0 + 3.5 * glu
        rate = max(8.0, raw_rate - 24.0 * gain)
        moto_rate[i] = rate

        # Fasciculations
        if glu > 4.0 and gain < 0.50:
            fasc = min(8.0, 0.45 * (glu - 4.0) * (1.0 - gain))
        else:
            fasc = 0.1
        fasc_freq[i] = fasc

    idx_healthy = int(0.4 / dt)
    idx_als = int(1.4 / dt)
    idx_rescued = int(2.6 / dt)

    print("=========================================================================")
    print("PHYSIOLOGICAL SIMULATION: ALS EXCITOTOXICITY & RENSHAW DISINHIBITION")
    print("=========================================================================")
    print(f"1. Healthy Baseline Phase (t = 0.4s):")
    print(f"   EAAT2 Expression:       {eaat2[idx_healthy]*100.0:.1f} %")
    print(f"   Synaptic [Glutamate]:   {glu_conc[idx_healthy]:.1f} uM (Clamped Clearance)")
    print(f"   Renshaw Feedback Gain:  {ren_gain[idx_healthy]*100.0:.1f} % (Intact Glycinergic Brake)")
    print(f"   Motoneuron Firing Rate: {moto_rate[idx_healthy]:.1f} Hz")
    print(f"   Fasciculation Rate:     {fasc_freq[idx_healthy]:.2f} Hz (Quiescent)\n")

    print(f"2. ALS Degenerative Phase (t = 1.4s):")
    print(f"   EAAT2 Expression:       {eaat2[idx_als]*100.0:.1f} % (Astrocytic Loss)")
    print(f"   Synaptic [Glutamate]:   {glu_conc[idx_als]:.1f} uM (Severe Toxic Accumulation)")
    print(f"   Renshaw Feedback Gain:  {ren_gain[idx_als]*100.0:.1f} % (Disinhibited Runaway)")
    print(f"   Motoneuron Firing Rate: {moto_rate[idx_als]:.1f} Hz (RUNAWAY EXCITOTOXICITY > 45 Hz)")
    print(f"   Fasciculation Rate:     {fasc_freq[idx_als]:.2f} Hz (PATHOLOGICAL DENERVATION BURSTS)\n")

    print(f"3. Dual Therapeutic Rescue Phase (t = 2.6s):")
    print(f"   Riluzole Glu Blockade:  {riluzole[idx_rescued]*100.0:.1f} %")
    print(f"   Glycinergic Agonism:    {glycine[idx_rescued]*100.0:.1f} %")
    print(f"   Synaptic [Glutamate]:   {glu_conc[idx_rescued]:.1f} uM (Reduced)")
    print(f"   Renshaw Gain Restored:  {ren_gain[idx_rescued]*100.0:.1f} %")
    print(f"   Motoneuron Firing Rate: {moto_rate[idx_rescued]:.1f} Hz (Stabilized < 25 Hz)")
    print(f"   Fasciculation Rate:     {fasc_freq[idx_rescued]:.2f} Hz (Suppressed)\n")

    # Assertions
    assert glu_conc[idx_healthy] < 2.0, "Healthy synaptic glutamate must remain < 2.0 uM"
    assert glu_conc[idx_als] > 10.0, "Astrocytic EAAT2 loss must drive glutamate > 10.0 uM"
    assert moto_rate[idx_als] > 45.0, "Excitotoxicity and Renshaw loss must drive motoneuron rate > 45.0 Hz"
    assert fasc_freq[idx_als] > 3.0, "ALS state must produce fasciculation frequencies > 3.0 Hz"
    assert moto_rate[idx_rescued] <= 25.0, "Dual rescue must restore motoneuron rate <= 25.0 Hz"
    print("[SUCCESS] All ALS excitotoxicity and Renshaw disinhibition invariants verified.")


if __name__ == "__main__":
    run_als_excitotoxicity_simulation()
