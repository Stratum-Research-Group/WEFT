#!/usr/bin/env python3
"""
Target #398: Striatal Cholinergic Interneuron (ChI) Hyper-Rebound & Dystonic Co-Contraction
Collaborative Research Target with Kevin W. McCairn, Ph.D.
Motor Systems Neurobiology & Theoretical Neuroscience Laboratory / Stratum Research Group

Simulates:
1. Tonic ChI pacemaking (4.5 Hz).
2. Cortical input burst triggering Dopamine D2 pause and unconstrained post-inhibitory hyper-rebound (Ih current).
3. Striatal Acetylcholine [ACh] elevation and reciprocal motor inhibition breakdown.
4. Co-contraction index (CCI) escalation in dystonic posturing.
5. Anticholinergic / M4 Muscarinic PAM therapeutic rescue.
"""

import numpy as np
import sys


def run_cholinergic_dystonia_simulation():
    dt = 0.001  # 1 ms resolution
    t_total = 3.0  # 3.0 seconds
    time = np.arange(0, t_total, dt)
    n_steps = len(time)

    # State traces
    d2_tone = np.ones(n_steps)
    m4_pam = np.zeros(n_steps)
    chi_rate = np.zeros(n_steps)
    ach_conc = np.zeros(n_steps)
    agonist_emg = np.zeros(n_steps)
    antagonist_emg = np.zeros(n_steps)
    cci_trace = np.zeros(n_steps)

    # Experimental Protocol:
    # 0.0s - 0.8s: Baseline physiological state (D2 = 1.0)
    # 0.8s - 2.0s: Striatal dopamine depletion (D2 drops to 0.05, triggering hyper-rebound)
    # 2.0s - 3.0s: M4 PAM / Anticholinergic infusion (M4 increases to 0.85)
    for i, t in enumerate(time):
        if t < 0.8:
            d2_tone[i] = 1.0
            m4_pam[i] = 0.0
        elif t < 2.0:
            d2_tone[i] = 0.05
            m4_pam[i] = 0.0
        else:
            d2_tone[i] = 0.05
            m4_pam[i] = 0.85

        # Cortical burst-pause cycle (1.5 Hz slow wave)
        cycle_phase = (t * 1.5) % 1.0
        d2 = d2_tone[i]
        m4 = m4_pam[i]

        if cycle_phase < 0.15:
            # Cortical/Thalamostriatal burst
            burst_rate = 15.0
            chi_rate[i] = burst_rate
        elif cycle_phase < 0.45:
            # Pause phase
            pause_rate = 0.8
            chi_rate[i] = pause_rate
        elif cycle_phase < 0.75:
            # Hyper-rebound phase (suppressed by D2, amplified by dopamine loss)
            rebound_amp = 4.5 + 24.0 * (1.0 - d2) * (1.0 - 0.70 * m4)
            chi_rate[i] = rebound_amp
        else:
            # Tonic recovery
            chi_rate[i] = 4.5

        # Striatal ACh concentration (nM)
        ach_conc[i] = chi_rate[i] * 4.2

        # Agonist voluntary drive
        agonist_emg[i] = 30.0 + 0.25 * chi_rate[i]

        # Antagonist reciprocal inhibition leakage:
        # Under normal conditions, antagonist remains quiet (~8%).
        # Under ChI hyper-rebound without D2, excessive ACh disinhibits antagonist striatal pathways.
        if d2 < 0.40:
            leakage = chi_rate[i] * 0.20 * (1.0 - 0.85 * m4)
            antagonist_emg[i] = 8.0 + leakage
        else:
            antagonist_emg[i] = 8.0

        # Co-contraction index (CCI) = 2 * min(Ag, Ant) / (Ag + Ant)
        ag = agonist_emg[i]
        ant = antagonist_emg[i]
        cci_trace[i] = (2.0 * min(ag, ant)) / (ag + ant)

    print("=========================================================================")
    print("PHYSIOLOGICAL SIMULATION: STRIATAL CHOLINERGIC INTERNEURON DYSTONIA")
    print("=========================================================================")
    
    # Milestone verification: compare rebound phases across regimes (cycle_phase ~ 0.60)
    idx_healthy = int(0.40 / dt)    # t = 0.40s, phase = 0.60
    idx_dystonic = int(1.70 / dt)   # t = 1.70s, phase = 0.55-0.60
    idx_rescued = int(2.40 / dt)    # t = 2.40s, phase = 0.60

    print(f"1. Baseline Physiological Rebound Phase (t = 0.40s):")
    print(f"   D2 Tone:               {d2_tone[idx_healthy]:.2f}")
    print(f"   ChI Rebound Peak Rate: {chi_rate[idx_healthy]:.1f} Hz (Physiological Baseline)")
    print(f"   ACh Concentration:     {ach_conc[idx_healthy]:.1f} nM")
    print(f"   Agonist EMG:           {agonist_emg[idx_healthy]:.1f} %")
    print(f"   Antagonist EMG:        {antagonist_emg[idx_healthy]:.1f} % (Reciprocal Inhibition Preserved)")
    print(f"   Co-Contraction Index:  {cci_trace[idx_healthy]*100.0:.1f} % (Safe: < 45%)\n")

    print(f"2. Dopamine-Depleted Dystonic Rebound Phase (t = 1.70s):")
    print(f"   D2 Tone:               {d2_tone[idx_dystonic]:.2f}")
    print(f"   ChI Hyper-Rebound Peak:{chi_rate[idx_dystonic]:.1f} Hz (Severe Pathological Rebound)")
    print(f"   ACh Overflow:          {ach_conc[idx_dystonic]:.1f} nM")
    print(f"   Agonist EMG:           {agonist_emg[idx_dystonic]:.1f} %")
    print(f"   Antagonist Leakage:    {antagonist_emg[idx_dystonic]:.1f} % (Inhibition Breakdown)")
    print(f"   Co-Contraction Index:  {cci_trace[idx_dystonic]*100.0:.1f} % (PATHOLOGICAL DYSTONIA: > 50%)\n")

    print(f"3. M4 PAM Anticholinergic Rescue Phase (t = 2.40s):")
    print(f"   M4 Infusion:           {m4_pam[idx_rescued]:.2f}")
    print(f"   ChI Rebound Rate:      {chi_rate[idx_rescued]:.1f} Hz (Attenuated)")
    print(f"   ACh Concentration:     {ach_conc[idx_rescued]:.1f} nM")
    print(f"   Antagonist EMG:        {antagonist_emg[idx_rescued]:.1f} % (Restored Restraint)")
    print(f"   Co-Contraction Index:  {cci_trace[idx_rescued]*100.0:.1f} % (Rescued: < 45%)\n")

    # Assertions
    assert cci_trace[idx_healthy] < 0.45, "Healthy state co-contraction must remain low"
    assert cci_trace[idx_dystonic] > 0.50, "Dopamine-depleted hyper-rebound must trigger dystonic co-contraction"
    assert cci_trace[idx_rescued] < 0.45, "M4 PAM therapy must rescue co-contraction below dystonia threshold"
    print("[SUCCESS] All biological invariant milestones verified cleanly.")


if __name__ == "__main__":
    run_cholinergic_dystonia_simulation()
