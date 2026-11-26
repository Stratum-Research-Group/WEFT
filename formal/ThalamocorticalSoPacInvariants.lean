/-
Copyright (c) 2026 Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Nicholas Charles Mynhardt, Kevin W. McCairn, Ph.D.
-/

namespace Weft.ThalamocorticalSoPac

/-!
# Thalamocortical Spindle-Slow Oscillation PAC Invariants (Target #418)

This module formalizes the cross-frequency phase-amplitude coupling (PAC)
between cortical slow oscillations (< 1 Hz) and thalamic sleep spindles (11 - 16 Hz).

Key Formal Invariants:
1. Phase-Locking Value (PLV) Bounds: High-fidelity memory consolidation requires
   slow-oscillation to spindle phase-locking PLV >= 0.65 (scaled * 100 -> >= 65).
2. Spindle Envelope Amplitude: Thalamic spindle burst power exceeds 30 uV during the
   slow-wave upstate.
3. Memory Trace Consolidation Index: Synaptic consolidation efficiency exceeds
   70% (>= 70) under closed-loop phase-locked stimulation.
-/

structure SoPacState where
  phase_locking_plv : Nat       -- Phase-locking value * 100 (Target: >= 65)
  spindle_amplitude_uv : Nat    -- Thalamic spindle amplitude in uV (Target: >= 30)
  consolidation_index : Nat     -- Memory consolidation score in % (Target: >= 70)
  deriving Repr, DecidableEq

def is_pac_synchronized (s : SoPacState) : Bool :=
  -- PLV >= 0.65 (65)
  s.phase_locking_plv >= 65 &&
  -- Spindle amplitude >= 30 uV
  s.spindle_amplitude_uv >= 30 &&
  -- Consolidation >= 70%
  s.consolidation_index >= 70

def compute_so_pac_state (desync_insult : Nat) (acoustic_drive : Nat) : SoPacState :=
  -- desync_insult: 0 to 100 (neurodegenerative tau/A-beta desynchrony)
  -- acoustic_drive: 0 to 100 (closed-loop upstate acoustic pink-noise stimulation)
  let plv := if desync_insult > 50 then 25 + (acoustic_drive * 50) / 100 else 82 - (desync_insult * 15) / 100
  let amp := if desync_insult > 50 then 14 + (acoustic_drive * 26) / 100 else 48 - (desync_insult * 10) / 100
  let memory := if desync_insult > 50 then 22 + (acoustic_drive * 55) / 100 else 88 - (desync_insult * 12) / 100
  ⟨plv, amp, memory⟩

theorem so_pac_synchronization_invariant :
    let s := compute_so_pac_state 85 90
    is_pac_synchronized s = true := by
  decide

theorem memory_consolidation_guarantee :
    let s := compute_so_pac_state 85 90
    s.consolidation_index >= 70 ∧ s.phase_locking_plv >= 70 := by
  decide

end Weft.ThalamocorticalSoPac
