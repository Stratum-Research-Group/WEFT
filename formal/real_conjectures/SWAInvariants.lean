/-
Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group. All rights reserved.
Author: Nicholas Charles Mynhardt <research@stratum.org>

Formalization of Invariants for AVX-512 Sliding-Window Flash-Attention (SWA).
Certifies exact window interval bounds [k_start, k_end), causal containment,
and strictly bounded window length <= W without dynamic mask materialization.
-/

import Lean

namespace WEFT.Formal.SWAInvariants

/-- Sliding-window start index: max(0, qi - W + 1) -/
def k_start (qi : Int) (W : Int) : Int :=
  if qi - W + 1 > 0 then qi - W + 1 else 0

/-- Sliding-window end index (causal upper bound): min(qi + 1, kv_len) -/
def k_end (qi : Int) (kv_len : Int) : Int :=
  if qi + 1 < kv_len then qi + 1 else kv_len

/--
Theorem 1: Non-negativity of Window Start Index.
For any query index qi and window size W, k_start(qi, W) >= 0.
-/
theorem k_start_nonneg (qi : Int) (W : Int) :
  k_start qi W >= 0 := by
  unfold k_start
  split
  · rename_i h
    omega
  · omega

/--
Theorem 2: Causal Upper Bound.
The key index end boundary never exceeds causal visibility qi + 1.
-/
theorem k_end_le_causal (qi : Int) (kv_len : Int) :
  k_end qi kv_len <= qi + 1 := by
  unfold k_end
  split
  · omega
  · omega

/--
Theorem 3: Cache Buffer Upper Bound.
The key index end boundary never exceeds the physical KV cache length kv_len.
-/
theorem k_end_le_kv_len (qi : Int) (kv_len : Int) :
  k_end qi kv_len <= kv_len := by
  unfold k_end
  split
  · omega
  · omega

/--
Theorem 4: Sliding Window Span Invariant.
For any valid sequence where W > 0, the total active tokens evaluated in the inner
attention reduction loop (k_end - k_start) is strictly bounded by W.
-/
theorem window_span_bounded (qi : Int) (W : Int) (kv_len : Int) (_hW : W > 0) (_hqi : qi >= 0) :
  k_end qi kv_len - k_start qi W <= W := by
  unfold k_end k_start
  split <;> split
  · omega
  · omega
  · omega
  · omega

/--
Theorem 5: Causal Window Well-Foundedness.
If kv_len > 0 and qi >= 0, then if tokens are attended (k_start < k_end),
every attended key index ki satisfies 0 <= ki <= qi (strict causal preservation).
-/
theorem causal_window_indices (qi : Int) (W : Int) (kv_len : Int) (ki : Int)
  (h_start : k_start qi W <= ki) (h_end : ki < k_end qi kv_len) :
  ki >= 0 ∧ ki <= qi := by
  have h0 : k_start qi W >= 0 := k_start_nonneg qi W
  have h1 : ki >= 0 := by omega
  have h2 : k_end qi kv_len <= qi + 1 := k_end_le_causal qi kv_len
  have h3 : ki <= qi := by omega
  exact ⟨h1, h3⟩

end WEFT.Formal.SWAInvariants
