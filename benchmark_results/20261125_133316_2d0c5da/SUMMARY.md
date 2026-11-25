# WEFT Architecture Suite Execution Summary

- **Run ID:** `20261125_133316_2d0c5da`
- **Timestamp (UTC):** `2026-11-25T13:33:16.420271+00:00`
- **Git Commit:** `2d0c5da314ed75f8c24322487a86d82a7db9aae7` (dirty=True)
- **Tier:** `smoke` | **Track:** `all` | **Device:** `cpu`
- **Verdict:** **ALL GATES PASSED**

| Test ID | Track | Gate Type | Status | Duration (s) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `B1_Causality` | Track A | HARD | **PASS** | 0.08s | Causality strictly preserved across all positions and architecture variants. |
| `B2_Prefill_Decode_Parity` | Track A | HARD | **PASS** | 0.01s | Prefill vs Decode parity certified (max diff = 0.000000) |
| `B3_Cache_Formats` | Track A | HARD | **PASS** | 0.01s | Cache formats and state retention agree with full sequence recompute. |
| `B5_Exact_Recurrence` | Track A | HARD | **PASS** | 0.00s | Global path recurrence formulation matches mathematical definition. |
| `B8_Padding_Batching` | Track A | HARD | **PASS** | 0.01s | Padding and batching invariant certified (diff = 0.000000) |
| `RG1_No_Synthetic_Claims` | Track B | HARD | **PASS** | 0.00s | No synthetic 'zero-degradation' claims detected. |
| `RG3_Gate_Active_Checks` | System | HARD | **PASS** | 0.00s | Fast gatekeeper executes active test scripts. |
| `RG4_Hardware_Accuracy` | System | HARD | **PASS** | 0.00s | Hardware profile correctly documents Blackwell SM 12.0 GDDR7. |
| `RG7_Repo_Hygiene` | System | HARD | **PASS** | 0.00s | Repository root is clean of release kit archives. |
