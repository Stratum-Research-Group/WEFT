# WEFT Architecture Suite Execution Summary

- **Run ID:** `20261125_143250_535fbe8`
- **Timestamp (UTC):** `2026-11-25T14:32:50.763775+00:00`
- **Git Commit:** `535fbe83a31870bc84d23c4dbdc2b019e5f1defe` (dirty=False)
- **Tier:** `standard` | **Track:** `all` | **Device:** `cpu`
- **Verdict:** **STANDARD TIER PASSED (Clean 535fbe8, cpu)**

| Test ID | Track | Gate Type | Status | Duration (s) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `B1_Causality` | Track A | HARD | **PASS** | 0.09s | Causality strictly preserved across all positions and architecture variants. |
| `B2_Prefill_Decode_Parity` | Track A | HARD | **PASS** | 0.01s | Prefill vs Decode parity certified (max diff = 0.000000) |
| `B3_Cache_Formats` | Track A | HARD | **PASS** | 0.01s | Cache formats and state retention agree with full sequence recompute. |
| `B5_Exact_Recurrence` | Track A | HARD | **PASS** | 0.03s | B5 certified: WeftAttention layer forward and single-token decode strictly match fp64 recurrence across lengths [1, 7, 64, 513] and 8 decode steps (diff < 1e-4). |
| `B8_Padding_Batching` | Track A | HARD | **PASS** | 0.01s | Padding and batching invariant certified (diff = 0.000000) |
| `C1_Format_Roundtrip` | Track B | HARD | **PASS** | 0.01s | Container format round-trip bit-exact preservation certified. |
| `C5_Arch_Labeling` | Track B | HARD | **PASS** | 0.00s | Track B results verified with 'transformer_converted' architecture label. |
| `RG1_No_Synthetic_Claims` | Track B | HARD | **PASS** | 0.00s | No synthetic 'zero-degradation' claims detected in active docs. |
| `RG3_Gate_Active_Checks` | System | HARD | **PASS** | 7.03s | Fast gatekeeper executes active test scripts and passes --ci. |
| `RG4_Hardware_Accuracy` | System | HARD | **PASS** | 0.00s | Hardware profile correctly documents Blackwell SM 12.0 GDDR7 across papers and README. |
| `RG7_Repo_Hygiene` | System | HARD | **PASS** | 0.00s | Repository root is clean of release kit archives. |
