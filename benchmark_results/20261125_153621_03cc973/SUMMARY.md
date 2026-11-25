# WEFT Architecture Suite Execution Summary

- **Run ID:** `20261125_153621_03cc973`
- **Timestamp (UTC):** `2026-11-25T15:36:21.799588+00:00`
- **Git Commit:** `03cc973c5f967c974b301816d9e3cb03b46721a4` (dirty=True)
- **Tier:** `standard` | **Track:** `all` | **Device:** `cuda`
- **Verdict:** **STANDARD PASS (DIRTY TREE - cuda)**

| Test ID | Track | Gate Type | Status | Duration (s) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `B1_Causality` | Track A | HARD | **PASS** | 0.87s | Causality strictly preserved across all positions and architecture variants. |
| `B2_Prefill_Decode_Parity` | Track A | HARD | **PASS** | 0.14s | Prefill vs Decode parity certified (max diff = 0.000000) |
| `B3_Cache_Formats` | Track A | HARD | **PASS** | 0.03s | Cache formats and state retention agree with full sequence recompute. |
| `B5_Exact_Recurrence` | Track A | HARD | **PASS** | 0.23s | B5 certified: WeftAttention layer forward and single-token decode strictly match fp64 recurrence across lengths [1, 7, 64, 513] and 8 decode steps (diff < 1e-4). |
| `B8_Padding_Batching` | Track A | HARD | **PASS** | 0.01s | Padding and batching invariant certified (diff = 0.000001) |
| `B13_Triton_Parity` | Track A | HARD | **PASS** | 0.00s | B13 certified: Triton recurrent kernels match PyTorch sequential scan within 1e-4 tolerance on CUDA. |
| `B14_State_Growth_Bounds` | Track A | HARD | **PASS** | 0.29s | B14 certified: Recurrent state strictly bounded and free of NaN/Inf across 4096 steps. |
| `C1_Format_Roundtrip` | Track B | HARD | **PASS** | 0.01s | Container format round-trip bit-exact preservation certified. |
| `C5_Arch_Labeling` | Track B | HARD | **PASS** | 0.00s | Track B results verified with 'transformer_converted' architecture label. |
| `RG1_No_Synthetic_Claims` | Track B | HARD | **PASS** | 0.00s | No synthetic 'zero-degradation' claims detected in active docs. |
| `RG3_Gate_Active_Checks` | System | HARD | **PASS** | 8.31s | Fast gatekeeper executes active test scripts and passes --ci. |
| `RG4_Hardware_Accuracy` | System | HARD | **PASS** | 0.00s | Hardware profile correctly documents Blackwell SM 12.0 GDDR7 across papers and README. |
| `RG7_Repo_Hygiene` | System | HARD | **PASS** | 0.00s | Repository root is clean of release kit archives. |
| `RG8_Archive_Integrity` | System | HARD | **PASS** | 0.02s | RG8 certified: 70 release archives verified intact against SHA-256 manifest. |
