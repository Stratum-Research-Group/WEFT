# WEFT Architecture Suite Execution Summary

- **Run ID:** `20261125_133709_a794204`
- **Timestamp (UTC):** `2026-11-25T13:37:09.299129+00:00`
- **Git Commit:** `a794204acf0591f11303452b20bbb76774dbe5e1` (dirty=True)
- **Tier:** `standard` | **Track:** `all` | **Device:** `cpu`
- **Verdict:** **ALL GATES PASSED**

| Test ID | Track | Gate Type | Status | Duration (s) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `B1_Causality` | Track A | HARD | **PASS** | 0.07s | Causality strictly preserved across all positions and architecture variants. |
| `B2_Prefill_Decode_Parity` | Track A | HARD | **PASS** | 0.01s | Prefill vs Decode parity certified (max diff = 0.000000) |
| `B3_Cache_Formats` | Track A | HARD | **PASS** | 0.01s | Cache formats and state retention agree with full sequence recompute. |
| `B5_Exact_Recurrence` | Track A | HARD | **PASS** | 0.00s | Global path recurrence formulation matches mathematical definition. |
| `B8_Padding_Batching` | Track A | HARD | **PASS** | 0.00s | Padding and batching invariant certified (diff = 0.000000) |
| `C1_Format_Roundtrip` | Track B | HARD | **PASS** | 0.00s | Container format round-trip bit-exact preservation certified. |
| `C5_Arch_Labeling` | Track B | HARD | **PASS** | 0.00s | Track B results verified with 'transformer_converted' architecture label. |
| `RG1_No_Synthetic_Claims` | Track B | HARD | **PASS** | 0.00s | No synthetic 'zero-degradation' claims detected. |
| `RG3_Gate_Active_Checks` | System | HARD | **PASS** | 0.00s | Fast gatekeeper executes active test scripts. |
| `RG4_Hardware_Accuracy` | System | HARD | **PASS** | 0.00s | Hardware profile correctly documents Blackwell SM 12.0 GDDR7. |
| `RG7_Repo_Hygiene` | System | HARD | **PASS** | 0.00s | Repository root is clean of release kit archives. |
