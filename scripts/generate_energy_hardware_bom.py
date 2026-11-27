#!/usr/bin/env python3
"""
Generate Precision Engineering Bill of Materials (BOM) & Equipment Specifications
for the Stratum Volume V Aneuronic Fusion & REBCO Prototype Cusp Testbed.

Author: Nicholas Charles Mynhardt
Affiliation: Stratum Research Group
"""

import json
import sys
from pathlib import Path

def generate_energy_bom():
    out_dir = Path("build/engineering_specs")
    out_dir.mkdir(parents=True, exist_ok=True)

    bom = {
        "project": "Stratum Volume V: Aneuronic Cusp Direct Energy Conversion Testbed",
        "lead_author": "Nicholas Charles Mynhardt",
        "research_entity": "Stratum Research Group",
        "chamber_subsystems": [
            {
                "subsystem": "Ultra-High Vacuum (UHV) Vessel",
                "material": "Inconel 718 / 316L Stainless Steel (Non-Magnetic)",
                "base_pressure_torr": 1e-9,
                "ports": "8x CF150 ConFlat Flanges with OFHC Copper Gaskets",
                "turbomolecular_pumping_lps": 2200
            },
            {
                "subsystem": "High-Temperature Superconducting (HTS) Coils",
                "material": "2G REBCO (Rare-Earth Barium Copper Oxide) Tape",
                "tape_width_mm": 12.0,
                "operating_temperature_k": 77.0,
                "cooling": "Liquid Nitrogen (LN2) Closed-Loop Cryostat",
                "confining_field_tesla": 14.5,
                "total_tape_length_meters": 1850
            },
            {
                "subsystem": "Electrostatic Alpha Decelerator Collector",
                "grid_stages": 4,
                "grid_voltages_kv": [15, 30, 45, 60],
                "collector_material": "Tungsten Mesh / Molybdenum Sputter Baffles",
                "expected_direct_conversion_efficiency": 0.864
            },
            {
                "subsystem": "Solid-State InGaAsSb Thermophotovoltaic Array",
                "cell_architecture": "Molecular Beam Epitaxy (MBE) Grown InGaAsSb/GaSb",
                "back_surface_reflector": "Gold Thin-Film (98.5% IR Reflectance)",
                "thermal_emitter_material": "Tungsten Micro-Photonic Crystal (1400 K)"
            }
        ],
        "compliance": "Lean 4 Certified Invariants (Zero Axioms)"
    }

    out_file = out_dir / "volume_v_equipment_bom.json"
    out_file.write_text(json.dumps(bom, indent=2), encoding="utf-8")
    print(f"✅ Generated Stratum Volume V Hardware Engineering BOM: {out_file}")
    return 0

if __name__ == "__main__":
    sys.exit(generate_energy_bom())
