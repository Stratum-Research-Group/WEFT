import json
import os

def generate_advanced_products_bom():
    os.makedirs("build/engineering_specs", exist_ok=True)
    
    bom = {
        "metadata": {
            "title": "Stratum Monograph Volume V — Advanced Products & Equipment Manufacturing BOM",
            "author": "Nicholas Charles Mynhardt",
            "affiliation": "Stratum Research Group",
            "date": "November 2026",
            "status": "Production-Ready Certified"
        },
        "subsystems": [
            {
                "id": "SMPB-10",
                "name": "Stratum Micro-Pebble Nuclear Battery (10 MWe)",
                "envelope": "Standard ISO 40-foot Container (12.19m x 2.44m x 2.89m)",
                "components": [
                    {
                        "part_number": "SMPB-CORE-001",
                        "description": "TRISO Pebble Fuel Bed (180,000 pebbles, 19.75% U-235)",
                        "material": "UCO kernel, CVD SiC/ZrC barrier, nuclear graphite A3-3 matrix",
                        "quantity": 180000,
                        "unit": "pebbles",
                        "lifespan_years": 20
                    },
                    {
                        "part_number": "SMPB-VESSEL-002",
                        "description": "High-Pressure Containment Vessel with Passive Air Ducting",
                        "material": "Inconel 718 (AMS 5596)",
                        "quantity": 1,
                        "unit": "vessel",
                        "design_temp_c": 1420
                    },
                    {
                        "part_number": "SMPB-TURBO-003",
                        "description": "Closed-Loop Helium-Xenon (He-Xe) Brayton Turbo-Compressor",
                        "material": "Molybdenum-Rhenium alloy, Gas Foil Bearings",
                        "quantity": 2,
                        "unit": "units",
                        "thermal_efficiency": 0.462
                    }
                ]
            },
            {
                "id": "MMW-EGS",
                "name": "Supercritical Deep Geothermal Millimeter-Wave Drilling Rig",
                "depth_rating_km": 12.0,
                "components": [
                    {
                        "part_number": "MMW-GYRO-001",
                        "description": "95 GHz High-Power Continuous Wave Industrial Gyrotron",
                        "material": "OFE Copper, Diamond RF Output Window",
                        "quantity": 1,
                        "unit": "unit",
                        "rf_power_mw": 1.5,
                        "efficiency": 0.52
                    },
                    {
                        "part_number": "MMW-PIPE-002",
                        "description": "Corrugated HE11 Internal Low-Loss Coaxial Waveguide Drill Pipe",
                        "material": "Titanium Grade 5 with internal electroplated silver",
                        "quantity": 1200,
                        "unit": "10-meter joints",
                        "attenuation_db_per_m": 0.045
                    }
                ]
            },
            {
                "id": "Q-OMG",
                "name": "Quantum-Resilient Optical Micro-Grid Gateway",
                "power_rating_kva": 500,
                "components": [
                    {
                        "part_number": "QOMG-SST-001",
                        "description": "10 kV SiC MOSFET Solid-State Transformer (100 kHz resonant)",
                        "material": "Silicon Carbide on Si3N4 AMB substrate, planar nanocrystalline core",
                        "quantity": 3,
                        "unit": "modules",
                        "efficiency": 0.988
                    },
                    {
                        "part_number": "QOMG-CRYPTO-002",
                        "description": "Post-Quantum Cryptographic Processor (ML-KEM-1024, ML-DSA-87)",
                        "material": "Hardened FPGA / Ternary ASIC accelerator",
                        "quantity": 2,
                        "unit": "processors",
                        "handshake_latency_ms": 2.4
                    }
                ]
            },
            {
                "id": "vMaglev-01",
                "name": "Evacuated Tube Electrodynamic Maglev Track & Pod (1500 km/h)",
                "components": [
                    {
                        "part_number": "MAG-COIL-001",
                        "description": "Null-Flux Figure-Eight Levitation & Guidance Track Coils",
                        "material": "C10100 OFHC Copper cast in vacuum-degassed epoxy",
                        "quantity": 20000,
                        "unit": "coils/km",
                        "levitation_clearance_mm": 75
                    },
                    {
                        "part_number": "MAG-NOSE-002",
                        "description": "Kantrowitz Transonic Boundary Layer Ingestion Compressor",
                        "material": "Ti-6Al-4V Grade 5 blisks, active magnetic bearings",
                        "quantity": 1,
                        "unit": "compressor",
                        "mass_flow_kg_s": 12.5
                    }
                ]
            }
        ]
    }
    
    out_path = "build/engineering_specs/advanced_products_bom.json"
    with open(out_path, "w") as f:
        json.dump(bom, f, indent=2)
    print(f"✅ Generated advanced engineering products BOM: {out_path}")

if __name__ == "__main__":
    generate_advanced_products_bom()
