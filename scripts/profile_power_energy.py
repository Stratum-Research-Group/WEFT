#!/usr/bin/env python3
"""
scripts/profile_power_energy.py
Hardware Energy & Power Profiler for Edge Workstations
Stratum Research Group • Nicholas Charles Mynhardt • Apache-2.0

Reads available hardware power telemetry:
- Linux RAPL /sys/class/powercap (if user has read access, e.g. via cap_sys_ptrace or root)
- NVIDIA NVML / nvidia-smi power telemetry for discrete GPU
- Reports measured vs restricted hardware metrics with full transparency.
"""

import os
import sys
import time
import subprocess
import json

def read_rapl_energy():
    rapl_path = "/sys/class/powercap/intel-rapl:0/energy_uj"
    if os.path.exists(rapl_path):
        try:
            with open(rapl_path, "r") as f:
                return int(f.read().strip()), "SUCCESS"
        except PermissionError:
            return None, "PERMISSION_DENIED_ROOT_REQUIRED"
    return None, "NOT_SUPPORTED"

def read_gpu_power():
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw,power.limit,temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        parts = [float(x.strip()) for x in res.stdout.strip().split(",")]
        return {
            "power_draw_w": parts[0],
            "power_limit_w": parts[1],
            "temp_c": parts[2],
            "status": "MEASURED"
        }
    except Exception as e:
        return {"error": str(e), "status": "UNAVAILABLE"}

def profile_inference_energy(duration_seconds=3):
    print("=================================================================")
    print("  WEFT HARDWARE ENERGY & POWER PROFILER")
    print("=================================================================")

    rapl_start, rapl_status = read_rapl_energy()
    gpu_data = read_gpu_power()

    print(f"[RAPL CPU Sensor] Status: {rapl_status}")
    print(f"[NVIDIA NVML GPU] Current Power Draw: {gpu_data.get('power_draw_w', 'N/A')} W (Limit: {gpu_data.get('power_limit_w', 'N/A')} W, Temp: {gpu_data.get('temp_c', 'N/A')} C)")

    # Execute active compute burst
    print(f"[COMPUTE] Profiling active compute loop for {duration_seconds}s...")
    t0 = time.time()
    count = 0
    while time.time() - t0 < duration_seconds:
        _ = [i * i for i in range(100000)]
        count += 1

    rapl_end, _ = read_rapl_energy()
    gpu_after = read_gpu_power()

    report = {
        "duration_seconds": duration_seconds,
        "cpu_rapl": {
            "sensor": "/sys/class/powercap/intel-rapl:0/energy_uj",
            "status": rapl_status,
            "note": "Requires root or CAP_SYS_RAWIO to read /sys/class/powercap/*/energy_uj directly." if rapl_status == "PERMISSION_DENIED_ROOT_REQUIRED" else "Read direct"
        },
        "gpu_telemetry": {
            "before": gpu_data,
            "after": gpu_after
        }
    }

    os.makedirs("results", exist_ok=True)
    with open("results/hardware_energy_profile.json", "w") as f:
        json.dump(report, f, indent=2)

    print("[PASS] Power profile written to results/hardware_energy_profile.json")

if __name__ == "__main__":
    profile_inference_energy()
