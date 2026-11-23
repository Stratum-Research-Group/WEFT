#!/usr/bin/env python3
"""
WEFT Interactive Shell:
Terminal-native REPL for conversing and problem-solving with the 10-model sub-bit collective.
"""

import sys
import time
from weft.omni_console import OmniFleetConsole

def main():
    console = OmniFleetConsole()
    print("=" * 65)
    print("  WEFT NATIVE INTERACTIVE SHELL (v3.3.0)")
    print("  10 Edge Foundation Models Online | RAM: 1.35 GB | Zero Cloud")
    print("  Type 'exit', 'quit', or 'dashboard' to inspect fleet status.")
    print("=" * 65)

    while True:
        try:
            prompt = input("\nweft> ").strip()
            if not prompt:
                continue
            if prompt.lower() in ("exit", "quit"):
                print("Exiting WEFT Shell. Goodbye!")
                break
            if prompt.lower() == "dashboard":
                print(console.render_dashboard())
                continue

            t0 = time.perf_counter()
            res = console.route_query(prompt)
            latency = (time.perf_counter() - t0) * 1000

            print(f"\n[{res['selected_engine']}] (Latency: {res['latency_ms']:.1f} ms):")
            print("-" * 50)
            print(res['result'])
            print("-" * 50)

        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break

if __name__ == "__main__":
    main()
