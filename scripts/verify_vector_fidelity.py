#!/usr/bin/env python3
"""
Verify XML and geometric fidelity of all vector SVG figures across docs/figures/.
Ensures viewBox validity, XML parseability, and zero NaN/infinite coordinate values.
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

FIGURES_DIR = Path(__file__).resolve().parent.parent / "docs" / "figures"

def verify_svgs():
    print("=== Verifying Vector SVG Graphics Fidelity ===")
    svg_files = sorted(list(FIGURES_DIR.glob("*.svg")))
    print(f"Auditing {len(svg_files)} vector SVG files in {FIGURES_DIR}...\n")

    if len(svg_files) != 190:
        print(f"Warning: Expected exactly 190 SVG files (95 Fig 1 + 95 Fig 2), found {len(svg_files)}.")

    failures = []
    for svg in svg_files:
        try:
            tree = ET.parse(svg)
            root = tree.getroot()
            if not root.tag.endswith("svg"):
                failures.append((svg.name, "Root tag is not <svg>"))
                continue
            viewbox = root.attrib.get("viewBox") or root.attrib.get("viewbox")
            if not viewbox:
                failures.append((svg.name, "Missing viewBox attribute"))
                continue
            # Verify standard bounding box aspects
            vb_parts = [float(x) for x in viewbox.replace(",", " ").split() if x.strip()]
            if len(vb_parts) == 4:
                min_x, min_y, width, height = vb_parts
                if width <= 0 or height <= 0:
                    failures.append((svg.name, f"Invalid non-positive dimensions: {width}x{height}"))
                    continue
                aspect_ratio = width / height
                if aspect_ratio < 0.2 or aspect_ratio > 5.0:
                    failures.append((svg.name, f"Extreme aspect ratio ({aspect_ratio:.2f}) outside [0.2, 5.0]"))
                    continue
            else:
                failures.append((svg.name, f"Malformed viewBox: {viewbox}"))
                continue
            raw_text = svg.read_text(encoding="utf-8")
            if "NaN" in raw_text or "Infinity" in raw_text:
                failures.append((svg.name, "Found NaN or Infinity coordinate values"))
        except Exception as e:
            failures.append((svg.name, f"XML Parse Error: {e}"))

    if failures:
        print(f"❌ FAILED: {len(failures)} SVG files failed fidelity check:")
        for name, err in failures:
            print(f"  - {name}: {err}")
        return 1
    else:
        print(f"✅ ALL {len(svg_files)} VECTOR SVGS VERIFIED: 100% valid XML and coordinates.")
        return 0

if __name__ == "__main__":
    sys.exit(verify_svgs())
