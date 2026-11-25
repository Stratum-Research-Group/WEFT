import glob
import os

mods = [f"`«{os.path.basename(f)[:-5]}»" for f in sorted(glob.glob("formal/*.lean"))]
roots_str = ",\n    ".join(mods)

lakefile_content = f"""import Lake
open Lake DSL

package «weft» where
  -- WEFT Formal Verification Suite
  -- Stratum Research Group • Nicholas Charles Mynhardt • Apache-2.0

@[default_target]
lean_lib «FormalSuite» where
  srcDir := "formal"
  roots := #[
    {roots_str}
  ]
"""

with open("lakefile.lean", "w", encoding="utf-8") as f:
    f.write(lakefile_content)

print(f"[OK] Generated lakefile.lean with {len(mods)} roots.")
