#!/usr/bin/env python3
"""
Test loading ZENOLOGY presets and saving corresponding plugin state files.

This will:
 - Load the given plugin (AU or VST3) via DawDreamer wrapper (DDRenderer)
 - Discover presets under common macOS preset folders
 - For each preset, try to load it (load_vst3_preset for .vstpreset, load_preset otherwise)
 - Save a .state file next to the output directory for validation

Usage:
  python code/test_preset_state_roundtrip.py \
    --plugin "/Library/Audio/Plug-Ins/Components/ZENOLOGY.component" \
    --out datasets/zenology_preset_states \
    --limit 10
"""

import argparse
from pathlib import Path
import sys
import traceback

from dd_renderer import DDRenderer


COMMON_PRESET_DIRS = [
    # VST3 user presets
    Path.home() / "Library/Audio/Presets",
    # AU user presets (older location used by some hosts)
    Path.home() / "Music/AU Presets",
    # System-wide presets
    Path("/Library/Audio/Presets"),
]


def find_presets(product_names=("ZENOLOGY", "Zenology")):
    candidates = []
    for root in COMMON_PRESET_DIRS:
        if not root.exists():
            continue
        for pn in product_names:
            # VST3 preset folders are usually Vendor/Product/*.vstpreset
            for p in root.rglob(f"**/{pn}/*.vstpreset"):
                candidates.append(p)
            # AU presets often stored as .aupreset
            for p in root.rglob(f"**/{pn}/*.aupreset"):
                candidates.append(p)
    # De-duplicate while preserving order
    seen = set()
    ordered = []
    for c in candidates:
        k = str(c)
        if k in seen:
            continue
        seen.add(k)
        ordered.append(c)
    return ordered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plugin", required=True, help="Path to AU or VST3 plugin")
    ap.add_argument("--out", required=True, help="Directory to save .state outputs")
    ap.add_argument("--limit", type=int, default=10, help="Max presets to test")
    args = ap.parse_args()

    out_dir = Path(args.out)
    (out_dir / "state").mkdir(parents=True, exist_ok=True)

    print("[Load plugin]")
    R = DDRenderer(sample_rate=22050, block_size=512)
    R.load_plugin(args.plugin)

    print("[Discover presets]")
    presets = find_presets()
    if not presets:
        print("No presets found in common paths.")
        print("Searched:")
        for d in COMMON_PRESET_DIRS:
            print(f"  - {d}")
        sys.exit(0)

    tested = 0
    ok = 0
    failures = []
    for p in presets:
        if tested >= args.limit:
            break
        tested += 1
        try:
            is_vstpreset = p.suffix.lower() == ".vstpreset"
            print(f"[{tested}] Loading preset: {p}")
            if is_vstpreset:
                R.load_vst3_preset(str(p))
            else:
                R.load_preset(str(p))
            # Save state file for this preset
            state_path = out_dir / "state" / f"preset_{tested:04d}.state"
            R.save_state(str(state_path))
            print(f"  -> Saved state: {state_path}")
            ok += 1
        except Exception as e:
            print(f"  !! Failed: {e}")
            failures.append((str(p), repr(e)))
            traceback.print_exc()

    print("\nSummary:")
    print(f"  Tested:  {tested}")
    print(f"  Success: {ok}")
    print(f"  Failed:  {len(failures)}")
    if failures:
        print("Failures:")
        for f in failures:
            print("  -", f[0], "::", f[1])


if __name__ == "__main__":
    main()

