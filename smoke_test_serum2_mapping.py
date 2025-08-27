#!/usr/bin/env python3
"""
Smoke test for Serum 2 parameter mapping and normalization.

Runs a small set of converted presets, applies mapped parameters conservatively,
renders audio, and prints per-preset stats plus a summary:
- total params considered, mapped count, coverage
- audio max amplitude and success flag

Usage: python smoke_test_serum2_mapping.py [--limit 8]
"""

import argparse
import json
import os
from pathlib import Path
from statistics import mean
from typing import Dict, List

import numpy as np

try:
    import dawdreamer as daw
except Exception as e:
    raise SystemExit(f"Failed to import dawdreamer: {e}")

from serum2_parameter_mapping import (
    map_preset_to_vst_parameters,
    get_mapping_coverage,
)


def find_serum2_plugin() -> str:
    candidates = [
        "/Library/Audio/Plug-Ins/VST3/Serum2.vst3",
        os.path.expanduser("~/Library/Audio/Plug-Ins/VST3/Serum2.vst3"),
        "/Library/Audio/Plug-Ins/Components/Serum2.component",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return ""


def list_presets(root: Path, limit: int) -> List[Path]:
    items = list(root.rglob("*.json"))
    items.sort()
    return items[:limit]


def run_smoke_test(limit: int = 8):
    presets_dir = Path("converted_presets")
    if not presets_dir.exists():
        print(f"Preset directory not found: {presets_dir}")
        return 1

    plugin_path = find_serum2_plugin()
    if not plugin_path:
        print("Serum2 plugin not found. Please install or adjust path in script.")
        return 1

    presets = list_presets(presets_dir, limit)
    if not presets:
        print("No converted preset JSONs found.")
        return 1

    print("Serum 2 Mapping Smoke Test")
    print("=" * 40)
    print(f"Plugin: {plugin_path}")
    print(f"Presets to test: {len(presets)}\n")

    results = []

    for i, preset_path in enumerate(presets, 1):
        print(f"[{i}/{len(presets)}] {preset_path}")
        try:
            with open(preset_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  Error reading preset: {e}")
            continue

        dd_params: Dict[str, float] = data.get("dawdreamer_params", {})
        if not isinstance(dd_params, dict) or not dd_params:
            print("  No dawdreamer_params present; skipping")
            continue

        # Init engine+plugin fresh per preset to avoid state carry-over
        engine = daw.RenderEngine(44100, 512)
        synth = engine.make_plugin_processor("serum2", plugin_path)
        engine.load_graph([(synth, [])])

        # Build plugin param name->index map
        param_desc = synth.get_parameters_description()
        name_to_index = {p["name"]: p["index"] for p in param_desc}

        # Coverage metrics
        cov = get_mapping_coverage(dd_params)
        total_considered = cov["total_preset_params"]

        # Map and apply
        mapped = map_preset_to_vst_parameters(dd_params, name_to_index)
        applied = 0
        for idx, val in mapped:
            try:
                synth.set_parameter(idx, float(val))
                applied += 1
            except Exception as e:
                # Keep going on individual failures
                print(f"    Set param failed at {idx}: {e}")

        print(f"  params: considered={total_considered}, mapped={len(mapped)} ({(len(mapped)/total_considered*100.0 if total_considered else 0):.1f}%)")

        # Render a short note
        synth.clear_midi()
        synth.add_midi_note(60, 100, 0.0, 2.0)
        engine.render(3.0)
        audio = engine.get_audio()
        synth.clear_midi()

        ok = False
        max_amp = 0.0
        if hasattr(audio, "size") and audio.size > 0:
            max_amp = float(np.max(np.abs(audio)))
            ok = (max_amp > 1e-4) and np.isfinite(max_amp) and (max_amp < 2.0)

        print(f"  audio: ok={ok}, max={max_amp:.4f}")

        results.append({
            "preset": str(preset_path),
            "considered": total_considered,
            "mapped": len(mapped),
            "coverage_pct": (len(mapped)/total_considered*100.0 if total_considered else 0.0),
            "audio_ok": ok,
            "max_amp": max_amp,
        })

    # Summary
    if results:
        avg_cov = mean(r["coverage_pct"] for r in results)
        ok_count = sum(1 for r in results if r["audio_ok"]) 
        amps = [r["max_amp"] for r in results]
        print("\nSummary")
        print("-" * 40)
        print(f"Presets tested: {len(results)}")
        print(f"Audio OK: {ok_count}/{len(results)}")
        print(f"Avg mapped coverage: {avg_cov:.1f}%")
        if amps:
            print(f"Max amplitude: min={min(amps):.4f}, mean={mean(amps):.4f}, max={max(amps):.4f}")
    else:
        print("No results to summarize.")

    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()
    return run_smoke_test(args.limit)


if __name__ == "__main__":
    raise SystemExit(main())

