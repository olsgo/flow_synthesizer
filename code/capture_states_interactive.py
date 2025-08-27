#!/usr/bin/env python3
"""
Interactive state capture for AU/VST3 plugins (e.g., ZENOLOGY).

Opens the plugin editor so you can manually switch presets. After each
change, press Enter in this terminal to save a plugin state file and an audio
preview. Later, use build_dataset_from_states.py to render a full dataset.

Example:
  conda run -n flow-synth python code/capture_states_interactive.py \
    --plugin "/Library/Audio/Plug-Ins/Components/ZENOLOGY.component" \
    --out_dir datasets/zenology_captured
"""

import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np
import soundfile as sf

import dawdreamer as daw


def ensure_dirs(base: Path):
    (base / "state").mkdir(parents=True, exist_ok=True)
    (base / "audio").mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plugin", required=True, help="Path to AU/VST3 instrument")
    ap.add_argument("--out_dir", required=True, help="Directory to store captured states and previews")
    ap.add_argument("--sample_rate", type=int, default=44100)
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--note", type=int, default=60)
    ap.add_argument("--velocity", type=int, default=110)
    ap.add_argument("--note_len", type=float, default=2.0)
    ap.add_argument("--render_len", type=float, default=3.0)
    ap.add_argument("--window_title", type=str, default="PluginEditor", help="Window title for plugin editor")
    ap.add_argument("--sleep_after_change", type=float, default=0.25, help="Seconds to wait after UI change before render")
    args = ap.parse_args()

    out = Path(args.out_dir)
    ensure_dirs(out)

    engine = daw.RenderEngine(args.sample_rate, args.block_size)
    inst = engine.make_plugin_processor("inst", args.plugin)
    engine.load_graph([(inst, [])])

    try:
        inst.open_editor(args.window_title)
        print("Opened plugin editor. Bring the window to front.")
    except Exception as e:
        print(f"Warning: failed to open editor: {e}")

    print("\nInstructions:
  1) Use the plugin window to select a preset.
  2) Return to this terminal and press Enter to capture.
     Type 'q' + Enter to quit.
  3) Each capture saves a .dwstate and a short audio preview.\n")

    idx = 0
    while True:
        try:
            user = input(f"Capture #{idx} [Enter=save, q=quit, or type name]: ")
        except EOFError:
            break
        if user.strip().lower() == 'q':
            break

        # Allow the preset to settle
        time.sleep(args.sleep_after_change)
        engine.render(0.1)

        name = user.strip() if user.strip() else f"capture_{idx:04d}"
        state_p = out / "state" / f"{name}.dwstate"
        wav_p = out / "audio" / f"{name}.wav"

        # Save state
        try:
            inst.save_state(str(state_p))
            print(f"  Saved state: {state_p}")
        except Exception as e:
            print(f"  Failed to save state: {e}")

        # Render short audio
        inst.clear_midi()
        inst.add_midi_note(args.note, args.velocity, 0.0, args.note_len)
        engine.render(args.render_len)
        audio = engine.get_audio()
        mono = np.mean(audio, axis=0).astype(np.float32)
        peak = float(np.max(np.abs(mono))) if mono.size else 0.0
        try:
            sf.write(str(wav_p), mono, args.sample_rate)
            print(f"  Wrote audio: {wav_p}  peak={peak:.3f}")
        except Exception as e:
            print(f"  Failed to write audio: {e}")

        idx += 1

    print("Done capturing. Use code/build_dataset_from_states.py to build a dataset from the saved states.")


if __name__ == "__main__":
    main()

