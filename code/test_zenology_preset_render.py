#!/usr/bin/env python3
"""
Quick test to validate ZENOLOGY can be:
- loaded via DawDreamer (AU or VST3)
- switched across presets (Bank/Program) via MIDI CC + Program Change
- saved as individual plugin state files
- rendered to short audio clips

Outputs under datasets/zenology_presets_test/{state,audio}.

Note: Bank MSB=85 and LSB=bank index follow prior working settings
for ZENOLOGY in this repo (see create_zenology_states.py).
"""

import os
import time
from pathlib import Path
import numpy as np
import soundfile as sf

import dawdreamer as daw
import mido


def find_plugin() -> str:
    candidates = [
        "/Library/Audio/Plug-Ins/VST3/ZENOLOGY.vst3",
        "/Library/Audio/Plug-Ins/Components/ZENOLOGY.component",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "ZENOLOGY plugin not found. Checked: " + ", ".join(candidates)
    )


def ensure_dirs(base: Path):
    (base / "state").mkdir(parents=True, exist_ok=True)
    (base / "audio").mkdir(parents=True, exist_ok=True)


def make_prog_change_file(path: Path, bank: int, program: int):
    mid = mido.MidiFile(type=0)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    # Bank select + program change
    track.append(mido.Message("control_change", channel=0, control=0, value=85, time=0))
    track.append(mido.Message("control_change", channel=0, control=32, value=bank, time=0))
    track.append(mido.Message("program_change", channel=0, program=program, time=0))
    mid.save(path)


def main():
    out = Path("datasets/zenology_presets_test")
    ensure_dirs(out)

    plugin_path = find_plugin()
    print(f"Using plugin: {plugin_path}")

    sr = 44100
    block = 256
    engine = daw.RenderEngine(sr, block)
    inst = engine.make_plugin_processor("zenology", plugin_path)
    engine.load_graph([(inst, [])])
    try:
        inst.open_editor("ZENOLOGY")
    except Exception:
        pass

    # small sweep for validation only
    banks = [0, 1]
    programs = list(range(0, 5))

    temp_mid = Path("temp_preset_change.mid")
    for b in banks:
        for p in programs:
            print(f"Switching to bank={b}, program={p}...")
            make_prog_change_file(temp_mid, b, p)
            inst.load_midi(str(temp_mid))
            # Allow preset to settle
            time.sleep(0.25)
            engine.render(0.1)

            # Save state
            state_path = out / "state" / f"zenology_bank_{b}_program_{p}.dwstate"
            try:
                inst.save_state(str(state_path))
                print(f"  Saved state: {state_path}")
            except Exception as e:
                print(f"  Failed to save state: {e}")

            # Render short audio clip with a middle-C note
            inst.clear_midi()
            inst.add_midi_note(60, 110, 0.0, 2.0)
            engine.render(3.0)
            audio = engine.get_audio()  # (channels, samples)
            mono = np.mean(audio, axis=0).astype(np.float32)
            wav_path = out / "audio" / f"zenology_bank_{b}_program_{p}.wav"
            try:
                sf.write(str(wav_path), mono, sr)
                print(f"  Wrote audio: {wav_path}  dur={len(mono)/sr:.2f}s  peak={np.max(np.abs(mono)):.3f}")
            except Exception as e:
                print(f"  Failed to write audio: {e}")

    # Cleanup temp MIDI
    try:
        if temp_mid.exists():
            temp_mid.unlink()
    except Exception:
        pass

    print("Done. Check datasets/zenology_presets_test/{state,audio}.")


if __name__ == "__main__":
    main()
