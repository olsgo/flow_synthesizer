# usage:
# python code/render_dataset_pb.py \
#   "/Library/Audio/Plug-Ins/VST3/Massive X.vst3" \
#   "/path/to/presets_or_states" \
#   "/path/to/out_dir"

import csv, json, sys, os
from pathlib import Path
import soundfile as sf
from pedalboard_renderer import PBRenderer

SR = 22050; NOTE_SEC = 3.0; RENDER_SEC = 4.0

def main(plugin_path, preset_dir, out_dir):
    # canonicalise & strip CLI paths
    plugin_path = str(Path(plugin_path.strip()).expanduser().resolve())
    preset_dir  = str(Path(preset_dir.strip()).expanduser().resolve())
    out_dir     = str(Path(out_dir.strip()).expanduser())
    print(f"[debug] plugin_path = '{plugin_path}'")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    R = PBRenderer(sample_rate=SR, buffer_size=512)
    
    # Load the plugin
    if not R.load_plugin(plugin_path):
        print(f"Failed to load plugin: {plugin_path}")
        return
    
    # capture a stable name→index map for reproducibility
    pdesc = R.get_parameters_description()
    name_to_index = {d.get("name", str(i)): i for i, d in enumerate(pdesc)}
    (out / "parameter_index_map.json").write_text(json.dumps(name_to_index, indent=2))

    rows = []
    for f in sorted(Path(preset_dir).rglob("*")):
        if f.suffix.lower() == ".vstpreset":
            # For VST presets, we would need plugin-specific loading
            # For now, skip these as Pedalboard doesn't have built-in VST preset loading
            print(f"Skipping VST preset (not yet supported): {f}")
            continue
        elif f.suffix.lower() == ".bin":
            # Load binary state file
            if R.load_state(str(f)):
                print(f"Loaded binary state: {f}")
            else:
                print(f"Failed to load binary state: {f}")
                continue
        elif f.suffix.lower() == ".json":
            # Load JSON parameter file
            try:
                with open(f, 'r') as json_file:
                    param_data = json.load(json_file)
                    # Convert JSON parameters to patch format
                    if isinstance(param_data, dict):
                        patch = []
                        for param_name, value in param_data.items():
                            if param_name in name_to_index:
                                patch.append((name_to_index[param_name], value))
                        R.set_patch(patch)
                        print(f"Loaded JSON parameters: {f}")
                    else:
                        print(f"Unexpected JSON format in {f}")
                        continue
            except Exception as e:
                print(f"Failed to load JSON parameters from {f}: {e}")
                continue
        elif f.suffix.lower() in (".state", ".fxb"):
            # These are DawDreamer-specific formats, skip for now
            print(f"Skipping DawDreamer-specific format: {f}")
            continue
        else:
            continue

        # Render audio with current plugin state
        try:
            audio = R.render_patch(midi_note=60, note_len_sec=NOTE_SEC, render_len_sec=RENDER_SEC)
            wav = out / (f.stem + ".wav")
            
            # Ensure audio is in the right format for soundfile
            if audio.ndim == 1:
                audio = audio.reshape(-1, 1)
            else:
                audio = audio.T  # Transpose from (channels, samples) to (samples, channels)
            
            sf.write(str(wav), audio, SR, subtype="FLOAT")
            print(f"Rendered audio: {wav}")

            # Get current patch state
            patch = R.get_patch()  # [(index, value_01), ...]
            
            # Create metadata row
            row = {
                "preset": str(f),
                "audio": str(wav), 
                "params_json": json.dumps(patch),
                "state_bin": str(f) if f.suffix.lower() == ".bin" else "",
                "plugin_path": plugin_path
            }
            rows.append(row)
            
        except Exception as e:
            print(f"Error rendering {f}: {e}")
            continue

    # Write metadata CSV
    if rows:
        fieldnames = ["preset", "audio", "params_json", "state_bin", "plugin_path"]
        with open(out / "metadata.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote metadata for {len(rows)} presets to {out / 'metadata.csv'}")
    else:
        print("No presets were successfully processed")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python render_dataset_pb.py <plugin_path> <preset_dir> <output_dir>")
        sys.exit(1)
    main(*sys.argv[1:4])