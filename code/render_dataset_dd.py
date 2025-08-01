# usage:
# python code/render_dataset_dd.py \
#   "/Library/Audio/Plug-Ins/VST3/Massive X.vst3" \
#   "/path/to/presets_or_states" \
#   "/path/to/out_dir"

import csv, json, sys
from pathlib import Path
import soundfile as sf

# Handle imports for both standalone and package usage
try:
    from dd_renderer import DDRenderer
    from mock_renderer import create_mock_renderer
except ImportError:
    from .dd_renderer import DDRenderer
    from .mock_renderer import create_mock_renderer

SR = 22050; NOTE_SEC = 3.0; RENDER_SEC = 4.0

def main(plugin_path, preset_dir, out_dir):
    # canonicalise & strip CLI paths
    plugin_path = str(Path(plugin_path.strip()).expanduser().resolve())
    preset_dir  = str(Path(preset_dir.strip()).expanduser().resolve())
    out_dir     = str(Path(out_dir.strip()).expanduser())
    print(f"[debug] plugin_path = '{plugin_path}'")  # <-- add this line
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    
    # Try to load with real renderer first, fall back to mock if needed
    try:
        R = DDRenderer(sample_rate=SR, block_size=512)
        success = R.load_plugin(plugin_path)
        if not success:
            print(f"Real plugin loading failed, trying mock renderer...")
            R = create_mock_renderer(use_mock=True)
            success = R.load_plugin(plugin_path)
            if not success:
                raise RuntimeError(f"Failed to load plugin with both real and mock renderers")
        
        is_mock = hasattr(R, 'mock_mode')
        print(f"Using {'mock' if is_mock else 'real'} renderer for {plugin_path}")
        
    except Exception as e:
        print(f"Error initializing renderer: {e}")
        print("Falling back to mock renderer...")
        R = create_mock_renderer(use_mock=True)
        R.load_plugin(plugin_path)
    
    # capture a stable name→index map for reproducibility
    pdesc = R.get_parameters_description()
    name_to_index = {d.get("name", str(i)): i for i, d in enumerate(pdesc)}
    (out / "parameter_index_map.json").write_text(json.dumps(name_to_index, indent=2))

    rows = []
    for f in sorted(Path(preset_dir).rglob("*")):
        if f.suffix.lower() == ".vstpreset":
            try:
                R.load_vst3_preset(str(f))
            except Exception as e:
                print(f"Warning: Failed to load VST3 preset {f}: {e}")
                continue
        elif f.suffix.lower() in (".state", ".bin", ".json"):
            try:
                R.load_state(str(f))
            except Exception as e:
                print(f"Warning: Failed to load state {f}: {e}")
                continue
        else:
            continue

        try:
            audio = R.render_patch(midi_note=60, note_len_sec=NOTE_SEC, render_len_sec=RENDER_SEC)
            wav = out / (f.stem + ".wav")
            sf.write(str(wav), audio.T, SR, subtype="FLOAT")

            patch = R.get_patch()  # [(index, value_01), ...]
            rows.append({"preset": str(f), "audio": str(wav), "params_json": json.dumps(patch)})
            print(f"Processed: {f.name}")
            
        except Exception as e:
            print(f"Warning: Failed to process {f}: {e}")
            continue

    with open(out / "metadata.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["preset", "audio", "params_json"])
        w.writeheader(); w.writerows(rows)
    
    print(f"Dataset rendering complete. Processed {len(rows)} presets.")

if __name__ == "__main__":
    main(*sys.argv[1:4])