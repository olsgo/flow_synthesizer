# usage:
# python code/render_dataset_dd.py \
#   "/Library/Audio/Plug-Ins/VST3/Massive X.vst3" \
#   "/path/to/presets_or_states" \
#   "/path/to/out_dir"

import csv, json, sys
from pathlib import Path
import soundfile as sf
from dd_renderer import DDRenderer

SR = 22050; NOTE_SEC = 3.0; RENDER_SEC = 4.0

def main(plugin_path, preset_dir, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    R = DDRenderer(sample_rate=SR, block_size=512)
    R.load_plugin(plugin_path)

    # capture a stable name→index map for reproducibility
    pdesc = R.get_parameters_description()
    name_to_index = {d.get("name", str(i)): i for i, d in enumerate(pdesc)}
    (out / "parameter_index_map.json").write_text(json.dumps(name_to_index, indent=2))

    rows = []
    for f in sorted(Path(preset_dir).rglob("*")):
        if f.suffix.lower() == ".vstpreset":
            R.load_vst3_preset(str(f))
        elif f.suffix.lower() in (".state", ".bin"):
            R.load_state(str(f))
        else:
            continue

        audio = R.render_patch(midi_note=60, note_len_sec=NOTE_SEC, render_len_sec=RENDER_SEC)
        wav = out / (f.stem + ".wav")
        sf.write(str(wav), audio.T, SR, subtype="FLOAT")

        patch = R.get_patch()  # [(index, value_01), ...]
        rows.append({"preset": str(f), "audio": str(wav), "params_json": json.dumps(patch)})

    with open(out / "metadata.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["preset", "audio", "params_json"])
        w.writeheader(); w.writerows(rows)

if __name__ == "__main__":
    main(*sys.argv[1:4])