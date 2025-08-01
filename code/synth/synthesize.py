#%%
# NOTE: This file now uses the DawDreamer backend (DDRenderer) instead of the
# legacy RenderMan implementation.

import argparse
import numpy as np
import json, ast
import librosa
import scipy
from dd_renderer import DDRenderer  # DawDreamer wrapper


def resample(y, orig_sr, target_sr):
    if orig_sr == target_sr:
        return y
    ratio = float(target_sr) / orig_sr
    n_samples = int(np.ceil(y.shape[-1] * ratio))
    y_hat = scipy.signal.resample(y, n_samples, axis=-1)
    return np.ascontiguousarray(y_hat, dtype=y.dtype)


def play_patch(engine, patch):
    """Render a single patch with DawDreamer engine and return audio."""
    engine.set_patch(patch)
    midi_note = 60
    midi_velocity = 127
    note_length = 3.0
    render_length = 4.0
    audio = engine.render_patch(
        midi_note=midi_note,
        velocity=midi_velocity,
        note_len_sec=note_length,
        render_len_sec=render_length,
    )
    # DawDreamer returns (channels, samples); convert to mono
    audio_mono = np.mean(audio, axis=0)
    return audio_mono, patch


def midiname2num(patch, rev_diva_midi_desc):
    """
    Convert parameter dictionary {param_name: value, ...} to a patch list
    of tuples [(param_number, value), ...]
    """
    return [(rev_diva_midi_desc[k], float(v)) for k, v in patch.items()]


def create_synth(dataset="toy", path="synth/diva.vst3"):
    with open("synth/diva_params.txt") as f:
        diva_midi_desc = ast.literal_eval(f.read())
    rev_idx = {diva_midi_desc[key]: key for key in diva_midi_desc}

    if dataset == "toy":
        with open("synth/param_nomod.json") as f:
            param_defaults = json.load(f)
    else:
        with open("synth/param_default_32.json") as f:
            param_defaults = json.load(f)

    engine = DDRenderer(sample_rate=44100, block_size=512)
    engine.load_plugin(path)
    return engine, param_defaults, rev_idx


def synthesize_audio(params, engine, params_default):
    patch = midiname2num(params, params_default)
    audio, _ = play_patch(engine, patch)
    return audio


def synthesize_batch(
    batch,
    param_names,
    engine,
    params_default,
    rev_idx,
    n_outs=2,
    orig_wave=None,
    name=None,
):
    final_audio = [None] * batch.shape[0]

    # (Optional) reset preset if you have a VST3 preset
    # engine.load_vst3_preset("synth/osc_reset.vstpreset")

    for b in range(batch.shape[0]):
        cur_params = batch[b]
        param_dict = params_default
        # Create dict out of params
        for p in range(len(cur_params)):
            param_dict[param_names[p]] = float(cur_params[p])
        final_audio[b] = synthesize_audio(param_dict, engine, rev_idx)
        final_audio[b] = resample(final_audio[b], 44100, 22050)

    if name is not None and orig_wave is not None:
        n_outs = min(n_outs, len(final_audio))
        orig_size = sum(int(o.shape[0]) for o in orig_wave[:n_outs]) + (2205 * n_outs)
        final_size = sum(int(f.shape[0]) for f in final_audio[:n_outs]) + (2205 * n_outs)
        wave_out = np.zeros(int(orig_size + final_size))
        cur_p = 0
        for b in range(n_outs):
            wave_out[cur_p : (cur_p + orig_wave[b].shape[0])] += orig_wave[b].numpy()
            cur_p += orig_wave[b].shape[0] + 2205
            wave_out[cur_p : (cur_p + final_audio[b].shape[0])] += final_audio[b]
            cur_p += final_audio[b].shape[0] + 2205
        librosa.output.write_wav(name + ".wav", wave_out, 22050)

    return final_audio


if __name__ == "__main__":
    """Sample program, generates default preset"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="/Users/esling/Datasets/diva_dataset",
        help="",
    )
    parser.add_argument("--output", type=str, default="outputs", help="")
    parser.add_argument("--dataset", type=str, default="toy", help="")
    parser.add_argument("--data", type=str, default="mel", help="")
    args = parser.parse_args()

    print("[Load the dataset]")
    loaded = np.load("synth/test_batch.npz")["arr_0"].item()
    fixed_data = loaded["fixed_data"]
    fixed_params = loaded["fixed_params"]
    fixed_meta = loaded["fixed_meta"]
    fixed_audio = loaded["fixed_audio"]

    print("[Create synth rendering]")
    final_params = [
        "ENV1: Decay",
        "VCF1: FilterFM",
        "OSC: Vibrato",
        "OSC: FM",
        "VCF1: Feedback",
        "ENV1: Attack",
        "ENV1: Sustain",
        "OSC: Volume3",
        "OSC: Volume2",
        "OSC: OscMix",
        "VCF1: Resonance",
        "VCF1: Frequency",
        "OSC: Tune3",
        "OSC: Tune2",
        "OSC: Shape1",
        "OSC: Shape2",
    ]

    engine, param_defaults, rev_idx = create_synth(args.dataset)

    print("[Synthesize batch]")
    audio = synthesize_batch(
        fixed_params[:16],
        final_params,
        engine,
        param_defaults,
        rev_idx,
        orig_wave=fixed_audio,
        name="check",
    )
    
