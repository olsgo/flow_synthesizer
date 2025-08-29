# PolyMAX Augmentation — Implementation Notes

This document summarizes the new PolyMAX dataset augmentation utilities and how they adhere to the standard dataset generation methodology in `DATASET_GENERATION_STANDARD.md`.

## Overview

- Individual processing: Each generated sample is rendered in an isolated process to prevent plugin state bleed and leftover tails.
- Pedalboard instrument hosting: Uses the UAD PolyMAX VST3 (`/Library/Audio/Plug-Ins/VST3/uaudio_polymax.vst3`).
- Dataset shape compatibility: Saves `raw/mel/mfcc/wav` with the same conventions used by `SynthesizerDataset`.
- Quality checks: Validates output WAVs (RMS, peak, tail RMS) and retries when needed.

## New Tools

- `tools/polymax_augment_worker.py` (per-sample worker)
  - Renders exactly one sample, then exits.
  - Modes:
    - `single`: sustained note.
    - `phrase`: multi-note phrase with `--bpm` and `--pattern` (up/down/random).
    - `sweep`: parameter sweeps across segments (e.g., `filter_cutoff_freq`, `filter_resonance`).
    - `preset_jitter`: load a random `.vstpreset`, jitter normalized params.
  - Outputs: `raw/*.npz`, `mel/*.npy`, `mfcc/*.npy`, `wav/*.wav` + sidecar `raw/*.json` with note events and automation.

- `tools/polymax_augment_isolated.py` (orchestrator)
  - Spawns a fresh worker process for each sample, per standard.
  - Adds quality validation (RMS > 0.005, peak < 0.99, tail sanity) and per-sample retries.
  - Inter-sample delay to encourage clean teardown.

- `tools/polymax_augment_dataset.py` (in-process, extended)
  - Same modes as the worker; helpful for quick local generation.
  - Not recommended for large production runs (prefer the isolated driver).

## Data Format

- Audio: mono 22050 Hz, 4.0 s default (`--duration`).
- Mel: `128 x 173` (power mel, `n_fft=2048, n_mels=128, hop=512, center=True, fmin=30, fmax=11000`) — matches `polymax_dataset`.
- MFCC: `13 x 173` (`n_mfcc=13, hop=512, center=True`) — matches `polymax_dataset`.
- Raw NPZ keys: `param` (dict name->normalized 0..1), `chars` (10x3 zeros), `audio` (float32).
- Sidecar JSON: `mode`, `note_events: [{t0,t1}]`, `automation: [{time,param,value}]`.

## Parameter Sampling

- Safe defaults for audibility (`osc_1_volume`, `amp_env_sustain`, `filter_cutoff_freq`, etc.).
- Strategies:
  - `jitter` (default): jitter around safe defaults; minimizes silent renders.
  - `uniform`: broad exploration (more silent/weak patches likely).
- Enforce `master_bypass=0` and `arp_enable=0` for clean renders.

## Usage Examples

1) Phrase dataset (standard compliant):

```bash
eval "$(conda shell.zsh hook)" && conda activate flow-synth
python tools/polymax_augment_isolated.py \
  --outdir datasets/polymax_phrases \
  --count 1000 \
  --mode phrase \
  --bpm 120 \
  --pattern random \
  --retries 1
```

2) Parameter sweep dataset:

```bash
python tools/polymax_augment_isolated.py \
  --outdir datasets/polymax_sweeps \
  --count 1000 \
  --mode sweep \
  --sweep-params filter_cutoff_freq,filter_resonance \
  --sweep-segments 4
```

3) Preset‑seeded jitter:

```bash
python tools/polymax_augment_isolated.py \
  --outdir datasets/polymax_preset_jitter \
  --count 1000 \
  --mode preset_jitter \
  --preset-dir "/Library/Audio/Presets/UADx PolyMAX Synth" \
  --jitter-std 0.05
```

## Integration Tips

- Place outputs under `datasets/polymax_dataset` to plug straight into the existing loaders (`--dataset polymax_dataset`).
- You can mix modes by running the isolated driver multiple times to the same `--outdir`.
- For large runs, use a job runner to parallelize across CPU cores while preserving one worker per process (do not reuse plugin instances across samples).

## Known Pitfalls

- Ensure PolyMAX is authorized and pedalboard is available in the active environment.
- The VST3 instrument call must use `reset=False` to retain state during a render; the isolated worker ensures clean state per sample.
- Silent output: the worker and generator implement audibility fallbacks (open filter, raise volumes) and retries on failure.

## File Map

- `tools/polymax_augment_worker.py` — per-sample isolated renderer.
- `tools/polymax_augment_isolated.py` — isolated orchestration + quality checks.
- `tools/polymax_augment_dataset.py` — in-process generator (dev use).
- `code/synth/pedalboard_synth.py:318` — `render_midi()` for phrase mode.

## Reproducibility

- All tools accept `--seed`; the isolated driver offsets seeds per sample (`seed + index`) and retries with an increment on failures.
