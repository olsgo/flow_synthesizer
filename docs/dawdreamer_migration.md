# DawDreamer Migration

This document describes the migration from RenderMan to DawDreamer for audio synthesis in the flow_synthesizer project.

## Overview

The original implementation used RenderMan for batch audio generation. This has been replaced with DawDreamer, which provides better VST3 support and more modern audio hosting capabilities.

## New Components

### 1. DDRenderer (`code/dd_renderer.py`)

A drop-in replacement for RenderMan functionality that wraps DawDreamer's API:

```python
from dd_renderer import DDRenderer

# Initialize renderer
renderer = DDRenderer(sample_rate=22050, block_size=512)

# Load a VST3 plugin
renderer.load_plugin("/path/to/plugin.vst3")

# Get parameter information
param_count = renderer.get_plugin_parameter_size()
param_desc = renderer.get_parameters_description()

# Set parameters
renderer.set_patch([(0, 0.5), (1, 0.7)])  # (index, value_0_1)

# Render audio
audio = renderer.render_patch(midi_note=60, note_len_sec=3.0, render_len_sec=4.0)
```

### 2. Dataset Renderer (`code/render_dataset_dd.py`)

A CLI script for batch rendering presets to create training datasets:

```bash
python code/render_dataset_dd.py \
  "/Library/Audio/Plug-Ins/VST3/Massive X.vst3" \
  "/path/to/presets" \
  "/path/to/output"
```

This generates:
- `.wav` files for each preset (22050 Hz, 4s duration)
- `metadata.csv` with columns: `preset`, `audio`, `params_json`
- `parameter_index_map.json` for reproducible parameter mapping

## Audio Specifications

The implementation maintains the same audio specifications as the original:
- MIDI note: C4 (60)
- Note duration: 3.0 seconds
- Render duration: 4.0 seconds  
- Sample rate: 22050 Hz
- Mel spectrogram: 128 bins, FFT 2048, hop 1024

## VST3 Plugin Support

### Massive X
- Supports VST3 presets (`.vstpreset` files)
- Host automation via 16 Macros plus pitch-bend, mod wheel, aftertouch
- Typical path: `/Library/Audio/Plug-Ins/VST3/Massive X.vst3` (macOS)

### Other Synthesizers
Any VST3 synthesizer can be used by:
1. Providing the correct plugin path to `render_dataset_dd.py`
2. Ensuring presets are in supported formats (`.vstpreset`, `.state`, `.bin`)

## Migration Benefits

1. **Modern VST3 support**: Better compatibility with current synthesizers
2. **No shared library dependencies**: DawDreamer is pip-installable
3. **Cross-platform**: Works on macOS, Windows, and Linux
4. **Headless operation**: No GUI requirements for batch processing
5. **Parameter introspection**: Better parameter discovery and mapping

## Compatibility

The new implementation is fully compatible with existing training/evaluation scripts:
- Same metadata format (with `params_json` containing parameter arrays)
- Same audio specifications
- No changes required to model training code

## Dependencies

Added to `requirements.txt`:
- `dawdreamer`: Audio plugin hosting
- `soundfile`: Audio file I/O

## Testing

Run the validation tests:
```bash
python test_dd_smoke.py      # Basic functionality
python test_dd_validation.py # Comprehensive validation
```