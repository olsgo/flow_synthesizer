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

### 3. Polyphonic Rendering (`code/polyphonic/`)

Extended polyphonic support building on DawDreamer's capabilities:

**PolyphonicRenderer** (`code/polyphonic/schedule.py`):
```python
from code.polyphonic.schedule import PolyphonicRenderer

# Initialize polyphonic renderer
renderer = PolyphonicRenderer(sample_rate=22050)
renderer.load_plugin("/path/to/synth.vst3")

# Render multiple simultaneous notes
events = [
    NoteEvent(pitch=60, onset_beats=0.0, duration_beats=2.0),
    NoteEvent(pitch=64, onset_beats=0.0, duration_beats=2.0),
    NoteEvent(pitch=67, onset_beats=0.0, duration_beats=2.0)
]
audio = renderer.render_single_instance(events, duration_beats=4.0, bpm=120.0)
```

**Polyphonic Pipeline** (`code/pipeline_poly.py`):
```bash
# End-to-end polyphonic reconstruction
python code/pipeline_poly.py input.wav \
  --plugin "/Library/Audio/Plug-Ins/VST3/Massive X.vst3" \
  --mode global-params --backend test --out reconstruction.wav
```

## Audio Specifications

The implementation maintains the same audio specifications as the original:
- MIDI note: C4 (60)
- Note duration: 3.0 seconds
- Render duration: 4.0 seconds  
- Sample rate: 22050 Hz
- Mel spectrogram: 128 bins, FFT 2048, hop 1024

For polyphonic rendering:
- Beat-based timing using `RenderEngine.set_bpm()` and `render(duration, beats=True)`
- Multiple note scheduling via `PluginProcessor.add_midi_note()`
- Support for overlapping notes and chord progressions

## VST3 Plugin Support

### Massive X
- Supports VST3 presets (`.vstpreset` files)
- Host automation via 16 Macros plus pitch-bend, mod wheel, aftertouch
- Typical path: `/Library/Audio/Plug-Ins/VST3/Massive X.vst3` (macOS)
- Polyphonic capability: Up to 32 voices

### Diva
- Classic analog modeling synthesizer
- Polyphonic operation with vintage-style voice allocation
- Path: `/Library/Audio/Plug-Ins/VST/u-he/Diva.vst`

### Other Synthesizers
Any VST3 synthesizer can be used by:
1. Providing the correct plugin path to rendering scripts
2. Ensuring presets are in supported formats (`.vstpreset`, `.state`, `.bin`)
3. Verifying polyphonic capability for multi-note rendering

## Polyphonic Features

### Beat-Based Timing

DawDreamer's beat-based rendering enables musical timing:

```python
# Set BPM and render in beats
engine.set_bpm(120.0)
plugin.add_midi_note(note=60, velocity=100, start_beat=0.0, duration_beats=2.0, beats=True)
engine.render(duration_beats=4.0, beats=True)
```

Benefits:
- Musical timing that adapts to tempo changes
- Consistent with MIDI and music notation
- Enables tempo-independent parameter automation

### Single vs Multi-Instance Rendering

**Single-Instance Mode**:
- All notes sent to one synthesizer instance
- Leverages plugin's built-in polyphony
- Shared parameter space across all voices
- Most efficient for standard polyphonic synths

**Multi-Instance Mode**:
- Separate plugin instance per voice
- Enables per-voice parameter control
- More complex routing and higher CPU usage
- Future support for advanced voice-specific effects

### Voice Assignment

Intelligent voice assignment for multi-instance mode:
- Greedy algorithm to minimize note overlaps
- Configurable maximum voice count
- Fallback to earliest-ending voice when full
- Future support for voice stealing algorithms

## Migration Benefits

1. **Modern VST3 support**: Better compatibility with current synthesizers
2. **No shared library dependencies**: DawDreamer is pip-installable
3. **Cross-platform**: Works on macOS, Windows, and Linux
4. **Headless operation**: No GUI requirements for batch processing
5. **Parameter introspection**: Better parameter discovery and mapping
6. **Polyphonic capabilities**: Native support for chord analysis and rendering
7. **Beat-based timing**: Musical tempo handling for complex arrangements

## Compatibility

The new implementation is fully compatible with existing training/evaluation scripts:
- Same metadata format (with `params_json` containing parameter arrays)
- Same audio specifications
- No changes required to model training code
- Extended capabilities for polyphonic material

## Dependencies

Added to `requirements.txt`:
- `dawdreamer`: Audio plugin hosting
- `soundfile`: Audio file I/O
- `mido`: MIDI file support (optional)
- `omnizart`: Polyphonic transcription (optional)

## Testing

### Monophonic Tests
Run the validation tests:
```bash
python test_massive_x_au.py      # Basic plugin loading
python test_polymax_au.py        # Alternative synthesizer
python test_fm8_au.py            # FM synthesis
```

### Polyphonic Tests
```bash
python test_polyphonic.py        # Core polyphonic functionality
python -m code.polyphonic.transcribe test_audio.wav
python -m code.polyphonic.evaluate original.wav reconstruction.wav
```

## Performance Considerations

### Latency
- Plugin latency handling via DawDreamer's automatic compensation
- Preroll rendering for accurate timing
- Buffer size configuration for real-time vs batch processing

### CPU Usage
- Single-instance mode: Similar to monophonic rendering
- Multi-instance mode: Linear scaling with voice count
- Plugin-dependent efficiency (some synths optimize polyphony better)

### Memory
- Multiple plugin instances increase memory usage
- Audio buffers scale with render duration and voice count
- Recommendation: Monitor memory usage with high voice counts

## Troubleshooting

### Plugin Loading Issues
1. Verify plugin path exists and is accessible
2. Check plugin format compatibility (VST3/AU)
3. Ensure plugin supports host automation
4. Test with simpler plugins first

### Audio Quality Issues
1. Check sample rate consistency (22050 Hz)
2. Verify parameter ranges (0.0-1.0 for normalized values)
3. Monitor for clipping with multiple voices
4. Test with different buffer sizes

### Timing Issues
1. Ensure beat-based rendering is enabled
2. Verify BPM settings are consistent
3. Check note onset/offset timing in events
4. Test with simpler note patterns first

### Performance Issues
1. Reduce voice count in multi-instance mode
2. Use single-instance mode for standard polyphony
3. Increase buffer size for batch processing
4. Profile CPU usage per plugin

This enhanced DawDreamer integration provides a solid foundation for both monophonic and polyphonic synthesis workflows while maintaining compatibility with existing Flow Synth research and applications.