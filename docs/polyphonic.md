# Polyphonic Material Handling in Flow-Synth

This document describes the polyphonic capabilities added to Flow-Synth, enabling transcription and reconstruction of polyphonic audio using Basic Pitch and DawDreamer.

## Overview

The polyphonic pipeline extends Flow-Synth to handle multiple simultaneous notes by:

1. **Transcribing** polyphonic audio to MIDI using Spotify's Basic Pitch
2. **Post-processing** MIDI to handle timing, pitch bends, and voice separation
3. **Rendering** through DawDreamer-hosted synthesizers with polyphonic support

## Architecture

```
Audio Input → Basic Pitch → MIDI Post-Processing → DawDreamer → Audio Output
     ↓              ↓               ↓                   ↓
  WAV/MP3       Note Events    Cleaned MIDI      Synthesized Audio
                + Pitch Bends   + Channelization
```

## Key Components

### 1. Basic Pitch Integration (`flowsynth.transcription.basicpitch_backend`)

- **Polyphonic transcription**: Handles multiple simultaneous notes
- **Instrument-agnostic**: Works with any harmonic instrument
- **Pitch bend support**: Extracts pitch modulation information
- **Configurable parameters**: Frequency range, thresholds, note length

### 2. MIDI Post-Processing (`flowsynth.transcription.midi_tools`)

- **Timing quantization**: Reduces jitter and aligns to grid
- **Gap merging**: Combines notes separated by micro-gaps
- **Note cleaning**: Removes sub-threshold notes
- **Velocity smoothing**: Reduces abrupt velocity changes

### 3. Pitch Bend Handling

Three modes for managing pitch bends in polyphonic contexts:

- **`none`**: Strip all pitch bends (cleanest for timbre-focused tasks)
- **`global`**: Keep bends but warn about cross-talk (simple, may sound wrong)
- **`per_voice`**: Separate overlapping notes to different MIDI channels (recommended)

### 4. DawDreamer Scheduler (`flowsynth.render.dawdreamer_scheduler`)

- **Plugin management**: Registry-based plugin loading
- **Dual rendering modes**: Note-only (fast) vs. full MIDI (complete)
- **BPM control**: Beat-quantized rendering
- **Preset management**: VST3 preset and state loading

## Usage

### Command Line Interface

```bash
# Basic polyphonic reconstruction
python scripts/polyphonic_reconstruct.py \
  --audio input_chord.wav \
  --synth diva \
  --output reconstructed.wav

# With pitch bends and custom settings  
python scripts/polyphonic_reconstruct.py \
  --audio vibrato_performance.wav \
  --synth massive_x \
  --bend_mode per_voice \
  --channels 8 \
  --bpm 120 \
  --renderer_beats 8
```

### Python API

```python
from flowsynth.transcription.basicpitch_backend import transcribe
from flowsynth.transcription.midi_tools import process_midi_notes
from flowsynth.render.dawdreamer_scheduler import DawDreamerScheduler

# Transcribe audio
result = transcribe("audio.wav", min_freq=80, max_freq=2000)
notes = result['note_events']

# Process MIDI
processed_notes = process_midi_notes(
    notes, 
    bend_mode="per_voice",
    max_channels=8
)

# Render with synthesizer
scheduler = DawDreamerScheduler()
scheduler.load_plugin("diva")
audio = scheduler.render_notes_only(processed_notes, duration_seconds=4.0)
```

## Configuration

### Plugin Registry (`flowsynth/configs/synths.yaml`)

```yaml
plugins:
  diva:
    path: "/Library/Audio/Plug-Ins/VST/u-he/Diva.vst"
    type: "VST"
  massive_x:
    path: "/Library/Audio/Plug-Ins/VST3/Massive X.vst3"
    type: "VST3"

presets:
  diva:
    init: "/path/to/init.h2p"
    brass: "/path/to/brass.h2p"
```

### Processing Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `quantize_time` | 0.010s | Time quantization resolution |
| `min_note_duration` | 0.050s | Minimum note length |
| `merge_gap_threshold` | 0.030s | Maximum gap for note merging |
| `velocity_smoothing` | 0.1 | Velocity smoothing factor |

## Pitch Bend Details

### The Challenge

Standard MIDI pitch bend is **per-channel**, not per-note. When multiple notes play simultaneously on one channel, pitch bends affect all notes together, which is usually incorrect for polyphonic material.

### Solution: Per-Voice Channelization

The `per_voice` mode assigns overlapping notes to separate MIDI channels (up to 8 by default), allowing independent pitch bend curves:

```
Note A (C4): Channel 0 + Bend curve A
Note B (E4): Channel 1 + Bend curve B  [overlaps with A]
Note C (G4): Channel 0 + Bend curve C  [no overlap with A]
```

### Channel Assignment Algorithm

1. Sort notes by start time
2. For each note, find the first available channel
3. A channel is available if no other note is active during the new note's duration
4. If all channels are occupied, use channel 0 with a warning

## Performance Considerations

### Basic Pitch
- **Speed**: ~10× real-time on modern hardware
- **Memory**: Downmixes to mono, resamples to 22.05 kHz
- **Quality**: Best results with single instruments; mixed sources may need separation

### DawDreamer Rendering
- **Note-only mode**: Fastest, uses `add_midi_note()`
- **MIDI file mode**: Full features but requires temporary file creation
- **Plugin caching**: Reuse loaded plugins between renders

### Optimization Tips

```python
# Fast rendering (no pitch bends)
audio = scheduler.render_notes_only(notes, duration)

# Full rendering only when needed
if any(note.pitch_bends for note in notes):
    audio = scheduler.render_with_midi_file(notes, duration)
else:
    audio = scheduler.render_notes_only(notes, duration)
```

## Gotchas and Limitations

### 1. Plugin Pitch Bend Range
Different synthesizers have different pitch bend ranges (±2 semitones is common but not universal). Check your synth's documentation and configure accordingly.

### 2. Channel Limitations
Standard MIDI has 16 channels, but we default to 8 for `per_voice` mode to leave room for other uses. Highly polyphonic material may exceed available channels.

### 3. Basic Pitch Requirements
- Works best with harmonic instruments
- Percussive sources may not transcribe well
- Mixed sources benefit from source separation (e.g., Demucs) preprocessing

### 4. Timing Precision
Basic Pitch operates at 22.05 kHz with ~10ms frame resolution. Very fast ornaments may not be captured accurately.

### 5. Platform Dependencies
- **macOS**: Full AudioUnit + VST support
- **Windows**: VST/VST3 support
- **Linux**: VST support (varies by plugin)

## Testing

Run the test suite to verify functionality:

```bash
# Run all tests
python -m unittest discover tests/

# Run specific test modules
python -m unittest tests.test_midi_tools
python -m unittest tests.test_dawdreamer_scheduler
python -m unittest tests.test_basicpitch_transcribe
```

### Test Data Requirements

For golden-master testing, create short audio files with known content:

```
tests/audio/
  triads.wav          # Simple C major triad
  four_note_chord.wav # Extended chord
  vibrato_chord.wav   # Chord with pitch bends
```

## Troubleshooting

### Basic Pitch Installation Issues
```bash
# Install with specific Python version (3.10 recommended)
pip install basic-pitch

# On Apple Silicon, ensure compatible versions
pip install --upgrade basic-pitch
```

### Plugin Loading Failures
1. Verify plugin paths in `synths.yaml`
2. Check plugin compatibility with DawDreamer
3. Test plugin loading separately:

```python
import dawdreamer as daw
engine = daw.RenderEngine(22050, 512)
plugin = engine.make_plugin_processor("test", "/path/to/plugin.vst")
```

### Audio Output Issues
- Check sample rate compatibility (default: 22.05 kHz)
- Verify stereo vs. mono output handling
- Test with simple note-only rendering first

## Future Enhancements

### Planned Features
1. **MPE Support**: Multi-dimensional polyphonic expression
2. **Source Separation**: Automatic instrument isolation preprocessing
3. **Real-time Mode**: Low-latency live processing
4. **Advanced Channelization**: Smart voice leading algorithms

### Integration Points
- **Flow-Synth Parameter Optimization**: Use polyphonic MIDI with existing parameter estimation
- **Multi-instrument Support**: Separate transcription and rendering per instrument
- **Performance Analysis**: Polyphony histograms and timing analysis tools

## References

- [Basic Pitch Documentation](https://github.com/spotify/basic-pitch)
- [DawDreamer Documentation](https://github.com/DBraun/DawDreamer)
- [Flow-Synth Original Paper](https://arxiv.org/abs/1907.00971)
- [MIDI Specification](https://www.midi.org/specifications)