# Polyphonic Design Document

## Overview

This document describes the design and implementation of polyphonic material support for Flow Synth, enabling analysis and reconstruction of multi-note audio using the existing DawDreamer infrastructure.

## Motivation

The original Flow Synth focused on monophonic synthesis parameter inference from single notes. Real-world musical material often contains chords, harmonies, and overlapping notes. This polyphonic extension enables:

1. **Chord Analysis**: Transcription of simultaneous notes in harmonic progressions
2. **Polyphonic Reconstruction**: Rendering multiple overlapping notes through synthesizers
3. **Musical Context**: Better handling of realistic musical content beyond single notes
4. **Advanced Parameter Control**: Per-voice or global parameter inference for complex sounds

## Architecture

### Core Components

#### 1. Transcription Backend (`code/polyphonic/transcribe.py`)

**Purpose**: Convert audio files to structured note events

**Backends Supported**:
- **SimpleTestBackend**: Generates synthetic chord progressions for testing and development
- **OmnizartBackend**: Integration with Omnizart for general polyphonic instrument transcription
- **OnsetsFramesBackend**: Piano-focused transcription using Magenta's Onsets & Frames (stub)

**NoteEvent Representation**:
```python
class NoteEvent:
    pitch: int           # MIDI note number (0-127)
    onset_beats: float   # Start time in beats
    duration_beats: float # Duration in beats
    velocity: int        # Velocity (0-127)
    channel: int         # MIDI channel (0-15, for future MPE support)
```

**Output Formats**:
- JSON: Human-readable, supports metadata
- MIDI: Standard format for DAW compatibility (when mido available)

#### 2. Event Scheduling (`code/polyphonic/schedule.py`)

**Purpose**: Render note events using DawDreamer's polyphonic capabilities

**Rendering Modes**:

**Single-Instance Polyphony**:
- All notes sent to one synthesizer instance
- Leverages synthesizer's built-in polyphony
- Simpler setup, shared parameter space
- Best for synths with good polyphonic handling

**Multi-Instance Polyphony**:
- Separate synthesizer instance per voice
- Enables per-voice parameter control
- More complex routing through DawDreamer
- Future support for advanced parameter mapping

**Voice Assignment Algorithm**:
- Greedy assignment to minimize overlaps
- Configurable maximum voice count
- Fallback to earliest-ending voice when full

#### 3. Parameter Inference (`code/polyphonic/parameter_inference.py`)

**Purpose**: Infer synthesizer parameters from polyphonic audio

**Current Implementation**:
- Spectral feature extraction (centroid, bandwidth, rolloff, RMS)
- Heuristic mapping to synthesizer parameters
- Fallback when Flow Synth models unavailable

**Future Integration**:
- Load pre-trained Flow Synth models
- Mel spectrogram preprocessing matching training data
- Temporal parameter inference for time-varying sounds
- Per-voice parameter prediction in multi-instance mode

#### 4. End-to-End Pipeline (`code/pipeline_poly.py`)

**Purpose**: Complete audio → transcription → parameters → rendering workflow

**Pipeline Steps**:
1. Load and validate input audio and plugin
2. Transcribe audio to note events
3. Infer synthesizer parameters (optional)
4. Render polyphonic audio using DawDreamer
5. Save output audio and sidecar event data

**Modes**:
- `global-params`: Single parameter set for entire audio
- `temporal-params`: Time-varying parameters (future)

#### 5. Evaluation Framework (`code/polyphonic/evaluate.py`)

**Purpose**: Assess reconstruction quality with objective metrics

**Metrics**:
- **Spectral Convergence**: Frobenius norm ratio of original vs reconstructed spectrograms
- **Log Magnitude MSE**: Mean squared error in log magnitude domain
- **Note Diagnostics**: Count, polyphony distribution, pitch/duration statistics
- **Onset Error Analysis**: Timing accuracy when ground truth available

**Output**:
- HTML reports with visualizations and color-coded quality indicators
- JSON data for programmatic analysis

## Technical Decisions

### Transcription Backend Choice

**Primary: Omnizart**
- **Pros**: General polyphonic support, handles various instruments, chord detection
- **Cons**: Requires installation, potential dependency issues
- **Use Case**: General musical content, chord progressions, mixed instruments

**Alternative: Onsets & Frames**
- **Pros**: Excellent piano transcription, precise onset/offset detection
- **Cons**: Piano-specific, requires TensorFlow/Magenta setup
- **Use Case**: Piano-focused content, classical music, detailed timing

**Fallback: SimpleTestBackend**
- **Pros**: Always available, deterministic output, good for testing
- **Cons**: Not real transcription, synthetic data only
- **Use Case**: Development, testing, demonstrations

### DawDreamer Integration

**Beat-Based Timing**:
- Use `RenderEngine.set_bpm()` and `render(duration, beats=True)`
- Enables musical timing that adapts to tempo changes
- Consistent with musical notation and MIDI standards

**Parameter Management**:
- Maintain compatibility with existing `DDRenderer` API
- Parameter sets as `[(index, value_0_1), ...]` tuples
- Support for preset loading and state management

**Plugin Compatibility**:
- VST3/AU plugin support through DawDreamer
- Parameter discovery via `get_parameters_description()`
- Tested with Massive X, Polymax, FM8, Diva

### Data Representation

**NoteEvent Design**:
- Minimal, efficient representation
- Beat-based timing for musical relevance
- Extensible for future MPE/per-note control
- JSON serializable for storage and interchange

**Event Storage**:
- JSON format with metadata (BPM, etc.)
- MIDI export for DAW compatibility
- Separation of note data from audio files

## Implementation Milestones

### M1: Transcription Backbone ✅
- [x] NoteEvent class and serialization
- [x] SimpleTestBackend for development
- [x] CLI interface: `python -m code.polyphonic.transcribe`
- [x] JSON output format with beat-based timing
- [x] BPM detection and tempo handling

### M2: Event Scheduling ✅
- [x] PolyphonicRenderer class
- [x] Single-instance polyphonic rendering
- [x] Multi-instance voice assignment (basic)
- [x] CLI interface: `python -m code.polyphonic.schedule`
- [x] DawDreamer integration with beat-based timing

### M3: Parameter Inference ✅
- [x] Spectral feature extraction
- [x] Heuristic parameter mapping
- [x] Integration with pipeline
- [x] Graceful fallback when models unavailable
- [x] Foundation for Flow Synth model integration

### M4: Evaluation & QA ✅
- [x] Spectral quality metrics
- [x] Note event diagnostics
- [x] Polyphony analysis
- [x] HTML report generation
- [x] CLI interface: `python -m code.polyphonic.evaluate`

### M5: Documentation & Integration ✅
- [x] Design document (this document)
- [x] Updated requirements.txt
- [x] Test suite
- [x] Example usage and CLI interfaces

## Usage Examples

### Basic Transcription
```bash
# Generate test events from audio
python -m code.polyphonic.transcribe input.wav --backend test --out events.json

# With BPM specification
python -m code.polyphonic.transcribe input.wav --bpm 140 --out events.json
```

### Polyphonic Rendering
```bash
# Single-instance rendering
python -m code.polyphonic.schedule events.json \
  --plugin "/Library/Audio/Plug-Ins/VST3/Massive X.vst3" \
  --mode single_instance --out rendered.wav

# Multi-instance with voice limit
python -m code.polyphonic.schedule events.json \
  --plugin "/path/to/synth.vst3" \
  --mode multi_instance --max_voices 4 --out rendered.wav
```

### End-to-End Pipeline
```bash
# Complete reconstruction
python code/pipeline_poly.py input.wav \
  --plugin "/Library/Audio/Plug-Ins/VST3/Polymax.vst3" \
  --mode global-params --backend test --out reconstruction.wav

# With specific BPM and rendering mode
python code/pipeline_poly.py input.wav \
  --plugin "/path/to/synth.vst3" \
  --bpm 120 --renderer multi_instance --out reconstruction.wav
```

### Evaluation
```bash
# Compare original vs reconstruction
python -m code.polyphonic.evaluate original.wav reconstruction.wav \
  --events events.json --output report.html

# JSON output for analysis
python -m code.polyphonic.evaluate original.wav reconstruction.wav \
  --events events.json --format json --output metrics.json
```

## Future Enhancements

### Near-Term Improvements

1. **Full Omnizart Integration**
   - Robust installation and model loading
   - Error handling for various audio formats
   - Configurable transcription parameters

2. **Flow Synth Model Integration**
   - Load pre-trained models from `code/results/`
   - Proper mel spectrogram preprocessing
   - Model-based parameter inference

3. **Enhanced Multi-Instance Rendering**
   - DawDreamer Add processor for voice mixing
   - Per-voice parameter control
   - Advanced voice assignment algorithms

### Long-Term Extensions

1. **Temporal Parameter Inference**
   - Time-varying parameter prediction
   - Segment-based analysis for long audio
   - Keyframe interpolation

2. **MPE and Per-Note Control**
   - MIDI channel assignment for notes
   - Pitch bend and aftertouch support
   - Advanced synthesizer control

3. **Real-Time Integration**
   - Extend Ableton Live plugin for polyphonic input
   - Low-latency transcription and rendering
   - Live performance capabilities

4. **Advanced Evaluation**
   - Perceptual quality metrics
   - Musical structure analysis
   - Human listening studies

## Dependencies and Compatibility

### Required Dependencies
- `torch`, `librosa`, `numpy`: Core ML and audio processing
- `dawdreamer`: DawDreamer rendering engine
- `soundfile`: Audio I/O

### Optional Dependencies
- `omnizart`: Polyphonic transcription (recommended)
- `mido`: MIDI file support
- `pretty_midi`: Enhanced MIDI processing

### Platform Support
- **macOS**: Full support with AU/VST3 plugins
- **Linux**: DawDreamer support, limited plugin ecosystem
- **Windows**: DawDreamer support with VST3

### Plugin Compatibility
Tested with:
- Native Instruments Massive X
- U-he Diva
- Custom synthesizers via VST3/AU

## Performance Considerations

### Computational Complexity
- **Transcription**: O(n) with audio length, depends on backend
- **Parameter Inference**: O(1) per spectrogram chunk
- **Rendering**: O(n*v) with audio length and voice count
- **Evaluation**: O(n) with audio length

### Memory Usage
- Audio loaded entirely into memory
- Spectrograms cached during inference
- Multiple plugin instances in multi-voice mode

### Optimization Strategies
- Chunked processing for long audio
- Lazy loading of optional dependencies
- Efficient voice assignment algorithms
- Reuse of plugin instances when possible

## Testing Strategy

### Unit Tests
- `NoteEvent` serialization/deserialization
- Backend functionality with synthetic data
- Parameter inference with known inputs
- Evaluation metrics with controlled examples

### Integration Tests
- End-to-end pipeline with test audio
- Multi-component workflows
- Error handling and edge cases
- Cross-platform compatibility

### Performance Tests
- Large audio file handling
- High polyphony scenarios
- Memory usage profiling
- Rendering latency measurement

This design enables Flow Synth to handle realistic polyphonic musical material while maintaining compatibility with the existing codebase and extending capabilities for future research and applications.