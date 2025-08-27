# Flow Synthesizer Serum Integration

This document describes the integration of Serum wavetable synthesizer support into the Flow Synthesizer framework, expanding beyond the original Diva-only implementation.

## Overview

The Flow Synthesizer now supports both Diva and Serum synthesizers, allowing users to leverage the power of normalizing flows with Serum's advanced wavetable synthesis capabilities. This integration maintains full backward compatibility with existing Diva workflows while adding comprehensive Serum support.

## What's New

### Serum Support
- Complete parameter mapping for Serum's synthesizer parameters (128 parameters)
- Default parameter configurations optimized for Serum
- Multi-synthesizer architecture that can be extended to other synthesizers

### Multi-Synthesizer Architecture
- Unified interface supporting multiple synthesizer backends
- Runtime synthesizer switching via OSC commands
- Synthesizer-specific parameter validation and handling

## Features

### Supported Synthesizers
1. **Diva** (original support maintained)
   - 281 parameters mapped
   - All existing functionality preserved
   - Compatible with existing models and presets

2. **Serum** (new)
   - 128 core parameters mapped including:
     - Oscillator A & B (wavetable position, warp, unison, etc.)
     - Sub oscillator and noise generator
     - Filter section with multiple types
     - Envelope generators (ADSR)
     - LFOs with various shapes
     - Effects rack (chorus, reverb, delay, distortion)
     - Modulation matrix
     - Global settings and arpeggiator

### Key Capabilities
- **Wavetable Control**: Direct manipulation of wavetable position and warp parameters
- **Advanced Modulation**: Full access to Serum's modulation matrix
- **Effects Processing**: Control over Serum's built-in effects
- **Real-time Parameter Control**: Live manipulation via OSC interface

## Installation & Setup

### Prerequisites
- Serum VST plugin installed
- Python environment with flow_synthesizer dependencies
- Compatible DAW (tested with Ableton Live)

### Serum Plugin Setup
Ensure Serum is installed in the standard VST location:
```
/Library/Audio/Plug-Ins/VST/Xfer/Serum.vst
```

### Configuration Files
The integration includes:
- `serum_params.txt`: Parameter ID mapping
- `serum_param_default.json`: Default parameter values

## Usage

### Command Line
Select synthesizer type when training or running:

```bash
# Using Serum
python train.py --synth_type serum --dataset my_dataset

# Using Diva (default, backward compatible)
python train.py --synth_type diva --dataset my_dataset
```

### OSC Interface
Switch synthesizers at runtime:

```
/set_synth_type serum
/set_synth_type diva
```

### Programmatic Usage
```python
from synth.synthesize import create_synth

# Create Serum synthesizer
engine, generator, defaults, rev_idx = create_synth(
    dataset='my_dataset', 
    synth_type='serum'
)

# Create Diva synthesizer  
engine, generator, defaults, rev_idx = create_synth(
    dataset='my_dataset', 
    synth_type='diva'
)
```

## Parameter Mapping

### Serum Parameter Categories
1. **Oscillators**: Wavetable synthesis with position and warp controls
2. **Filter**: Multiple filter types with resonance and drive
3. **Envelopes**: Standard ADSR with additional curve controls
4. **LFOs**: Rate, shape, and amount controls
5. **Effects**: Comprehensive effects processing
6. **Modulation**: Matrix-based modulation routing
7. **Global**: Master controls and arpeggiator

### Example Parameter Access
```python
# Key Serum parameters for synthesis control
serum_params = {
    "OSC A: Volume": 0.8,
    "OSC A: Wavetable Position": 0.3,
    "OSC A: Warp": 0.5,
    "FILTER: Cutoff": 0.7,
    "FILTER: Resonance": 0.2,
    "ENV1: Attack": 0.01,
    "ENV1: Decay": 0.3,
    "ENV1: Sustain": 0.7,
    "ENV1: Release": 0.3
}
```

## Technical Details

### Implementation
- Extended `create_synth()` function with synthesizer type parameter
- Added synthesizer-specific parameter loading
- Maintained librenderman compatibility
- OSC server enhanced with synthesizer switching

### File Structure
```
code/synth/
├── diva_params.txt          # Diva parameter mapping (existing)
├── serum_params.txt         # Serum parameter mapping (new)
├── param_default_32.json    # Diva defaults (existing)
├── serum_param_default.json # Serum defaults (new)
└── synthesize.py           # Multi-synth synthesis engine (updated)
```

### Backward Compatibility
- All existing Diva functionality preserved
- Default synthesizer type remains 'diva'
- Existing scripts work without modification
- OSC interface maintains existing commands

## Research Applications

### Wavetable Flow Synthesis
Serum's wavetable capabilities open new research directions:
- **Wavetable Position Control**: Direct manipulation of wavetable scanning
- **Spectral Morphing**: Smooth transitions between wavetable frames
- **Harmonic Content Control**: Precise control over harmonic structure

### Advanced Modulation Studies
- **Matrix Modulation**: Complex routing between sources and destinations
- **LFO Shape Analysis**: Impact of different LFO shapes on synthesis
- **Multi-Source Modulation**: Combining multiple modulation sources

### Comparative Synthesis Studies
- **Diva vs Serum**: Comparative analysis of analog vs wavetable synthesis
- **Parameter Space Mapping**: Different parameter relationships between synthesizers
- **Timbral Comparison**: Spectral analysis across synthesizer types

## Examples

### Basic Serum Synthesis
```python
# Load Serum with basic wavetable sound
engine, generator, defaults, rev_idx = create_synth('toy', 'serum')

params = {
    "OSC A: Volume": 0.8,
    "OSC A: Wavetable Position": 0.2,
    "FILTER: Cutoff": 0.6,
    "ENV1: Attack": 0.1,
    "ENV1: Release": 0.5
}

audio = synthesize_audio(params, engine, generator, rev_idx)
```

### Cross-Synthesizer Parameter Mapping
```python
# Map flow representations between synthesizers
diva_engine, diva_gen, diva_def, diva_idx = create_synth('dataset', 'diva')
serum_engine, serum_gen, serum_def, serum_idx = create_synth('dataset', 'serum')

# Encode audio with Diva, decode with Serum for cross-synthesis
```

## Limitations & Future Work

### Current Limitations
- Serum plugin path may need manual configuration
- Some advanced Serum features not yet mapped
- Real-time performance depends on host system

### Future Enhancements
- Additional synthesizer support (Massive, Sylenth1, etc.)
- Advanced wavetable import/export
- Real-time wavetable scanning control
- Enhanced modulation matrix support

## Troubleshooting

### Common Issues
1. **Plugin Not Found**: Verify Serum installation path
2. **Parameter Mismatch**: Check parameter names match serum_params.txt
3. **Audio Artifacts**: Ensure proper buffer settings in host

### Debug Mode
Enable debug output for troubleshooting:
```python
server = FlowServer(debug=True, synth_type='serum')
```

## Contributing

To add support for additional synthesizers:
1. Create parameter mapping file (`synth_params.txt`)
2. Define default parameters (`synth_param_default.json`)
3. Update `create_synth()` function
4. Add OSC server support
5. Update documentation

## References

- [Serum by Xfer Records](https://xferrecords.com/products/serum)
- [Original Flow Synthesizer Paper](https://arxiv.org/abs/1907.00971)
- [RenderMan Library](https://github.com/fedden/RenderMan)

---

*This integration expands the Flow Synthesizer's capabilities while maintaining its core principle of universal audio synthesizer control through normalizing flows.*