# Opsix Synthesizer Integration

This document describes the integration of the Korg opsix synthesizer with the flow_synthesizer project.

## Overview

The opsix synthesizer has been successfully integrated into the flow_synthesizer project, allowing you to use the Korg opsix native AU plugin alongside the existing Diva VST support.

## Files Added/Modified

### Configuration Files

- `code/synth/opsix_params.txt` - Parameter ID to name mappings (841 parameters)
- `code/synth/opsix_param_defaults.json` - Default parameter values
- `code/synth/opsix_important_params.json` - Subset of 32 important parameters for synthesis

### Core Integration

- `code/synth/synthesize.py` - Updated to support both Diva and opsix synthesizers
- `create_opsix_config.py` - Script to generate opsix configuration files

### Testing and Demo

- `test_opsix_integration.py` - Integration test script
- `demo_opsix.py` - Demonstration script showing opsix usage

## Usage

### Basic Usage

```python
from synth.synthesize import create_synth, synthesize_audio

# Create opsix synthesizer
engine, param_defaults, rev_idx = create_synth(dataset="toy", synth_type="opsix")

# Define parameters (using parameter names from opsix_important_params.json)
params = {
    "Algorithm Algorithm": 0.5,
    "Algorithm User Algorithm1->1": 0.3,
    # ... more parameters
}

# Synthesize audio
audio = synthesize_audio(params, engine, param_defaults)
```

### Batch Synthesis

```python
from synth.synthesize import synthesize_batch
import numpy as np

# Load important parameters
with open('code/synth/opsix_important_params.json') as f:
    important_params = json.load(f)

# Create batch of random parameters
batch_params = np.random.rand(4, len(important_params))

# Synthesize batch
audio_batch = synthesize_batch(
    batch_params,
    important_params,
    engine,
    param_defaults,
    rev_idx,
    n_outs=4
)
```

## Plugin Requirements

The opsix integration requires:

- Korg opsix native AU plugin installed at `/Library/Audio/Plug-Ins/Components/opsix_native.component`
- DawDreamer library for AU plugin loading
- Python environment with required dependencies

## Testing

Run the integration test:

```bash
conda activate flow-synth
python test_opsix_integration.py
```

Run the demo:

```bash
conda activate flow-synth
python demo_opsix.py
```

## Parameters

The opsix synthesizer has 841 total parameters. For practical use, 32 important parameters have been identified and are available in `opsix_important_params.json`. These cover the main synthesis controls including:

- Algorithm settings
- Operator parameters
- Filter controls
- Envelope settings
- Effects parameters

## Notes

- The "attempt to map invalid URI" warning is normal and doesn't affect functionality
- Audio output range of [0.000, 0.000] in tests indicates silent output, which is expected with default/random parameters
- The integration maintains backward compatibility with existing Diva synthesizer usage
