# Multi-Synthesizer Support

This version of the Flow Synthesizer now supports multiple synthesizers, including both Diva VST and Native Instruments Massive X.

## Supported Synthesizers

- **Diva VST** (u-he) - Original synthesizer with 281 parameters
- **Massive X** (Native Instruments) - Wavetable synthesizer with 128 core parameters

## Configuration

### Basic Usage

You can specify which synthesizer to use via command line:

```bash
# Use Diva (default)
python osc_launch.py --synth_type diva

# Use Massive X
python osc_launch.py --synth_type massive_x

# List available synthesizers
python osc_launch.py --list_synths
```

### Plugin Paths

Update the `config.json` file to specify plugin paths for your system:

```json
{
  "synthesizer": {
    "default_type": "diva",
    "plugin_paths": {
      "diva": {
        "mac": "/Library/Audio/Plug-Ins/VST/u-he/Diva.vst",
        "linux": "synth/diva.64.so",
        "windows": "synth/diva.dll"
      },
      "massive_x": {
        "mac": "/Library/Audio/Plug-Ins/VST/Native Instruments/Massive X.vst",
        "linux": "synth/massive_x.64.so",
        "windows": "synth/massive_x.dll"
      }
    }
  }
}
```

Or specify a custom plugin path:

```bash
python osc_launch.py --synth_type massive_x --plugin_path "/path/to/Massive X.vst"
```

## Installing Massive X

1. Install Native Instruments Massive X through Native Access
2. Ensure the plugin is available in your system's VST/AU plugin directory
3. Update the plugin path in `config.json` or use `--plugin_path` argument

## Parameter Mapping

Each synthesizer has its own parameter mapping:
- **Diva**: `code/synth/diva_params.txt` (281 parameters)
- **Massive X**: `code/synth/massive_x_params.txt` (128 parameters)

Default parameter values:
- **Diva**: `code/synth/param_default_32.json`
- **Massive X**: `code/synth/massive_x_default.json`

## Architecture Changes

### New Files
- `code/synth/synthesizer_interface.py` - Abstraction layer for multiple synthesizers
- `code/synth/massive_x_params.txt` - Massive X parameter mapping
- `code/synth/massive_x_default.json` - Default Massive X parameters
- `code/config_manager.py` - Configuration management
- `code/config.json` - Configuration file
- `code/test_synthesizer.py` - Test script for synthesizer interfaces

### Modified Files
- `code/osc_launch.py` - Added synthesizer selection arguments
- `code/osc_server.py` - Updated to handle synthesizer type
- `code/synth/synthesize.py` - Updated to use new abstraction layer

## Compatibility

The changes are designed to be backward compatible:
- Default behavior remains unchanged (uses Diva)
- Existing Diva-based projects will continue to work
- All original command-line arguments are preserved

## Adding New Synthesizers

To add support for a new synthesizer:

1. Create parameter mapping file (e.g., `synth/new_synth_params.txt`)
2. Create default parameters file (e.g., `synth/new_synth_default.json`)
3. Add new synthesizer class to `synthesizer_interface.py`
4. Register in `SynthesizerFactory._interfaces`
5. Update `config.json` with plugin paths

## Limitations

- Massive X parameter mapping is currently basic (128 core parameters)
- Some advanced Massive X features may not be accessible
- Models trained on Diva parameters may need retraining for Massive X
- librenderman library support required for audio synthesis