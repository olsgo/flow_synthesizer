# Pedalboard Migration Guide

This document explains the migration from DawDreamer to Pedalboard for plugin state capture and audio rendering in the Flow Synthesizer project.

## Overview

The Pedalboard migration provides enhanced plugin state management capabilities by exposing full plugin internal state through binary snapshots (`*.bin` files), while maintaining compatibility with existing parameter-based workflows using JSON files.

## Key Differences from DawDreamer

| Feature | DawDreamer | Pedalboard |
|---------|------------|------------|
| **State Capture** | Only automatable parameters | Full internal state via `raw_state` |
| **State Format** | JSON parameters only | Binary `.bin` files + JSON parameters |
| **Plugin Loading** | `make_plugin_processor()` | `load_plugin()` |
| **Parameter Access** | `get_parameter()` / `set_parameter()` | `plugin.parameters[i].raw_value` |
| **State Persistence** | Limited to automatable params | Complete plugin state |
| **UI State** | Not captured | Included in `raw_state` |

## Architecture

### PBRenderer Class

The `PBRenderer` class in `code/pedalboard_renderer.py` provides a drop-in replacement for `DDRenderer` with these key methods:

```python
# Core API (mirrors DDRenderer)
renderer = PBRenderer(sample_rate=22050, buffer_size=512)
renderer.load_plugin(plugin_path)
renderer.get_patch()  # Returns [(index, normalized_value), ...]
renderer.set_patch(patch_data)
renderer.render_patch(midi_note=60, note_len_sec=3.0, render_len_sec=4.0)

# Enhanced state management
renderer.save_state("preset.bin")  # Save full plugin state
renderer.load_state("preset.bin")  # Restore full plugin state
```

### Binary State Capture

The key innovation is using Pedalboard's `plugin.raw_state` property:

```python
# Save complete plugin state
with open('preset.bin', 'wb') as f:
    f.write(plugin.raw_state)

# Restore complete plugin state  
with open('preset.bin', 'rb') as f:
    plugin.raw_state = f.read()
```

This captures:
- All parameter values (including non-automatable ones)
- Internal plugin state
- UI positions and settings
- Factory vs. user presets
- Plugin-specific data

## Usage Examples

### 1. Capture Initial Plugin States

```bash
# Capture factory states for multiple plugins
python capture_init_state.py --plugins Serum.vst3 Diva.vst3

# Process all plugins in a directory
python capture_init_state.py --plugin_dir /Library/Audio/Plug-Ins/VST3/
```

### 2. Render Dataset with Pedalboard

```bash
# Render using binary states and JSON presets
python code/render_dataset_pb.py \
    "/Library/Audio/Plug-Ins/VST3/Massive X.vst3" \
    "/path/to/presets" \
    "/path/to/output"
```

### 3. Convert JSON Parameters to Plugin State

```bash
# Apply JSON parameters and save as binary state
python json_to_parameters.py \
    --plugin /path/to/plugin.vst3 \
    --json parameters.json \
    --output tweaked_preset.bin
```

## File Formats

### Supported Input Formats

| Format | Extension | Description | Support Level |
|--------|-----------|-------------|---------------|
| **Binary State** | `.bin` | Full plugin state via `raw_state` | ✅ Full |
| **JSON Parameters** | `.json` | Parameter name/value pairs | ✅ Full |
| **VST Presets** | `.vstpreset` | Plugin-specific presets | ⚠️ Limited* |
| **DawDreamer States** | `.state`, `.fxb` | DawDreamer-specific | ❌ Not supported |

*VST preset support depends on individual plugin implementations.

### Output Formats

The renderer produces identical output structure to DawDreamer:

```
output_dir/
├── metadata.csv              # Preset metadata
├── parameter_index_map.json  # Parameter name→index mapping
├── preset1.wav               # Rendered audio
├── preset2.wav
└── ...
```

**metadata.csv columns:**
- `preset`: Source preset file path
- `audio`: Generated WAV file path
- `params_json`: Current parameter values as JSON
- `state_bin`: Binary state file path (if applicable)
- `plugin_path`: Plugin used for rendering

## Integration with Existing Workflows

### ML Training Compatibility

The parameter JSON format remains unchanged for ML compatibility:

```python
# Extract parameters for ML training
patch = renderer.get_patch()  # [(index, value), ...]
param_dict = {str(idx): val for idx, val in patch}
```

### Preset Management

Two workflows are supported:

**A. Fast Record + Replay (Recommended)**
```python
# Record a tweaked preset
renderer.set_patch(your_parameters)
renderer.save_state("tweaked.bin")

# Replay exactly
renderer.load_state("tweaked.bin")
audio = renderer.render_patch()
```

**B. Parameter-Only Mapping**
```python
# Traditional parameter-based workflow
renderer.set_patch(json_parameters)
audio = renderer.render_patch()
```

## Known Limitations and Edge Cases

### Plugin Compatibility

1. **Silent Activation**: Some plugins require UI interaction before producing sound
2. **License Dialogs**: Commercial plugins may show authorization dialogs
3. **Sample Libraries**: Plugins with large sample libraries may take time to load
4. **UI Dependencies**: Some parameters may only be accessible through UI

### Platform Differences

| Platform | VST3 | AU | VST2 |
|----------|------|----|----- |
| **macOS** | ✅ | ✅ | ⚠️ |
| **Linux** | ✅ | ❌ | ⚠️ |
| **Windows** | ✅ | ❌ | ⚠️ |

### Threading and Performance

- Pedalboard is thread-safe for audio processing
- Plugin loading should be done on the main thread
- Some plugins may not support parallel instances

## Troubleshooting

### Common Issues

**Plugin Not Loading**
```bash
# Check plugin path and permissions
python -c "from pedalboard import load_plugin; load_plugin('/path/to/plugin')"
```

**Silent Audio Output**
- Check if plugin requires MIDI input
- Verify sample rate compatibility
- Some instruments need UI activation

**Parameter Mapping Issues**
- Use `--verbose` flag to see parameter names
- Check `parameter_index_map.json` for name→index mapping
- Some parameters may be read-only

### Debug Mode

Enable verbose output for troubleshooting:

```bash
python capture_init_state.py --plugins MyPlugin.vst3 --verbose
python json_to_parameters.py --plugin plugin.vst3 --json params.json --verbose
```

## Extending for New Plugins

### Plugin-Specific Adaptations

For plugins requiring special handling:

```python
class CustomPBRenderer(PBRenderer):
    def load_plugin(self, plugin_path, **kwargs):
        success = super().load_plugin(plugin_path)
        if success and "Serum" in plugin_path:
            # Serum-specific initialization
            self._init_serum_defaults()
        return success
    
    def _init_serum_defaults(self):
        # Plugin-specific setup
        pass
```

### Adding New File Format Support

```python
def load_custom_preset(self, preset_path):
    """Add support for custom preset formats."""
    if preset_path.endswith('.myplugin'):
        # Custom loading logic
        return self._load_myplugin_format(preset_path)
    return False
```

## Migration Checklist

- [ ] Install Pedalboard: `pip install pedalboard`
- [ ] Test basic plugin loading with `PBRenderer`
- [ ] Capture initial states with `capture_init_state.py`
- [ ] Update rendering scripts to use `render_dataset_pb.py`
- [ ] Verify output format matches DawDreamer version
- [ ] Test binary state round-trip functionality
- [ ] Update ML training pipelines if needed

## Performance Considerations

### Memory Usage
- Binary states can be large (1-10MB per preset)
- Consider compression for long-term storage
- Use streaming for large dataset processing

### CPU Usage
- Plugin loading has one-time overhead
- Audio rendering is typically faster than DawDreamer
- Consider process pooling for batch operations

## Future Enhancements

### Planned Features
- Automatic plugin type detection (instrument vs. effect)
- GUI toggle for parameter debugging
- Parallel rendering with multiprocessing
- Compressed state format
- VST preset format support

### Experimental Features
- Real-time parameter tweaking
- MIDI pattern variations
- Multi-output routing for complex plugins