# Audio Plugin Setup Guide

This guide explains how to set up audio plugins for the Flow Synthesizer project and troubleshoot common issues.

## Quick Start

### 1. Test Current Setup

Run the diagnostic tool to check your current plugin setup:

```bash
python plugin_debug_tool.py
```

This will:
- List all configured plugins
- Show which plugins are available on your system
- Provide specific guidance for missing plugins

### 2. Configure Plugin Paths

Create or edit `plugin_config.yml` in the project root:

```yaml
plugins:
  diva:
    vst3: "/Library/Audio/Plug-Ins/VST3/Diva.vst3"
    au: "/Library/Audio/Plug-Ins/Components/Diva.component"
    vst: "/Library/Audio/Plug-Ins/VST/u-he/Diva.vst"
  
  massive_x:
    vst3: "/Library/Audio/Plug-Ins/VST3/Massive X.vst3"
    au: "/Library/Audio/Plug-Ins/Components/Massive X.component"

defaults:
  preferred_format: "vst3"
  sample_rate: 22050
  block_size: 512
```

### 3. Test Plugin Loading

Test a specific plugin:

```bash
python plugin_debug_tool.py --test-plugin diva
```

## Supported Plugins

The Flow Synthesizer works best with these plugins:

### Primary Support
- **Diva** (u-he) - Main plugin used in research paper
- **Massive X** (Native Instruments) - Modern VST3 support
- **FM8** (Native Instruments) - FM synthesis

### Experimental Support
- **Polymax** (Universal Audio)
- **Serum** (Xfer Records)
- **Sylenth1** (LennarDigital)

## Platform-Specific Setup

### macOS

**VST3 Plugins:**
```
/Library/Audio/Plug-Ins/VST3/
~/Library/Audio/Plug-Ins/VST3/
```

**Audio Units:**
```
/Library/Audio/Plug-Ins/Components/
~/Library/Audio/Plug-Ins/Components/
```

**VST2 Plugins:**
```
/Library/Audio/Plug-Ins/VST/
~/Library/Audio/Plug-Ins/VST/
```

### Windows

**VST3 Plugins:**
```
C:\Program Files\Common Files\VST3\
C:\Program Files (x86)\Common Files\VST3\
```

**VST2 Plugins:**
```
C:\Program Files\VSTPlugins\
C:\Program Files (x86)\VSTPlugins\
```

### Linux

**VST3 Plugins:**
```
~/.vst3/
/usr/lib/vst3/
```

**VST2 Plugins:**
```
~/.vst/
/usr/lib/vst/
```

## Plugin Installation

### Diva (u-he)

1. Download from [u-he.com](https://u-he.com/products/diva/)
2. Install to standard plugin directory
3. Test with: `python plugin_debug_tool.py --test-plugin diva`

**Default Locations:**
- macOS VST: `/Library/Audio/Plug-Ins/VST/u-he/Diva.vst`
- macOS VST3: `/Library/Audio/Plug-Ins/VST3/Diva.vst3`
- macOS AU: `/Library/Audio/Plug-Ins/Components/Diva.component`

### Massive X (Native Instruments)

1. Install via Native Access
2. Verify installation in standard directories
3. Test with: `python plugin_debug_tool.py --test-plugin massive_x`

## Troubleshooting

### Plugin Not Loading

**Error:** "Unable to load plugin"

**Solutions:**
1. Check if plugin file exists:
   ```bash
   python plugin_debug_tool.py --test-path "/path/to/plugin"
   ```

2. Verify plugin format is supported:
   - VST3 (.vst3) - Recommended
   - Audio Units (.component) - macOS only
   - VST2 (.vst, .dll) - Legacy

3. Check plugin permissions:
   ```bash
   ls -la "/Library/Audio/Plug-Ins/VST3/"
   ```

### Configuration Issues

**Error:** "No configured path found for plugin"

**Solutions:**
1. Check configuration file exists:
   ```bash
   ls -la plugin_config.yml
   ```

2. Validate YAML syntax:
   ```bash
   python -c "import yaml; print(yaml.safe_load(open('plugin_config.yml')))"
   ```

3. Add missing plugin to configuration:
   ```yaml
   custom_plugins:
     my_synth:
       vst3: "/path/to/MySynth.vst3"
   ```

### Search for Plugins

**Find plugins automatically:**
```bash
python plugin_debug_tool.py --search "Diva"
```

**List all configured plugins:**
```bash
python plugin_debug_tool.py --list-configured
```

## Development and Testing

### Running Tests

**Comprehensive test suite:**
```bash
python test_plugin_framework.py
```

**Test specific functionality:**
```bash
# Test basic DDRenderer
python -c "from code.dd_renderer import DDRenderer; print('OK')"

# Test configuration loading
python -c "from code.plugin_config import get_config; print(get_config().list_configured_plugins())"
```

### CI/CD Environments

The testing framework is designed to work in environments without plugins:

```python
# Tests gracefully handle missing plugins
success = renderer.load_plugin_by_name("diva")
if not success:
    print("Plugin not available - using simulation mode")
```

### Adding New Plugins

1. **Install the plugin** using the vendor's installer
2. **Find the plugin path:**
   ```bash
   python plugin_debug_tool.py --search "PluginName"
   ```
3. **Add to configuration:**
   ```yaml
   custom_plugins:
     plugin_name:
       vst3: "/path/to/Plugin.vst3"
   ```
4. **Test loading:**
   ```bash
   python plugin_debug_tool.py --test-plugin plugin_name
   ```

## Performance Tips

### Plugin Format Preference

1. **VST3** - Recommended for best compatibility and features
2. **Audio Units** - macOS native, good performance
3. **VST2** - Legacy, use only if VST3 not available

### Audio Settings

Optimize for your use case:

```yaml
defaults:
  sample_rate: 22050    # Lower for faster processing
  block_size: 512       # Balance between latency and stability
  preferred_format: "vst3"
```

## Getting Help

### Diagnostic Information

Generate a diagnostic report:

```bash
python plugin_debug_tool.py --verbose > diagnostic_report.txt
```

### Common Issues

1. **Plugin licensing** - Ensure plugins are properly licensed
2. **Architecture mismatch** - Ensure 64-bit plugins on 64-bit systems
3. **Dependencies** - Some plugins require additional libraries
4. **Permissions** - Check file/directory permissions

### Support Resources

- **Project Issues:** [GitHub Issues](https://github.com/olsgo/flow_synthesizer/issues)
- **Plugin Vendors:** Check vendor documentation for installation issues
- **Community:** Audio production forums for plugin-specific questions