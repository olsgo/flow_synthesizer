# M1 Mac Installation Guide

This guide provides specific instructions for installing FlowSynth on Apple Silicon Macs (M1/M2/M3).

## Prerequisites

1. **macOS Sequoia (15.x)** or later (tested on macOS Sequoia)
2. **Python 3.12+** (tested with Python 3.13)
3. **Homebrew** installed via `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`

## Installation Steps

### 1. Install Python and Dependencies

```bash
# Install Python 3.13 via Homebrew
brew install python@3.13

# Verify installation
python3.13 --version
```

### 2. Install FlowSynth

```bash
# Clone the repository
git clone https://github.com/olsgo/flow_synthesizer.git
cd flow_synthesizer

# Install Python dependencies
pip3.13 install -r requirements.txt
```

### 3. Install RenderMan for Apple Silicon

The RenderMan library needs to be built specifically for Apple Silicon. Follow these steps:

```bash
# Clone RenderMan
git clone https://github.com/fedden/RenderMan.git
cd RenderMan

# Build for Apple Silicon with Python 3.13 and JUCE 8
# Follow the specific build instructions for your system
# Ensure the built library is accessible in /opt/homebrew/lib or /usr/local/lib
```

### 4. Configure Library Paths

The FlowSynth codebase now automatically handles library path configuration for Apple Silicon Macs. The compatibility layer will:

- Automatically detect your architecture (Apple Silicon vs Intel)
- Set up the appropriate library paths (`/opt/homebrew/lib` for Apple Silicon)
- Handle DYLD_LIBRARY_PATH configuration

### 5. Test Installation

```bash
# Test basic functionality
cd flow_synthesizer/code
python3.13 osc_launch.py --help

# Test RenderMan integration (if RenderMan is installed)
cd ..
python3.13 test_renderman_flowsynth.py --verbose
```

## Compatibility Notes

- **PyTorch**: Updated to use PyTorch 2.7+ with proper `weights_only=False` parameters for model loading
- **NumPy**: Compatible with NumPy 2.0+ 
- **macOS Library Paths**: Automatic detection and configuration for both Intel and Apple Silicon
- **Python 3.13**: Full compatibility including all language features and standard library updates

## Troubleshooting

### Library Path Issues

If you encounter library loading issues:

```bash
# Check your architecture
python3.13 -c "import platform; print(f'Architecture: {platform.machine()}')"

# Check library paths
python3.13 code/synth/macos_compat.py
```

### RenderMan Issues

If RenderMan fails to load:

1. Ensure RenderMan is built for your specific architecture
2. Verify the library is in the correct path (`/opt/homebrew/lib` for Apple Silicon)
3. Check that JUCE 8 and Python 3.13 are compatible in your build

### PyTorch Model Loading

If you see warnings about `weights_only`, this is normal and expected - the codebase has been updated to handle the new PyTorch 2.6+ security requirements.

## Performance Notes

Apple Silicon Macs should see improved performance due to:
- Native ARM64 execution
- Optimized NumPy and PyTorch builds for Apple Silicon
- Metal GPU acceleration (where available)