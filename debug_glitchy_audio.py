#!/usr/bin/env python3
"""
Debug Glitchy Audio Issue

This script investigates why the Serum 2 audio is still glitchy despite parameter mapping.
We'll test different approaches:
1. No parameter changes (default plugin state)
2. Minimal parameter changes
3. VST3 preset loading instead of parameter mapping
"""

import os
import json
import numpy as np
from pathlib import Path
from scipy.io import wavfile
import dawdreamer as daw
from code.dd_renderer import DDRenderer


def test_default_serum2_audio():
    """Test Serum 2 with no parameter changes - just default state."""
    print("🔍 Testing Serum 2 default audio...")
    
    plugin_path = "/Library/Audio/Plug-Ins/VST3/Serum2.vst3"
    
    # Check if plugin exists
    if not os.path.exists(plugin_path):
        print(f"  ❌ Plugin not found at: {plugin_path}")
        return None, 0, 0
        
    # Initialize DDRenderer
    renderer = DDRenderer(sample_rate=44100, block_size=512)
    
    # Load plugin
    ok = renderer.load_plugin(plugin_path)
    if not ok:
        print(f"  ❌ Failed to load plugin")
        return None, 0, 0
        
    print(f"  Plugin loaded successfully")
    
    # Render audio with default settings
    audio_stereo = renderer.render_patch(midi_note=60, velocity=100, note_len_sec=2.0, render_len_sec=3.0)
    audio_mono = audio_stereo[0] if len(audio_stereo.shape) > 1 else audio_stereo
        
    # Analyze audio
    max_amp = np.max(np.abs(audio_mono))
    rms = np.sqrt(np.mean(audio_mono**2))
    
    print(f"  Audio shape: {audio_mono.shape}")
    print(f"  Max amplitude: {max_amp:.4f}")
    print(f"  RMS level: {rms:.4f}")
    
    # Check for clipping or unusual patterns
    clipped_samples = np.sum(np.abs(audio_mono) > 0.99)
    zero_samples = np.sum(audio_mono == 0.0)
    
    print(f"  Clipped samples: {clipped_samples}")
    print(f"  Zero samples: {zero_samples}")
    
    # Save audio
    output_path = "debug_default_serum2.wav"
    if max_amp > 0:
        audio_normalized = audio_mono / max_amp * 0.8
    else:
        audio_normalized = audio_mono
        
    wavfile.write(output_path, 44100, audio_normalized)
    print(f"  💾 Saved: {output_path}")
    
    return audio_mono, max_amp, rms


def test_vst3_preset_loading():
    """Test loading VST3 preset files directly instead of parameter mapping."""
    print("\n🔍 Testing VST3 preset loading...")
    
    plugin_path = "/Library/Audio/Plug-Ins/VST3/Serum2.vst3"
    sample_rate = 44100
    block_size = 512
    
    # Initialize engine
    engine = daw.RenderEngine(sample_rate, block_size)
    synth = engine.make_plugin_processor("serum2", plugin_path)
    
    # Load graph
    graph = [(synth, [])]
    engine.load_graph(graph)
    
    # Try to find original .SerumPreset files
    preset_dirs = [
        "/Users/gjb/Library/Audio/Presets/Xfer Records/Serum Presets",
        "/Library/Audio/Presets/Xfer Records/Serum Presets",
        "--preset_dir"  # Our original preset directory
    ]
    
    preset_file = None
    for preset_dir in preset_dirs:
        if os.path.exists(preset_dir):
            preset_files = list(Path(preset_dir).glob("**/*.SerumPreset"))
            if preset_files:
                preset_file = preset_files[0]
                break
                
    if preset_file:
        print(f"  Found preset: {preset_file}")
        try:
            # Try loading VST3 preset
            synth.load_vst3_preset(str(preset_file))
            print(f"  ✅ Loaded VST3 preset successfully")
            
            # Render audio
            synth.clear_midi()
            synth.add_midi_note(60, 100, 0.0, 2.0)
            engine.render(3.0)
            
            audio = engine.get_audio()
            if audio.shape[0] > 1:
                audio_mono = np.mean(audio, axis=0)
            else:
                audio_mono = audio[0]
                
            max_amp = np.max(np.abs(audio_mono))
            rms = np.sqrt(np.mean(audio_mono**2))
            
            print(f"  Max amplitude: {max_amp:.4f}")
            print(f"  RMS level: {rms:.4f}")
            
            # Save audio
            output_path = "debug_vst3_preset.wav"
            if max_amp > 0:
                audio_normalized = audio_mono / max_amp * 0.8
            else:
                audio_normalized = audio_mono
                
            wavfile.write(output_path, sample_rate, audio_normalized)
            print(f"  💾 Saved: {output_path}")
            
            return audio_mono, max_amp, rms
            
        except Exception as e:
            print(f"  ❌ Error loading VST3 preset: {e}")
    else:
        print(f"  ⚠️  No .SerumPreset files found in standard locations")
        
    return None, 0, 0


def test_minimal_parameter_changes():
    """Test with minimal, safe parameter changes."""
    print("\n🔍 Testing minimal parameter changes...")
    
    plugin_path = "/Library/Audio/Plug-Ins/VST3/Serum2.vst3"
    
    # Initialize DDRenderer
    renderer = DDRenderer(sample_rate=44100, block_size=512)
    
    # Load plugin
    ok = renderer.load_plugin(plugin_path)
    if not ok:
        print(f"  ❌ Failed to load plugin")
        return None, 0, 0
    
    # Get parameter info
    params = renderer.get_parameters_description()
    print(f"  Total parameters: {len(params)}")
    
    # Find safe parameters to modify
    safe_params = []
    for param in params:
        name = param['name'].lower()
        # Look for volume, gain, or level parameters
        if any(keyword in name for keyword in ['volume', 'gain', 'level', 'amp']):
            safe_params.append(param)
            
    print(f"  Found {len(safe_params)} safe parameters:")
    for param in safe_params[:5]:  # Show first 5
        print(f"    {param['name']} (index: {param['index']})")
        
    # Apply minimal changes
    if safe_params:
        # Set first safe parameter to 0.5
        param = safe_params[0]
        renderer.inst.set_parameter(param['index'], 0.5)
        print(f"  Set {param['name']} to 0.5")
        
    # Render audio
    audio_stereo = renderer.render_patch(midi_note=60, velocity=100, note_len_sec=2.0, render_len_sec=3.0)
    audio_mono = audio_stereo[0] if len(audio_stereo.shape) > 1 else audio_stereo
        
    max_amp = np.max(np.abs(audio_mono))
    rms = np.sqrt(np.mean(audio_mono**2))
    
    print(f"  Max amplitude: {max_amp:.4f}")
    print(f"  RMS level: {rms:.4f}")
    
    # Save audio
    output_path = "debug_minimal_changes.wav"
    if max_amp > 0:
        audio_normalized = audio_mono / max_amp * 0.8
    else:
        audio_normalized = audio_mono
        
    wavfile.write(output_path, 44100, audio_normalized)
    print(f"  💾 Saved: {output_path}")
    
    return audio_mono, max_amp, rms


def analyze_parameter_ranges():
    """Analyze what parameter ranges Serum 2 actually expects."""
    print("\n🔍 Analyzing Serum 2 parameter ranges...")
    
    plugin_path = "/Library/Audio/Plug-Ins/VST3/Serum2.vst3"
    
    # Initialize DDRenderer
    renderer = DDRenderer(sample_rate=44100, block_size=512)
    
    # Load plugin
    ok = renderer.load_plugin(plugin_path)
    if not ok:
        print(f"  ❌ Failed to load plugin")
        return {}
    
    # Get parameter info
    params = renderer.get_parameters_description()
    
    print(f"  Analyzing {len(params)} parameters...")
    
    # Check parameter ranges
    range_info = {}
    for i, param in enumerate(params[:20]):  # Check first 20 parameters
        name = param['name']
        index = param['index']
        
        # Get current value
        current_value = renderer.inst.get_parameter(index)
        
        # Test setting different values
        test_values = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, -0.5]
        valid_values = []
        
        for test_val in test_values:
            try:
                renderer.inst.set_parameter(index, test_val)
                actual_val = renderer.inst.get_parameter(index)
                valid_values.append((test_val, actual_val))
            except:
                pass
                
        # Restore original value
        renderer.inst.set_parameter(index, current_value)
        
        range_info[name] = {
            'index': index,
            'current': current_value,
            'valid_values': valid_values
        }
        
    # Print analysis
    print("\n  Parameter Range Analysis:")
    for name, info in list(range_info.items())[:10]:  # Show first 10
        print(f"    {name}:")
        print(f"      Current: {info['current']:.4f}")
        print(f"      Valid range: {[f'{v[0]:.2f}->{v[1]:.4f}' for v in info['valid_values'][:3]]}")
        
    return range_info


if __name__ == "__main__":
    print("🎵 Debugging Serum 2 Glitchy Audio")
    print("=" * 50)
    
    # Test 1: Default audio
    default_audio, default_max, default_rms = test_default_serum2_audio()
    
    # Test 2: VST3 preset loading
    vst3_audio, vst3_max, vst3_rms = test_vst3_preset_loading()
    
    # Test 3: Minimal parameter changes
    minimal_audio, minimal_max, minimal_rms = test_minimal_parameter_changes()
    
    # Test 4: Parameter range analysis
    range_info = analyze_parameter_ranges()
    
    # Summary
    print("\n📊 Summary:")
    print(f"  Default audio - Max: {default_max:.4f}, RMS: {default_rms:.4f}")
    if vst3_max > 0:
        print(f"  VST3 preset - Max: {vst3_max:.4f}, RMS: {vst3_rms:.4f}")
    print(f"  Minimal changes - Max: {minimal_max:.4f}, RMS: {minimal_rms:.4f}")
    
    print("\n🔧 Recommendations:")
    if default_max > 0.1:
        print("  ✅ Default Serum 2 produces audio - issue may be with parameter mapping")
    else:
        print("  ❌ Default Serum 2 produces no/low audio - plugin issue")
        
    if vst3_max > default_max * 2:
        print("  ✅ VST3 preset loading works better - use this approach")
    elif vst3_max > 0:
        print("  ⚠️  VST3 preset loading works but similar to default")
        
    print("\n💡 Next steps based on results above...")