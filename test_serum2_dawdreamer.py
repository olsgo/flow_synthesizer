#!/usr/bin/env python3
"""
Test script for Serum 2 preset integration with DawDreamer.
This script demonstrates loading converted Serum 2 presets and using them with DawDreamer.
"""

import os
import json
import random
import dawdreamer as daw
import numpy as np
from pathlib import Path

def find_serum2_plugin():
    """Find the Serum 2 plugin file."""
    # Common locations for Serum 2 plugin on macOS
    plugin_paths = [
        "/Library/Audio/Plug-Ins/VST3/Serum2.vst3",
        "/Library/Audio/Plug-Ins/Components/Serum2.component",
        "/Library/Audio/Plug-Ins/VST/Serum2.vst",
        "~/Library/Audio/Plug-Ins/VST3/Serum2.vst3",
        "~/Library/Audio/Plug-Ins/Components/Serum2.component",
        "~/Library/Audio/Plug-Ins/VST/Serum2.vst"
    ]
    
    for path in plugin_paths:
        expanded_path = os.path.expanduser(path)
        if os.path.exists(expanded_path):
            return expanded_path
    
    return None

def load_converted_presets():
    """Load all converted JSON presets."""
    preset_dir = Path('./converted_presets')
    if not preset_dir.exists():
        print(f"Preset directory not found: {preset_dir}")
        return []
    
    presets = []
    for json_file in preset_dir.rglob('*.json'):
        try:
            with open(json_file, 'r') as f:
                preset_data = json.load(f)
                preset_data['_file_path'] = str(json_file)
                presets.append(preset_data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return presets

def test_serum2_with_dawdreamer():
    """Test Serum 2 integration with DawDreamer."""
    print("Testing Serum 2 integration with DawDreamer...")
    
    # Find Serum 2 plugin
    plugin_path = find_serum2_plugin()
    if not plugin_path:
        print("Serum 2 plugin not found. Please install Serum 2.")
        return False
    
    print(f"Found Serum 2 plugin: {plugin_path}")
    
    # Load converted presets
    presets = load_converted_presets()
    
    if not presets:
        print("No converted presets found.")
        return False
    
    print(f"Loaded {len(presets)} converted presets")
    
    # Initialize DawDreamer engine
    SAMPLE_RATE = 44100
    BUFFER_SIZE = 512
    
    try:
        engine = daw.RenderEngine(SAMPLE_RATE, BUFFER_SIZE)
        
        # Load Serum 2 plugin
        synth = engine.make_plugin_processor("serum2", plugin_path)
        
        if synth is None:
            print("Failed to load Serum 2 plugin")
            return False
        
        print("Successfully loaded Serum 2 plugin")
        
        # Test with a few random presets
        test_presets = random.sample(presets, min(5, len(presets)))
        
        for i, preset_data in enumerate(test_presets):
            preset_name = preset_data.get('presetName', Path(preset_data['_file_path']).stem)
            print(f"\nTesting preset {i+1}: {preset_name}")
            
            # Use parameter mapping approach since .SerumPreset files are not directly supported by DawDreamer
            # DawDreamer supports .vstpreset, .fxp, .state, and .bin files, but not Serum's proprietary .SerumPreset format
            
            # Look for DawDreamer parameters in multiple possible locations
            dawdreamer_params = None
            if 'dawdreamer_params' in preset_data:
                dawdreamer_params = preset_data['dawdreamer_params']
            elif 'data' in preset_data and 'parameters' in preset_data['data']:
                dawdreamer_params = preset_data['data']['parameters']
            elif 'parameters' in preset_data:
                dawdreamer_params = preset_data['parameters']
            
            if not dawdreamer_params:
                print("  No DawDreamer parameters found in preset")
                continue
                
            print(f"  Found {len(dawdreamer_params)} parameters to set")
            
            # Get plugin parameter descriptions to build name-to-index mapping
            try:
                param_descriptions = synth.get_parameters_description()
                print(f"  Plugin has {len(param_descriptions)} parameters available")
                
                # Build name-to-index mapping
                name_to_index = {}
                for i, desc in enumerate(param_descriptions):
                    param_name = desc.get('name', str(i))
                    name_to_index[param_name] = i
                    # Also try variations of the name
                    name_to_index[param_name.lstrip('.')] = i
                    if param_name.startswith('.'):
                        name_to_index[param_name[1:]] = i
                
                print(f"  Built parameter mapping for {len(name_to_index)} parameter names")
                
                # Debug: Print first 20 plugin parameter names to understand naming convention
                print("  First 20 plugin parameter names:")
                for i, desc in enumerate(param_descriptions[:20]):
                    param_name = desc.get('name', str(i))
                    print(f"    {i}: '{param_name}'")
                
                # Debug: Print first 10 preset parameter names
                print("  First 10 preset parameter names:")
                for i, (param_name, param_value) in enumerate(list(dawdreamer_params.items())[:10]):
                    print(f"    '{param_name}': {param_value}")
                
                # Set a subset of parameters to avoid overwhelming the plugin
                param_items = list(dawdreamer_params.items())[:20]  # Test with first 20 parameters
                params_set = 0
                
                for param_name, param_value in param_items:
                    try:
                        # Normalize parameter value to 0-1 range if needed
                        normalized_value = max(0.0, min(1.0, float(param_value) / 100.0 if param_value > 1.0 else float(param_value)))
                        
                        # Try to find parameter index
                        param_index = None
                        clean_param_name = param_name.lstrip('.')
                        
                        # Try different name variations
                        for name_variant in [param_name, clean_param_name, f".{clean_param_name}"]:
                            if name_variant in name_to_index:
                                param_index = name_to_index[name_variant]
                                break
                        
                        if param_index is not None:
                            synth.set_parameter(param_index, normalized_value)
                            print(f"    Set parameter {param_index} ({param_name}) = {normalized_value:.3f}")
                            params_set += 1
                        else:
                            print(f"    Could not find parameter index for: {param_name}")
                            
                    except Exception as e:
                        print(f"    Failed to set {param_name}: {e}")
                
                print(f"  Attempted to set {len(param_items)} parameters, {params_set} successful")
                
            except Exception as e:
                print(f"  Could not get parameter descriptions: {e}")
                continue
            
            # Generate a short audio test
            try:
                # Create a simple MIDI sequence
                midi_notes = [
                    (0.0, 60, 100, 1.0),  # C4 for 1 second
                    (1.5, 64, 100, 1.0),  # E4 for 1 second
                    (3.0, 67, 100, 1.0),  # G4 for 1 second
                ]
                
                synth.clear_midi()
                for start_time, note, velocity, duration in midi_notes:
                    synth.add_midi_note(note, velocity, start_time, duration)
                
                # Render audio
                engine.load_graph([(synth, [])])
                audio = engine.render(5.0)  # 5 seconds
                
                # Check if audio was generated
                if audio is not False and hasattr(audio, 'size') and audio.size > 0:
                    max_amplitude = np.max(np.abs(audio))
                    if max_amplitude > 0.001:
                        print(f"  ✓ Successfully generated audio (max amplitude: {max_amplitude:.3f})")
                    else:
                        print(f"  ⚠ Audio generated but very quiet (max amplitude: {max_amplitude:.6f})")
                else:
                    print(f"  ✗ No audio generated or invalid audio data")
                    
            except Exception as e:
                print(f"  ✗ Error generating audio: {e}")
        
        print("\n=== Test Summary ===")
        print(f"Plugin loaded: ✓")
        print(f"Presets loaded: {len(presets)}")
        print(f"Test presets: {len(test_presets)}")
        
        return True
        
    except Exception as e:
        print(f"Error initializing DawDreamer: {e}")
        return False

def main():
    """Main function."""
    success = test_serum2_with_dawdreamer()
    
    if success:
        print("\n🎉 Serum 2 integration test completed successfully!")
    else:
        print("\n❌ Serum 2 integration test failed.")
        
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())