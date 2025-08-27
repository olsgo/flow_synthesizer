#!/usr/bin/env python3
"""
Diagnostic script to identify which parameter mappings cause glitchy audio.
This will help isolate the problematic parameters in the preset loading process.
"""

import os
import json
import numpy as np
from pathlib import Path
import dawdreamer as daw
from scipy.io import wavfile
from serum2_parameter_mapper import Serum2ParameterMapper

def test_parameter_application():
    """Test parameter application step by step to identify issues."""
    
    # Plugin path
    plugin_path = "/Library/Audio/Plug-Ins/VST3/Serum2.vst3"
    
    # Create output directory
    output_dir = Path("/Users/gjb/Datasets/serum2/diagnostic")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Diagnostic: Parameter Mapping Issues")
    print("=" * 50)
    
    # Test 1: Baseline - no parameters changed
    print("\n1. Testing baseline (no parameter changes)...")
    engine = daw.RenderEngine(44100, 512)
    synth = engine.make_plugin_processor("serum2", plugin_path)
    graph = [(synth, [])]
    engine.load_graph(graph)
    
    # Generate baseline audio
    synth.clear_midi()
    synth.add_midi_note(60, 80, 0.0, 2.0)
    engine.render(4.0)
    baseline_audio = engine.get_audio()
    synth.clear_midi()
    
    baseline_max = np.max(np.abs(baseline_audio))
    print(f"   Baseline audio: max={baseline_max:.4f}, shape={baseline_audio.shape}")
    
    # Save baseline
    wavfile.write(str(output_dir / "baseline.wav"), 44100, baseline_audio.transpose())
    
    # Test 2: Load a simple preset and apply parameters gradually
    print("\n2. Testing gradual parameter application...")
    
    # Load a simple preset
    preset_path = "converted_presets/Piano/PN - Piano Classic Layer.json"
    if not os.path.exists(preset_path):
        print(f"   ❌ Preset not found: {preset_path}")
        return
    
    with open(preset_path, 'r') as f:
        preset_data = json.load(f)
    
    print(f"   Loaded preset with {len(preset_data)} parameters")
    
    # Get plugin parameters
    plugin_params = synth.get_parameters_description()
    param_name_to_index = {param["name"]: param["index"] for param in plugin_params}
    
    # Test applying parameters in small batches
    batch_size = 10
    param_items = list(preset_data.items())
    
    for batch_start in range(0, len(param_items), batch_size):
        batch_end = min(batch_start + batch_size, len(param_items))
        batch = param_items[batch_start:batch_end]
        
        print(f"\n   Batch {batch_start//batch_size + 1}: parameters {batch_start+1}-{batch_end}")
        
        # Apply this batch of parameters
        applied_count = 0
        for param_name, param_value in batch:
            # Simple mapping - just try to find exact match
            if param_name in param_name_to_index:
                try:
                    # Validate parameter value
                    if isinstance(param_value, (int, float)) and 0.0 <= param_value <= 1.0:
                        synth.set_parameter(param_name_to_index[param_name], param_value)
                        applied_count += 1
                    else:
                        print(f"     ⚠️  Invalid value for {param_name}: {param_value}")
                except Exception as e:
                    print(f"     ❌ Error setting {param_name}: {e}")
        
        print(f"     Applied {applied_count}/{len(batch)} parameters")
        
        # Test audio after this batch
        synth.clear_midi()
        synth.add_midi_note(60, 80, 0.0, 2.0)
        engine.render(4.0)
        test_audio = engine.get_audio()
        synth.clear_midi()
        
        if test_audio.size == 0:
            print(f"     ❌ No audio generated after batch {batch_start//batch_size + 1}")
            break
            
        test_max = np.max(np.abs(test_audio))
        print(f"     Audio: max={test_max:.4f}")
        
        # Check for glitchy audio (very high amplitude or NaN)
        if test_max > 2.0 or np.isnan(test_max) or np.isinf(test_max):
            print(f"     ❌ GLITCHY AUDIO DETECTED after batch {batch_start//batch_size + 1}!")
            print(f"     Problematic parameters in this batch:")
            for param_name, param_value in batch:
                print(f"       {param_name}: {param_value}")
            
            # Save the glitchy audio for analysis
            glitch_filename = f"glitchy_batch_{batch_start//batch_size + 1}.wav"
            try:
                # Clip extreme values before saving
                clipped_audio = np.clip(test_audio, -1.0, 1.0)
                wavfile.write(str(output_dir / glitch_filename), 44100, clipped_audio.transpose())
                print(f"     Saved clipped glitchy audio: {glitch_filename}")
            except Exception as e:
                print(f"     Could not save glitchy audio: {e}")
            break
        else:
            # Save good audio
            batch_filename = f"batch_{batch_start//batch_size + 1}.wav"
            wavfile.write(str(output_dir / batch_filename), 44100, test_audio.transpose())
    
    # Test 3: Check specific parameter ranges
    print("\n3. Testing parameter value ranges...")
    
    # Reset to baseline
    engine = daw.RenderEngine(44100, 512)
    synth = engine.make_plugin_processor("serum2", plugin_path)
    graph = [(synth, [])]
    engine.load_graph(graph)
    
    # Test extreme parameter values
    test_params = [
        ("Master", [0.0, 0.5, 1.0]),
        ("A Level", [0.0, 0.5, 1.0]),
        ("Filter Cutoff", [0.0, 0.5, 1.0]),
    ]
    
    for param_name, values in test_params:
        if param_name in param_name_to_index:
            print(f"\n   Testing {param_name}:")
            for value in values:
                try:
                    synth.set_parameter(param_name_to_index[param_name], value)
                    
                    synth.clear_midi()
                    synth.add_midi_note(60, 80, 0.0, 1.0)
                    engine.render(2.0)
                    test_audio = engine.get_audio()
                    synth.clear_midi()
                    
                    test_max = np.max(np.abs(test_audio))
                    print(f"     {param_name}={value}: max={test_max:.4f}")
                    
                    if test_max > 2.0 or np.isnan(test_max):
                        print(f"     ❌ PROBLEMATIC VALUE: {param_name}={value}")
                        
                except Exception as e:
                    print(f"     ❌ Error with {param_name}={value}: {e}")
    
    print(f"\n🔍 Diagnostic complete! Check files in: {output_dir}")

if __name__ == "__main__":
    test_parameter_application()