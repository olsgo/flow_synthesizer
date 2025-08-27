#!/usr/bin/env python3
"""
List all available parameters in Serum 2 VST plugin.
This will help us understand the correct parameter names for mapping.
"""

import dawdreamer as daw
import json
from pathlib import Path

def list_serum2_parameters():
    """List all parameters available in Serum 2."""
    
    plugin_path = "/Library/Audio/Plug-Ins/VST3/Serum2.vst3"
    
    print("Serum 2 VST Parameters")
    print("=" * 50)
    
    # Initialize engine and plugin
    engine = daw.RenderEngine(44100, 512)
    synth = engine.make_plugin_processor("serum2", plugin_path)
    graph = [(synth, [])]
    engine.load_graph(graph)
    
    # Get all parameters
    params = synth.get_parameters_description()
    
    print(f"Total parameters: {len(params)}\n")
    
    # Group parameters by category (if possible)
    categories = {}
    for param in params:
        name = param["name"]
        # Try to categorize by common prefixes
        if any(prefix in name.lower() for prefix in ['osc', 'oscillator']):
            category = "Oscillator"
        elif any(prefix in name.lower() for prefix in ['filter', 'filt']):
            category = "Filter"
        elif any(prefix in name.lower() for prefix in ['env', 'envelope']):
            category = "Envelope"
        elif any(prefix in name.lower() for prefix in ['lfo']):
            category = "LFO"
        elif any(prefix in name.lower() for prefix in ['fx', 'effect', 'reverb', 'delay', 'chorus']):
            category = "Effects"
        elif any(prefix in name.lower() for prefix in ['master', 'main', 'volume', 'level']):
            category = "Master"
        else:
            category = "Other"
        
        if category not in categories:
            categories[category] = []
        categories[category].append(param)
    
    # Print parameters by category
    for category, params_list in sorted(categories.items()):
        print(f"\n{category} ({len(params_list)} parameters):")
        print("-" * (len(category) + 20))
        
        for param in sorted(params_list, key=lambda x: x["name"]):
            index = param["index"]
            name = param["name"]
            print(f"  [{index:3d}] {name}")
    
    # Save to JSON for reference
    output_file = Path("/Users/gjb/Datasets/serum2/serum2_parameters.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(params, f, indent=2)
    
    print(f"\n💾 Full parameter list saved to: {output_file}")
    
    # Also check a sample preset to see what parameter names it uses
    print("\n" + "=" * 50)
    print("Sample Preset Parameter Names")
    print("=" * 50)
    
    preset_path = "converted_presets/Piano/PN - Piano Classic Layer.json"
    try:
        with open(preset_path, 'r') as f:
            preset_data = json.load(f)
        
        print(f"\nPreset: {preset_path}")
        print(f"Parameters in preset: {len(preset_data)}\n")
        
        for param_name, param_value in sorted(preset_data.items()):
            print(f"  {param_name}: {param_value}")
            
    except FileNotFoundError:
        print(f"\nPreset file not found: {preset_path}")
    except Exception as e:
        print(f"\nError reading preset: {e}")

if __name__ == "__main__":
    list_serum2_parameters()