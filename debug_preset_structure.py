#!/usr/bin/env python3
"""
Debug script to understand the preset data structure.
"""

import json
import os

def debug_preset_structure():
    preset_path = "converted_presets/Piano/PN - Piano Classic Layer.json"
    
    if not os.path.exists(preset_path):
        print(f"Preset file not found: {preset_path}")
        return
    
    with open(preset_path, 'r') as f:
        preset_json = json.load(f)
    
    print("Preset JSON structure:")
    print(f"Top-level keys: {list(preset_json.keys())}")
    
    if 'data' in preset_json:
        data = preset_json['data']
        print(f"\nData section type: {type(data)}")
        print(f"Data section keys (first 10): {list(data.keys())[:10]}")
        
        # Look for parameters that start with '.'
        dot_params = [k for k in data.keys() if k.startswith('.')]
        print(f"\nParameters starting with '.': {len(dot_params)}")
        print(f"First 10 dot parameters: {dot_params[:10]}")
        
        # Look for oscillator parameters specifically
        osc_params = [k for k in data.keys() if 'Oscillator' in k and k.startswith('.')]
        print(f"\nOscillator parameters: {len(osc_params)}")
        print(f"Oscillator parameters: {osc_params[:10]}")
        
        # Show some actual parameter values
        print("\nSample parameter values:")
        for param in dot_params[:5]:
            print(f"  {param}: {data[param]}")
    else:
        print("\nNo 'data' section found in preset")
    
    # Check dawdreamer_params section
    if 'dawdreamer_params' in preset_json:
        dawdreamer_params = preset_json['dawdreamer_params']
        print(f"\nDawDreamer params section type: {type(dawdreamer_params)}")
        
        if isinstance(dawdreamer_params, dict):
            print(f"DawDreamer params keys (first 10): {list(dawdreamer_params.keys())[:10]}")
            
            # Look for parameters that start with '.'
            dot_params = [k for k in dawdreamer_params.keys() if k.startswith('.')]
            print(f"\nDawDreamer parameters starting with '.': {len(dot_params)}")
            print(f"First 10 dot parameters: {dot_params[:10]}")
            
            # Show some actual parameter values
            print("\nSample DawDreamer parameter values:")
            for param in dot_params[:5]:
                print(f"  {param}: {dawdreamer_params[param]}")
        else:
            print(f"DawDreamer params content: {dawdreamer_params}")
    
    # Check if parameters are at the top level
    dot_params = [k for k in preset_json.keys() if k.startswith('.')]
    print(f"\nTop-level parameters starting with '.': {len(dot_params)}")
    print(f"First 10: {dot_params[:10]}")

if __name__ == "__main__":
    debug_preset_structure()