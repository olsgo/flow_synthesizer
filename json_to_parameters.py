#!/usr/bin/env python3
"""
Convert JSON parameter files to plugin parameters using Pedalboard.

Usage:
    python json_to_parameters.py --plugin /path/to/plugin.vst3 --json /path/to/params.json
    python json_to_parameters.py --plugin /path/to/plugin.vst3 --json /path/to/params.json --output state.bin
"""

import argparse
import json
import sys
import os
from pathlib import Path

# Add the code directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))

from pedalboard_renderer import PBRenderer

def load_json_parameters(json_path):
    """
    Load parameter data from a JSON file.
    
    Args:
        json_path: Path to JSON file
    
    Returns:
        dict: Parameter data or None if failed
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"❌ Error loading JSON file {json_path}: {e}")
        return None

def apply_json_to_plugin(renderer, json_data, parameter_map=None):
    """
    Apply JSON parameter data to a plugin.
    
    Args:
        renderer: PBRenderer instance with loaded plugin
        json_data: Dictionary of parameter data
        parameter_map: Optional mapping of parameter names to indices
    
    Returns:
        tuple: (success_count, total_count, warnings)
    """
    if not isinstance(json_data, dict):
        print("❌ JSON data must be a dictionary")
        return 0, 0, ["JSON data is not a dictionary"]
    
    # Get parameter descriptions
    param_descriptions = renderer.get_parameters_description()
    
    # Create name-to-index mapping if not provided
    if parameter_map is None:
        parameter_map = {desc["name"]: desc["index"] for desc in param_descriptions}
    
    # Also create an index-to-name mapping for fallback
    index_map = {str(i): i for i in range(len(param_descriptions))}
    
    success_count = 0
    total_count = 0
    warnings = []
    
    patch = []
    
    for param_key, value in json_data.items():
        total_count += 1
        
        # Try to find parameter by name first
        param_index = None
        if param_key in parameter_map:
            param_index = parameter_map[param_key]
        elif param_key in index_map:
            # Fallback to index-based lookup
            param_index = index_map[param_key]
        elif param_key.isdigit():
            # Direct index lookup
            idx = int(param_key)
            if 0 <= idx < len(param_descriptions):
                param_index = idx
        
        if param_index is not None:
            try:
                # Ensure value is in 0-1 range (normalized)
                normalized_value = float(value)
                if not (0.0 <= normalized_value <= 1.0):
                    warnings.append(f"Parameter '{param_key}' value {value} is outside 0-1 range, clamping")
                    normalized_value = max(0.0, min(1.0, normalized_value))
                
                patch.append((param_index, normalized_value))
                success_count += 1
                
            except (ValueError, TypeError) as e:
                warnings.append(f"Invalid value for parameter '{param_key}': {value} ({e})")
        else:
            warnings.append(f"Parameter '{param_key}' not found in plugin")
    
    # Apply the patch
    if patch:
        try:
            renderer.set_patch(patch)
            print(f"✓ Applied {success_count} parameters to plugin")
        except Exception as e:
            print(f"❌ Error applying parameters: {e}")
            return 0, total_count, warnings + [f"Failed to apply patch: {e}"]
    
    return success_count, total_count, warnings

def main():
    parser = argparse.ArgumentParser(description="Convert JSON parameter files to plugin parameters")
    parser.add_argument("--plugin", required=True, help="Path to plugin file")
    parser.add_argument("--json", required=True, help="Path to JSON parameter file")
    parser.add_argument("--output", help="Optional output .bin file to save final state")
    parser.add_argument("--parameter_map", help="Optional JSON file with parameter name->index mapping")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.plugin):
        print(f"❌ Plugin file not found: {args.plugin}")
        return 1
    
    if not os.path.exists(args.json):
        print(f"❌ JSON file not found: {args.json}")
        return 1
    
    # Load JSON data
    json_data = load_json_parameters(args.json)
    if json_data is None:
        return 1
    
    print(f"📄 Loaded {len(json_data)} parameters from {args.json}")
    
    # Load optional parameter mapping
    parameter_map = None
    if args.parameter_map:
        if os.path.exists(args.parameter_map):
            try:
                with open(args.parameter_map, 'r') as f:
                    parameter_map = json.load(f)
                print(f"📋 Loaded parameter mapping from {args.parameter_map}")
            except Exception as e:
                print(f"⚠️  Failed to load parameter mapping: {e}")
        else:
            print(f"⚠️  Parameter mapping file not found: {args.parameter_map}")
    
    # Initialize renderer and load plugin
    try:
        renderer = PBRenderer(sample_rate=22050, buffer_size=512)
        
        if not renderer.load_plugin(args.plugin):
            print(f"❌ Failed to load plugin: {args.plugin}")
            return 1
        
        plugin_name = Path(args.plugin).stem
        print(f"✓ Loaded plugin: {plugin_name}")
        
        # Show plugin info
        param_descriptions = renderer.get_parameters_description()
        print(f"  Plugin has {len(param_descriptions)} parameters")
        
        if args.verbose:
            print("  Parameters:")
            for i, desc in enumerate(param_descriptions[:10]):  # Show first 10
                print(f"    {i}: {desc.get('name', 'Unknown')}")
            if len(param_descriptions) > 10:
                print(f"    ... and {len(param_descriptions) - 10} more")
        
        # Apply JSON parameters
        success_count, total_count, warnings = apply_json_to_plugin(
            renderer, json_data, parameter_map
        )
        
        # Show warnings
        if warnings:
            print("\n⚠️  Warnings:")
            for warning in warnings:
                print(f"    {warning}")
        
        # Show results
        print(f"\n📊 Results: {success_count}/{total_count} parameters applied successfully")
        
        # Save state if requested
        if args.output:
            if renderer.save_state(args.output):
                print(f"✓ Saved plugin state to: {args.output}")
            else:
                print(f"❌ Failed to save state to: {args.output}")
                return 1
        
        # Show current parameter values for verification
        if args.verbose:
            current_patch = renderer.get_patch()
            print(f"\n🔍 Current parameter values (first 5):")
            for i, (idx, value) in enumerate(current_patch[:5]):
                param_name = param_descriptions[idx].get('name', f'Param_{idx}') if idx < len(param_descriptions) else f'Param_{idx}'
                print(f"    {idx}: {param_name} = {value:.3f}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())