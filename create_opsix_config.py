#!/usr/bin/env python3
"""
Script to create opsix synthesizer configuration files for flow_synthesizer integration.
This will generate the parameter mappings and default values needed for the project.
"""

import json
from code.dd_renderer import DDRenderer

def create_opsix_config():
    """Create opsix configuration files."""
    print("Loading opsix synthesizer...")
    
    # Initialize renderer with opsix
    renderer = DDRenderer(sample_rate=22050, block_size=512)
    success = renderer.load_plugin("/Library/Audio/Plug-Ins/Components/opsix_native.component")
    
    if not success:
        print("Failed to load opsix plugin")
        return
    
    print(f"Plugin loaded successfully")
    
    # Get parameter descriptions
    params = renderer.get_parameters_description()
    total_params = len(params)
    print(f"Total parameters: {total_params}")
    
    # Create parameter index mapping (similar to diva_params.txt)
    param_mapping = {}
    param_defaults = {}
    
    for i, param in enumerate(params):
        param_name = param.get('name', f'param_{i}')
        param_mapping[i] = param_name
        
        # Get current parameter value as default
        current_value = renderer.inst.get_parameter(i)
        param_defaults[param_name] = float(current_value)
        
        if i < 20:  # Print first 20 for verification
            print(f"{i}: {param_name} = {current_value}")
    
    # Save parameter mapping (similar to diva_params.txt)
    with open('code/synth/opsix_params.txt', 'w') as f:
        f.write(str(param_mapping).replace(', ', ',\n  '))
    
    # Save parameter defaults (similar to param_nomod.json)
    with open('code/synth/opsix_param_defaults.json', 'w') as f:
        json.dump(param_defaults, f, indent=2)
    
    print(f"\nConfiguration files created:")
    print(f"- code/synth/opsix_params.txt ({total_params} parameters)")
    print(f"- code/synth/opsix_param_defaults.json")
    
    # Create a subset of important parameters for training (similar to final_params)
    # For now, we'll select the first 32 parameters as a starting point
    important_params = [param_mapping[i] for i in range(min(32, total_params))]
    
    with open('code/synth/opsix_important_params.json', 'w') as f:
        json.dump(important_params, f, indent=2)
    
    print(f"- code/synth/opsix_important_params.json (first 32 parameters)")
    
    return param_mapping, param_defaults, important_params

if __name__ == "__main__":
    create_opsix_config()