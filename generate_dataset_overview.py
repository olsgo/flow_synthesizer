#!/usr/bin/env python3
"""
Generate a comprehensive overview JSON file with parameter schemas from all presets.
"""

import json
import os
from pathlib import Path
import numpy as np
from collections import defaultdict

def generate_dataset_overview():
    """Generate overview with parameter schemas from all presets."""
    
    # Load parameter schema
    schema_path = Path('/Users/gjb/Projects/flow_synthesizer/params_schema.json')
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    
    # Load all parameter files
    params_dir = Path('/Users/gjb/Projects/flow_synthesizer/params')
    param_files = list(params_dir.glob('*.json'))
    
    print(f"Processing {len(param_files)} parameter files...")
    
    # Initialize data structures
    all_parameters = []
    parameter_stats = defaultdict(list)
    preset_data = []
    
    # Process each parameter file
    for param_file in sorted(param_files):
        try:
            with open(param_file, 'r') as f:
                data = json.load(f)
            
            preset_name = data.get('preset_name', param_file.stem)
            param_vector = data.get('parameter_vector', [])
            
            if len(param_vector) != len(schema['parameter_order']):
                print(f"Warning: {preset_name} has {len(param_vector)} params, expected {len(schema['parameter_order'])}")
                continue
            
            # Store preset data
            preset_info = {
                'preset_name': preset_name,
                'file_name': param_file.name,
                'parameter_count': len(param_vector),
                'parameters': {}
            }
            
            # Map parameters to names and collect statistics
            for i, (param_name, value) in enumerate(zip(schema['parameter_order'], param_vector)):
                preset_info['parameters'][param_name] = value
                parameter_stats[param_name].append(value)
            
            preset_data.append(preset_info)
            all_parameters.append(param_vector)
            
        except Exception as e:
            print(f"Error processing {param_file}: {e}")
    
    # Calculate parameter statistics
    param_statistics = {}
    for param_name in schema['parameter_order']:
        values = parameter_stats[param_name]
        if values:
            param_statistics[param_name] = {
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'median': float(np.median(values)),
                'unique_values': len(set(values)),
                'zero_count': sum(1 for v in values if v == 0.0),
                'non_zero_count': sum(1 for v in values if v != 0.0)
            }
    
    # Create comprehensive overview
    overview = {
        'dataset_info': {
            'plugin_name': schema['plugin_name'],
            'total_presets': len(preset_data),
            'total_parameters': schema['total_parameters'],
            'generation_timestamp': str(np.datetime64('now')),
            'parameter_vector_length': len(schema['parameter_order'])
        },
        'parameter_schema': {
            'parameter_names': schema['parameter_order'],
            'parameter_count': len(schema['parameter_order']),
            'parameter_statistics': param_statistics
        },
        'preset_summaries': [
            {
                'preset_name': preset['preset_name'],
                'file_name': preset['file_name'],
                'parameter_count': preset['parameter_count'],
                'active_parameters': sum(1 for v in preset['parameters'].values() if v != 0.0),
                'zero_parameters': sum(1 for v in preset['parameters'].values() if v == 0.0)
            }
            for preset in preset_data
        ],
        'detailed_presets': preset_data,
        'global_statistics': {
            'parameters_with_variation': sum(1 for stats in param_statistics.values() if stats['std'] > 0.001),
            'parameters_always_zero': sum(1 for stats in param_statistics.values() if stats['max'] == 0.0),
            'parameters_always_one': sum(1 for stats in param_statistics.values() if stats['min'] == 1.0 and stats['max'] == 1.0),
            'most_varied_parameters': sorted(
                [(name, stats['std']) for name, stats in param_statistics.items()],
                key=lambda x: x[1], reverse=True
            )[:10],
            'least_varied_parameters': sorted(
                [(name, stats['std']) for name, stats in param_statistics.items()],
                key=lambda x: x[1]
            )[:10]
        }
    }
    
    # Save overview file
    overview_path = Path('/Users/gjb/Projects/flow_synthesizer/dataset_overview.json')
    with open(overview_path, 'w') as f:
        json.dump(overview, f, indent=2)
    
    print(f"\nDataset overview saved to: {overview_path}")
    print(f"Total presets processed: {len(preset_data)}")
    print(f"Parameters per preset: {schema['total_parameters']}")
    print(f"Parameters with variation: {overview['global_statistics']['parameters_with_variation']}")
    print(f"Parameters always zero: {overview['global_statistics']['parameters_always_zero']}")
    
    # Show most and least varied parameters
    print("\nMost varied parameters:")
    for name, std in overview['global_statistics']['most_varied_parameters'][:5]:
        print(f"  {name}: std={std:.4f}")
    
    print("\nLeast varied parameters:")
    for name, std in overview['global_statistics']['least_varied_parameters'][:5]:
        print(f"  {name}: std={std:.4f}")
    
    return overview_path

if __name__ == "__main__":
    generate_dataset_overview()