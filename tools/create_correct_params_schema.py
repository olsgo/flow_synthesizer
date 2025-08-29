#!/usr/bin/env python3
"""
Create Correct Parameter Schema for PolyMAX

Extracts the actual parameter names from PolyMAX VST3 plugin
and creates a corrected params_schema.json with 66 parameters.
"""

import json
import logging
from pedalboard import load_plugin

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_correct_schema():
    """Create corrected parameter schema from actual plugin parameters."""
    try:
        # Load PolyMAX plugin
        plugin = load_plugin('/Library/Audio/Plug-Ins/VST3/uaudio_polymax.vst3')
        
        # Extract actual parameter names in order
        actual_params = list(plugin.parameters.keys())
        
        logger.info(f"Found {len(actual_params)} actual parameters")
        
        # Create corrected schema
        corrected_schema = {
            "plugin_name": "UAD PolyMAX",
            "total_parameters": len(actual_params),
            "parameter_order": actual_params,
            "schema_version": "2.0",
            "notes": "Corrected schema using actual VST3 parameters exposed by pedalboard"
        }
        
        # Save corrected schema
        with open('/Users/gjb/Projects/flow_synthesizer/params_schema.json', 'w') as f:
            json.dump(corrected_schema, f, indent=2)
            
        logger.info(f"Created corrected params_schema.json with {len(actual_params)} parameters")
        
        # Print parameter list for verification
        print("\nActual PolyMAX Parameters:")
        for i, param_name in enumerate(actual_params):
            print(f"  {i:2d}: {param_name}")
            
        return True
        
    except Exception as e:
        logger.error(f"Failed to create corrected schema: {e}")
        return False

if __name__ == '__main__':
    success = create_correct_schema()
    exit(0 if success else 1)