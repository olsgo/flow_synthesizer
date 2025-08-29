#!/usr/bin/env python3
"""
Extract Parameter Vectors from PolyMAX Presets

This script loads each PolyMAX preset and extracts normalized [0,1] parameter vectors
for training the flow synthesizer model.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from pedalboard import load_plugin, VST3Plugin
from polymax_helpers import find_vstpreset_files, create_safe_filename

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PolyMAXParameterExtractor:
    """Extract normalized parameter vectors from PolyMAX presets."""
    
    def __init__(self):
        self.vst3_path = '/Library/Audio/Plug-Ins/VST3/uaudio_polymax.vst3'
        self.presets_dir = '/Library/Audio/Presets/UADx PolyMAX Synth'
        self.params_dir = '/Users/gjb/Projects/flow_synthesizer/params'
        self.plugin = None
        self.param_schema = None
        
        # Create params directory if it doesn't exist
        os.makedirs(self.params_dir, exist_ok=True)
        
        # Load parameter schema
        self.load_parameter_schema()
        
    def load_parameter_schema(self):
        """Load the parameter schema to ensure consistent ordering."""
        schema_path = '/Users/gjb/Projects/flow_synthesizer/params_schema.json'
        try:
            with open(schema_path, 'r') as f:
                self.param_schema = json.load(f)
            logger.info(f"Loaded parameter schema with {len(self.param_schema['parameter_order'])} parameters")
        except Exception as e:
            logger.error(f"Failed to load parameter schema: {e}")
            raise
            
    def initialize_plugin(self):
        """Initialize the PolyMAX VST3 plugin."""
        try:
            self.plugin = load_plugin(self.vst3_path)
            logger.info(f"Loaded PolyMAX plugin with {len(self.plugin.parameters)} parameters")
            return True
        except Exception as e:
            logger.error(f"Failed to load PolyMAX plugin: {e}")
            return False
            
    def get_parameter_vector(self) -> List[float]:
        """Extract normalized parameter vector in schema order."""
        if not self.plugin:
            return []
            
        # Get current parameter values
        current_params = {
            key: param.raw_value 
            for key, param in self.plugin.parameters.items()
        }
        
        # Create ordered vector based on schema
        param_vector = []
        for param_name in self.param_schema['parameter_order']:
            # Try to find matching parameter by name
            param_value = 0.0  # Default value
            
            # Look for exact match first
            if param_name in current_params:
                param_value = current_params[param_name]
            else:
                # Try to find by index (fallback)
                param_index = len(param_vector)
                if param_index < len(current_params):
                    param_keys = list(current_params.keys())
                    if param_index < len(param_keys):
                        param_value = current_params[param_keys[param_index]]
                        
            param_vector.append(float(param_value))
            
        return param_vector
        
    def load_preset_and_extract_params(self, preset_path: str) -> Optional[List[float]]:
        """Load a preset and extract its parameter vector."""
        try:
            # Load the preset using pedalboard's native method
            self.plugin.load_preset(preset_path)
            logger.debug(f"Loaded preset: {preset_path}")
            
            # Extract parameter vector
            param_vector = self.get_parameter_vector()
            
            if len(param_vector) != len(self.param_schema['parameter_order']):
                logger.warning(f"Parameter vector length mismatch: got {len(param_vector)}, expected {len(self.param_schema['parameter_order'])}")
                
            return param_vector
            
        except Exception as e:
            logger.error(f"Failed to load preset {preset_path}: {e}")
            return None
            
    def save_parameter_vector(self, preset_name: str, param_vector: List[float]) -> bool:
        """Save parameter vector to JSON file."""
        try:
            safe_name = create_safe_filename(preset_name)
            params_file = os.path.join(self.params_dir, f"{safe_name}.json")
            
            param_data = {
                "preset_name": preset_name,
                "parameter_vector": param_vector,
                "parameter_count": len(param_vector),
                "schema_version": "1.0"
            }
            
            with open(params_file, 'w') as f:
                json.dump(param_data, f, indent=2)
                
            logger.debug(f"Saved parameters for {preset_name} to {params_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save parameters for {preset_name}: {e}")
            return False
            
    def process_all_presets(self) -> Dict[str, bool]:
        """Process all presets and extract their parameter vectors."""
        if not self.initialize_plugin():
            return {}
            
        # Find all preset files
        preset_files = find_vstpreset_files(self.presets_dir)
        logger.info(f"Found {len(preset_files)} preset files")
        
        results = {}
        processed = 0
        
        for preset_path in preset_files:
            preset_name = Path(preset_path).stem
            logger.info(f"Processing preset {processed + 1}/{len(preset_files)}: {preset_name}")
            
            # Extract parameters
            param_vector = self.load_preset_and_extract_params(preset_path)
            
            if param_vector is not None:
                # Save parameter vector
                success = self.save_parameter_vector(preset_name, param_vector)
                results[preset_name] = success
                
                if success:
                    processed += 1
                    logger.info(f"✓ Extracted {len(param_vector)} parameters for {preset_name}")
                else:
                    logger.error(f"✗ Failed to save parameters for {preset_name}")
            else:
                results[preset_name] = False
                logger.error(f"✗ Failed to extract parameters for {preset_name}")
                
        logger.info(f"\nParameter extraction complete: {processed}/{len(preset_files)} presets processed successfully")
        return results
        
def main():
    """Main function to extract parameters from all presets."""
    extractor = PolyMAXParameterExtractor()
    results = extractor.process_all_presets()
    
    # Print summary
    successful = sum(1 for success in results.values() if success)
    total = len(results)
    
    print(f"\n=== Parameter Extraction Summary ===")
    print(f"Total presets: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {total - successful}")
    print(f"Success rate: {successful/total*100:.1f}%" if total > 0 else "No presets processed")
    
    return 0 if successful == total else 1
    
if __name__ == '__main__':
    exit(main())