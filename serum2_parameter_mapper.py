#!/usr/bin/env python3
"""
Serum 2 Parameter Mapper for DawDreamer Integration

This script creates a parameter mapping between Serum 2 preset parameters
(from converted JSON files) and DawDreamer plugin parameters, using techniques
inspired by the XML synth sound rendering example.

Key features:
- Manual mapping dictionary for known parameter correspondences
- Fuzzy string matching for unmapped parameters
- JSON-based parameter mapping storage and loading
- Support for Serum 2 VST3 plugin integration
"""

import os
import json
import re
import difflib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import dawdreamer as daw


# Manual mapping dictionary for Serum 2 parameters
# Based on analysis of plugin parameter names vs preset parameter names
SERUM2_TO_DAWDREAMER_MAPPING = {
    # Oscillator parameters
    ".Osc0.kParamLevel": "A Level",
    ".Osc1.kParamLevel": "B Level",
    ".Osc0.kParamPan": "A Pan",
    ".Osc1.kParamPan": "B Pan",
    ".Osc0.kParamDetune": "A Fine",
    ".Osc1.kParamDetune": "B Fine",
    ".Osc0.kParamCoarse": "A Coarse",
    ".Osc1.kParamCoarse": "B Coarse",
    ".Osc0.kParamPhase": "A Phase",
    ".Osc1.kParamPhase": "B Phase",
    
    # Filter parameters
    ".Filter0.kParamCutoff": "Filter Cutoff",
    ".Filter0.kParamResonance": "Filter Reso",
    ".Filter0.kParamDrive": "Filter Drive",
    ".Filter0.kParamFat": "Filter Fat",
    ".Filter0.kParamKeytrack": "Filter Key",
    
    # Envelope parameters
    ".Env0.kParamAttack": "Env 1 A",
    ".Env0.kParamDecay": "Env 1 D",
    ".Env0.kParamSustain": "Env 1 S",
    ".Env0.kParamRelease": "Env 1 R",
    ".Env1.kParamAttack": "Env 2 A",
    ".Env1.kParamDecay": "Env 2 D",
    ".Env1.kParamSustain": "Env 2 S",
    ".Env1.kParamRelease": "Env 2 R",
    ".Env2.kParamAttack": "Env 3 A",
    ".Env2.kParamDecay": "Env 3 D",
    ".Env2.kParamSustain": "Env 3 S",
    ".Env2.kParamRelease": "Env 3 R",
    ".Env3.kParamAttack": "Env 4 A",
    ".Env3.kParamDecay": "Env 4 D",
    ".Env3.kParamSustain": "Env 4 S",
    ".Env3.kParamRelease": "Env 4 R",
    
    # LFO parameters
    ".Lfo0.kParamRate": "LFO 1 Rate",
    ".Lfo1.kParamRate": "LFO 2 Rate",
    ".Lfo2.kParamRate": "LFO 3 Rate",
    ".Lfo3.kParamRate": "LFO 4 Rate",
    
    # Global parameters
    ".Global.kParamMasterVolume": "Master",
    ".Global.kParamMasterTune": "Master Tune",
    ".Global.kParamPolyphony": "Voices",
    ".Global.kParamPortamento": "Glide",
    ".Global.kParamBendRange": "Bend Range",
    
    # Effects parameters
    ".Fx0.kParamMix": "FX Mix",
    ".Fx1.kParamMix": "FX2 Mix",
    ".Compressor.kParamThreshold": "Comp Thresh",
    ".Compressor.kParamRatio": "Comp Ratio",
    ".Compressor.kParamAttack": "Comp Attack",
    ".Compressor.kParamRelease": "Comp Release",
    
    # Unison parameters
    ".Osc0.kParamUnisonVoices": "A Uni Voices",
    ".Osc1.kParamUnisonVoices": "B Uni Voices",
    ".Osc0.kParamUnisonDetune": "A Uni Detune",
    ".Osc1.kParamUnisonDetune": "B Uni Detune",
    ".Osc0.kParamUnisonBlend": "A Uni Blend",
    ".Osc1.kParamUnisonBlend": "B Uni Blend",
    ".Osc0.kParamUnisonWarp": "A Uni Warp",
    ".Osc1.kParamUnisonWarp": "B Uni Warp",
}


class Serum2ParameterMapper:
    """Handles parameter mapping between Serum 2 presets and DawDreamer plugin."""
    
    def __init__(self, plugin_path: str, mapping_cache_dir: str = "serum2_mappings"):
        self.plugin_path = plugin_path
        self.mapping_cache_dir = Path(mapping_cache_dir)
        self.mapping_cache_dir.mkdir(exist_ok=True)
        
        # Initialize DawDreamer engine and plugin with larger buffer size
        self.engine = daw.RenderEngine(44100, 1024)
        self.synth = self.engine.make_plugin_processor("serum2", plugin_path)
        
        # Load the graph with the synth
        graph = [(self.synth, [])]
        self.engine.load_graph(graph)
        
        # Load plugin parameters
        self.plugin_params = self.synth.get_parameters_description()
        self.param_name_to_index = {param["name"]: param["index"] for param in self.plugin_params}
        
        print(f"Loaded Serum 2 plugin with {len(self.plugin_params)} parameters")
    
    def find_closest_match(self, preset_param: str, threshold: float = 0.6) -> Optional[str]:
        """Find the closest matching plugin parameter using fuzzy string matching."""
        # Remove common prefixes and clean up parameter names
        cleaned_preset = preset_param.replace(".", " ").replace("kParam", "").strip()
        
        best_match = None
        best_ratio = 0.0
        
        for plugin_param in self.param_name_to_index.keys():
            # Calculate similarity ratio
            ratio = difflib.SequenceMatcher(None, cleaned_preset.lower(), plugin_param.lower()).ratio()
            
            if ratio > best_ratio and ratio >= threshold:
                best_ratio = ratio
                best_match = plugin_param
        
        return best_match
    
    def create_parameter_mapping(self, preset_path: str, verbose: bool = True) -> str:
        """Create a JSON parameter mapping file for a specific preset."""
        preset_name = Path(preset_path).stem
        output_path = self.mapping_cache_dir / f"serum2-{preset_name}-parameter-mapping.json"
        
        if output_path.exists():
            if verbose:
                print(f"Using cached mapping: {output_path}")
            return str(output_path)
        
        # Load preset JSON
        with open(preset_path, 'r') as f:
            preset_data = json.load(f)
        
        parameter_mapping = {}
        mapped_count = 0
        fuzzy_mapped_count = 0
        
        # Extract parameters from nested data structure
        def extract_parameters(data, prefix=""):
            """Recursively extract parameters from nested preset data."""
            params = {}
            if isinstance(data, dict):
                for key, value in data.items():
                    if key == "plainParams" and isinstance(value, dict):
                        # Extract parameters from plainParams
                        for param_key, param_value in value.items():
                            if isinstance(param_value, (int, float)):
                                full_key = f"{prefix}.{param_key}" if prefix else param_key
                                params[full_key] = param_value
                    elif isinstance(value, dict):
                        # Recurse into nested objects
                        nested_prefix = f"{prefix}.{key}" if prefix else key
                        params.update(extract_parameters(value, nested_prefix))
                    elif isinstance(value, (int, float)):
                        # Direct numeric parameter
                        full_key = f"{prefix}.{key}" if prefix else key
                        params[full_key] = value
            return params
        
        # Extract parameters from the preset data
        if "data" in preset_data:
            extracted_params = extract_parameters(preset_data["data"])
        else:
            extracted_params = extract_parameters(preset_data)
        
        print(f"Extracted {len(extracted_params)} parameters from preset")
        if len(extracted_params) > 0:
            print("Sample parameters:")
            for i, (key, value) in enumerate(list(extracted_params.items())[:10]):
                print(f"  {key}: {value}")
        
        # Process each extracted parameter
        for param_name, param_value in extracted_params.items():
            if not isinstance(param_value, (int, float)):
                continue
                
            plugin_param = None
            mapping_type = "unmapped"
            
            # Try manual mapping first
            if param_name in SERUM2_TO_DAWDREAMER_MAPPING:
                plugin_param = SERUM2_TO_DAWDREAMER_MAPPING[param_name]
                mapping_type = "manual"
                mapped_count += 1
            else:
                # Try fuzzy matching
                plugin_param = self.find_closest_match(param_name)
                if plugin_param:
                    mapping_type = "fuzzy"
                    fuzzy_mapped_count += 1
            
            if plugin_param and plugin_param in self.param_name_to_index:
                parameter_mapping[param_name] = {
                    'match': plugin_param,
                    'value': float(param_value),
                    'index': self.param_name_to_index[plugin_param],
                    'type': mapping_type
                }
            elif verbose:
                print(f"No mapping found for: {param_name}")
        
        # Save mapping to JSON
        with open(output_path, 'w') as f:
            json.dump(parameter_mapping, f, indent=2)
        
        if verbose:
            total_params = len(extracted_params)
            print(f"Parameter mapping created: {output_path}")
            print(f"  Total preset parameters: {total_params}")
            print(f"  Manual mappings: {mapped_count}")
            print(f"  Fuzzy mappings: {fuzzy_mapped_count}")
            print(f"  Total mapped: {len(parameter_mapping)}")
            if total_params > 0:
                print(f"  Success rate: {len(parameter_mapping)/total_params*100:.1f}%")
            else:
                print(f"  Success rate: 0.0%")
        
        return str(output_path)
    
    def load_preset_with_mapping(self, preset_path: str, verbose: bool = True) -> int:
        """Load a preset using parameter mapping and return the number of successfully set parameters."""
        # Create or load parameter mapping
        mapping_path = self.create_parameter_mapping(preset_path, verbose)
        
        # Load mapping
        with open(mapping_path, 'r') as f:
            parameter_mapping = json.load(f)
        
        # Apply parameters to plugin
        success_count = 0
        for param_name, mapping_info in parameter_mapping.items():
            try:
                self.synth.set_parameter(mapping_info['index'], mapping_info['value'])
                success_count += 1
                if verbose:
                    print(f"Set {mapping_info['match']} = {mapping_info['value']} ({mapping_info['type']})")
            except Exception as e:
                if verbose:
                    print(f"Failed to set {param_name}: {e}")
        
        if verbose:
            print(f"Successfully set {success_count}/{len(parameter_mapping)} parameters")
        
        return success_count
    
    def test_preset_loading(self, preset_path: str) -> bool:
        """Test loading a preset and rendering audio."""
        print(f"\nTesting preset: {Path(preset_path).name}")
        
        # Load preset with mapping
        success_count = self.load_preset_with_mapping(preset_path)
        
        # Test audio rendering
        try:
            # Add a test note
            self.synth.add_midi_note(60, 100, 0.0, 2.0)  # C4, velocity 100, 2 seconds
            
            # Render audio
            self.engine.render(3.0)  # 3 seconds total
            audio = self.engine.get_audio()
            
            # Clear MIDI for next test
            self.synth.clear_midi()
            
            # Check if audio was generated
            max_amplitude = float(abs(audio).max())
            audio_generated = max_amplitude > 1e-6
            
            print(f"Audio generated: {audio_generated} (max amplitude: {max_amplitude:.6f})")
            print(f"Parameters set: {success_count}")
            
            return audio_generated and success_count > 0
            
        except Exception as e:
            print(f"Error during audio rendering: {e}")
            return False


def main():
    """Main function to test the parameter mapper."""
    # Configuration
    plugin_path = "/Library/Audio/Plug-Ins/VST3/Serum2.vst3"
    converted_presets_dir = "converted_presets"
    
    # Check if plugin exists
    if not os.path.exists(plugin_path):
        print(f"Error: Serum 2 plugin not found at {plugin_path}")
        return
    
    # Find a test preset
    test_preset = None
    for root, dirs, files in os.walk(converted_presets_dir):
        for file in files:
            if file.endswith('.json'):
                test_preset = os.path.join(root, file)
                break
        if test_preset:
            break
    
    if not test_preset:
        print(f"Error: No converted presets found in {converted_presets_dir}")
        return
    
    # Initialize mapper and test
    try:
        mapper = Serum2ParameterMapper(plugin_path)
        success = mapper.test_preset_loading(test_preset)
        
        if success:
            print("\n✅ Preset loading test PASSED!")
        else:
            print("\n❌ Preset loading test FAILED!")
            
    except Exception as e:
        print(f"Error initializing mapper: {e}")


if __name__ == "__main__":
    main()