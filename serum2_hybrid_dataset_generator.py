#!/usr/bin/env python3
"""
Serum 2 Hybrid Dataset Generator

This script combines approaches from:
1. DawDreamer examples (xml_synth_sound_rendering) - parameter mapping methodology
2. Flow-synth developers - VST integration and audio generation

Key differences from previous attempts:
- Uses DawDreamer's parameter mapping approach (like TAL-UNO example)
- Creates JSON parameter mappings for caching
- Applies fuzzy matching for unmapped parameters
- Uses proper VST3 preset loading when possible
"""

import os
import json
import difflib
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import dawdreamer as daw
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

# Import our existing parameter mapping
from serum2_parameter_mapping import SERUM2_PARAMETER_MAPPING


class Serum2HybridRenderer:
    """Hybrid renderer combining DawDreamer and flow-synth approaches."""
    
    def __init__(self, plugin_path: str, sample_rate: int = 44100, block_size: int = 512):
        self.plugin_path = plugin_path
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.engine = None
        self.synth = None
        self.vst_params = None
        self.param_name_to_index = None
        self.mapping_cache_dir = Path("serum2_mappings")
        self.mapping_cache_dir.mkdir(exist_ok=True)
        
    def startup(self):
        """Initialize the DawDreamer engine and plugin."""
        self.engine = daw.RenderEngine(self.sample_rate, self.block_size)
        self.synth = self.engine.make_plugin_processor("serum2", self.plugin_path)
        
        # Load graph
        graph = [(self.synth, [])]
        self.engine.load_graph(graph)
        
        # Get VST parameter information
        self.vst_params = self.synth.get_parameters_description()
        self.param_name_to_index = {param["name"]: param["index"] for param in self.vst_params}
        
        print(f"Initialized Serum 2 with {len(self.vst_params)} parameters")
        
    def make_parameter_mapping(self, preset_path: str, preset_data: dict) -> str:
        """Create parameter mapping JSON file (similar to TAL-UNO example)."""
        preset_name = Path(preset_path).stem
        mapping_file = self.mapping_cache_dir / f"serum2-{preset_name}-parameter-mapping.json"
        
        if mapping_file.exists():
            return str(mapping_file)
            
        print(f"Creating parameter mapping for {preset_name}...")
        
        # Extract parameters from dawdreamer_params section
        dawdreamer_params = preset_data.get('dawdreamer_params', {})
        
        parameter_mapping = {}
        mapped_count = 0
        
        for preset_param, value in dawdreamer_params.items():
            if not preset_param.startswith('.'):
                continue
                
            vst_param_name = None
            param_index = None
            
            # 1. Try exact mapping from our dictionary
            if preset_param in SERUM2_PARAMETER_MAPPING:
                vst_param_name = SERUM2_PARAMETER_MAPPING[preset_param]
                if vst_param_name in self.param_name_to_index:
                    param_index = self.param_name_to_index[vst_param_name]
                    
            # 2. Try fuzzy matching (similar to TAL-UNO approach)
            if param_index is None:
                vst_param_name = self._find_closest_parameter_match(preset_param)
                if vst_param_name and vst_param_name in self.param_name_to_index:
                    param_index = self.param_name_to_index[vst_param_name]
                    
            # 3. Store mapping if found
            if param_index is not None and isinstance(value, (int, float)):
                # Normalize value to 0.0-1.0 range if needed
                normalized_value = self._normalize_parameter_value(value)
                
                parameter_mapping[preset_param] = {
                    'match': vst_param_name,
                    'value': normalized_value,
                    'index': param_index,
                    'original_value': value
                }
                mapped_count += 1
                
        print(f"  Mapped {mapped_count}/{len(dawdreamer_params)} parameters")
        
        # Save mapping to JSON file
        with open(mapping_file, 'w') as f:
            json.dump(parameter_mapping, f, indent=2)
            
        return str(mapping_file)
        
    def _find_closest_parameter_match(self, preset_param: str) -> Optional[str]:
        """Find closest VST parameter using fuzzy matching."""
        # Clean up preset parameter name for matching
        clean_preset = preset_param.replace('.', '').replace('kParam', '').lower()
        
        best_match = None
        best_ratio = 0.6  # Minimum similarity threshold
        
        for vst_param in self.param_name_to_index.keys():
            clean_vst = vst_param.lower().replace(' ', '')
            
            # Try different matching strategies
            ratios = [
                difflib.SequenceMatcher(None, clean_preset, clean_vst).ratio(),
                difflib.SequenceMatcher(None, clean_preset, vst_param.lower()).ratio()
            ]
            
            max_ratio = max(ratios)
            if max_ratio > best_ratio:
                best_ratio = max_ratio
                best_match = vst_param
                
        return best_match
        
    def _normalize_parameter_value(self, value: float) -> float:
        """Normalize parameter value to 0.0-1.0 range."""
        if 0.0 <= value <= 1.0:
            return value
        elif value > 1.0:
            # Handle different scaling scenarios
            if value <= 100.0:
                return value / 100.0
            elif value <= 127.0:
                return value / 127.0  # MIDI-style scaling
            else:
                # For very large values, use log scaling or cap at 1.0
                return min(1.0, value / 1000.0)
        else:
            # Handle negative values - map to 0.0-1.0 range
            if value >= -100.0:
                return (value + 100.0) / 200.0  # Map -100 to 100 -> 0 to 1
            else:
                return 0.0
            
    def load_preset_mapping(self, mapping_file: str):
        """Load preset using parameter mapping (similar to TAL-UNO example)."""
        with open(mapping_file, 'r') as f:
            parameter_mapping = json.load(f)
            
        applied_count = 0
        
        # Reset plugin to default state first
        self.startup()  # Reinitialize to reset state
        
        # Apply mapped parameters
        for preset_param, mapping_info in parameter_mapping.items():
            try:
                param_index = mapping_info['index']
                param_value = mapping_info['value']
                
                # Validate parameter value
                if 0.0 <= param_value <= 1.0:
                    self.synth.set_parameter(param_index, param_value)
                    applied_count += 1
                else:
                    print(f"  ⚠️  Invalid value for {preset_param}: {param_value}")
                    
            except Exception as e:
                print(f"  ❌ Error applying {preset_param}: {e}")
                
        print(f"  ✅ Applied {applied_count}/{len(parameter_mapping)} parameters")
        return applied_count
        
    def render_audio(self, midi_note: int = 60, velocity: int = 100, 
                    note_duration: float = 3.0, render_duration: float = 4.0) -> np.ndarray:
        """Render audio with current preset settings."""
        self.synth.clear_midi()
        self.synth.add_midi_note(midi_note, velocity, 0.0, note_duration)
        self.engine.render(render_duration)
        
        audio = self.engine.get_audio()  # (channels, samples)
        
        # Convert to mono
        if audio.shape[0] > 1:
            audio_mono = np.mean(audio, axis=0)
        else:
            audio_mono = audio[0]
            
        return audio_mono
        
    def process_preset(self, preset_path: str) -> Tuple[Optional[np.ndarray], dict]:
        """Process a single preset file and return audio + metadata."""
        try:
            # Load preset JSON
            with open(preset_path, 'r') as f:
                preset_data = json.load(f)
                
            # Create parameter mapping
            mapping_file = self.make_parameter_mapping(preset_path, preset_data)
            
            # Load preset using mapping
            applied_params = self.load_preset_mapping(mapping_file)
            
            # Render audio
            audio = self.render_audio()
            
            # Create metadata
            metadata = {
                'preset_name': Path(preset_path).stem,
                'preset_path': str(preset_path),
                'mapping_file': mapping_file,
                'applied_parameters': applied_params,
                'audio_max_amplitude': float(np.max(np.abs(audio))),
                'audio_shape': audio.shape,
                'sample_rate': self.sample_rate
            }
            
            return audio, metadata
            
        except Exception as e:
            print(f"  ❌ Error processing {preset_path}: {e}")
            return None, {'error': str(e), 'preset_path': str(preset_path)}


def generate_hybrid_dataset(preset_dir: str, output_dir: str, max_presets: int = None):
    """Generate dataset using hybrid approach."""
    
    plugin_path = "/Library/Audio/Plug-Ins/VST3/Serum2.vst3"
    preset_dir = Path(preset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🎵 Serum 2 Hybrid Dataset Generation")
    print("=" * 50)
    print(f"Plugin: {plugin_path}")
    print(f"Presets: {preset_dir}")
    print(f"Output: {output_dir}")
    
    # Find preset files
    preset_files = list(preset_dir.glob("**/*.json"))
    if max_presets:
        preset_files = preset_files[:max_presets]
        
    print(f"Found {len(preset_files)} preset files")
    
    # Initialize renderer
    renderer = Serum2HybridRenderer(plugin_path)
    renderer.startup()
    
    # Process presets
    successful_renders = 0
    failed_renders = 0
    all_metadata = []
    
    for i, preset_file in enumerate(tqdm(preset_files, desc="Processing presets")):
        print(f"\n[{i+1}/{len(preset_files)}] {preset_file.name}")
        
        audio, metadata = renderer.process_preset(preset_file)
        
        if audio is not None:
            # Save audio
            audio_filename = f"{metadata['preset_name']}_{i+1:03d}.wav"
            audio_path = output_dir / audio_filename
            
            # Normalize audio to prevent clipping
            max_amp = np.max(np.abs(audio))
            if max_amp > 0:
                audio_normalized = audio / max_amp * 0.8  # Leave some headroom
            else:
                audio_normalized = audio
                
            wavfile.write(str(audio_path), renderer.sample_rate, audio_normalized)
            
            metadata['audio_file'] = audio_filename
            successful_renders += 1
            
            print(f"  🔊 Audio: max amplitude = {max_amp:.4f}")
            print(f"  💾 Saved: {audio_filename}")
        else:
            failed_renders += 1
            
        all_metadata.append(metadata)
        
    # Save metadata
    metadata_file = output_dir / "hybrid_dataset_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(all_metadata, f, indent=2)
        
    # Summary
    print(f"\n🎵 Hybrid Dataset Generation Complete!")
    print(f"Successful renders: {successful_renders}")
    print(f"Failed renders: {failed_renders}")
    print(f"Success rate: {successful_renders/(successful_renders+failed_renders)*100:.1f}%")
    print(f"Output directory: {output_dir}")
    print(f"Metadata saved: {metadata_file}")


if __name__ == "__main__":
    # Test with a larger subset of presets
    generate_hybrid_dataset(
        preset_dir="converted_presets",
        output_dir="/Users/gjb/Datasets/serum2/hybrid_renders_v2",
        max_presets=15  # Test with 15 presets to validate improvements
    )