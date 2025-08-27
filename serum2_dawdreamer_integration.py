#!/usr/bin/env python3
"""
Serum 2 DawDreamer Integration

This script integrates converted Serum 2 JSON presets with DawDreamer to:
1. Load Serum 2 VST plugin
2. Apply preset parameters from JSON files
3. Generate 4-second audio samples
4. Create a dataset for training flow-synth models
"""

import os
import json
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Any, Tuple
import dawdreamer as daw
import librosa
import soundfile as sf
from tqdm import tqdm


class Serum2DawDreamerRenderer:
    def __init__(self, 
                 serum2_plugin_path: str = "/Library/Audio/Plug-Ins/VST3/Serum2.vst3",
                 sample_rate: int = 44100,
                 buffer_size: int = 512,
                 audio_duration: float = 4.0):
        """
        Initialize the Serum 2 DawDreamer renderer.
        
        Args:
            serum2_plugin_path: Path to Serum 2 VST3 plugin
            sample_rate: Audio sample rate
            buffer_size: Audio buffer size
            audio_duration: Duration of generated audio in seconds
        """
        self.serum2_plugin_path = serum2_plugin_path
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.audio_duration = audio_duration
        self.audio_samples = int(sample_rate * audio_duration)
        
        # Initialize DawDreamer engine
        self.engine = daw.RenderEngine(sample_rate, buffer_size)
        
        # Verify Serum 2 plugin exists
        if not os.path.exists(serum2_plugin_path):
            raise FileNotFoundError(f"Serum 2 plugin not found at: {serum2_plugin_path}")
    
    def load_serum2_plugin(self) -> daw.ProcessorBase:
        """Load the Serum 2 VST plugin."""
        try:
            # Load Serum 2 as a VST3 plugin
            serum2 = self.engine.make_plugin_processor("serum2", self.serum2_plugin_path)
            return serum2
        except Exception as e:
            print(f"Error loading Serum 2 plugin: {e}")
            raise
    
    def load_preset_json(self, json_path: Path) -> Dict[str, Any]:
        """Load a converted Serum 2 preset JSON file."""
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def extract_serum2_parameters(self, preset_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract parameter values from Serum 2 preset data for DawDreamer."""
        parameters = {}
        
        if 'data' not in preset_data:
            return parameters
        
        data = preset_data['data']
        
        # Extract parameters from different sections
        for section_name, section_data in data.items():
            if not isinstance(section_data, dict):
                continue
            
            # Handle plainParams
            if 'plainParams' in section_data:
                plain_params = section_data['plainParams']
                if isinstance(plain_params, dict):
                    for param_name, param_value in plain_params.items():
                        if isinstance(param_value, (int, float)):
                            # Normalize parameter values to 0-1 range if needed
                            normalized_value = self._normalize_parameter_value(param_value)
                            full_param_name = f"{section_name}_{param_name}"
                            parameters[full_param_name] = normalized_value
            
            # Handle direct numeric parameters
            for key, value in section_data.items():
                if key != 'plainParams' and isinstance(value, (int, float)):
                    normalized_value = self._normalize_parameter_value(value)
                    full_param_name = f"{section_name}_{key}"
                    parameters[full_param_name] = normalized_value
        
        return parameters
    
    def _normalize_parameter_value(self, value: float) -> float:
        """Normalize parameter values to 0-1 range."""
        # Most Serum parameters are already in 0-100 range, normalize to 0-1
        if value > 1.0:
            return min(value / 100.0, 1.0)
        return max(0.0, min(value, 1.0))
    
    def apply_preset_to_plugin(self, plugin: daw.ProcessorBase, parameters: Dict[str, float]) -> int:
        """Apply preset parameters to the Serum 2 plugin."""
        applied_count = 0
        
        # Get available plugin parameters
        plugin_params = plugin.get_parameter_names()
        
        for param_name, param_value in parameters.items():
            # Try to find matching parameter in plugin
            matching_param = self._find_matching_parameter(param_name, plugin_params)
            
            if matching_param:
                try:
                    plugin.set_parameter(matching_param, param_value)
                    applied_count += 1
                except Exception as e:
                    print(f"Warning: Could not set parameter {matching_param}: {e}")
        
        return applied_count
    
    def _find_matching_parameter(self, preset_param: str, plugin_params: List[str]) -> str:
        """Find matching parameter name in plugin parameter list."""
        # Direct match
        if preset_param in plugin_params:
            return preset_param
        
        # Try case-insensitive match
        preset_param_lower = preset_param.lower()
        for plugin_param in plugin_params:
            if plugin_param.lower() == preset_param_lower:
                return plugin_param
        
        # Try partial matches (remove section prefixes)
        if '_' in preset_param:
            param_suffix = preset_param.split('_', 1)[1]
            for plugin_param in plugin_params:
                if plugin_param.lower() == param_suffix.lower():
                    return plugin_param
        
        return None
    
    def generate_midi_sequence(self) -> np.ndarray:
        """Generate a simple MIDI sequence for testing presets."""
        # Create a simple chord progression
        notes = [60, 64, 67, 72]  # C major chord
        midi_data = []
        
        # Play chord for the full duration
        for note in notes:
            midi_data.extend([
                (0, 'note_on', note, 80),  # Note on at time 0
                (self.audio_duration - 0.1, 'note_off', note, 0)  # Note off near end
            ])
        
        return midi_data
    
    def render_preset_audio(self, preset_json_path: Path, output_audio_path: Path) -> bool:
        """Render audio for a single preset."""
        try:
            # Load preset data
            preset_data = self.load_preset_json(preset_json_path)
            parameters = self.extract_serum2_parameters(preset_data)
            
            # Load Serum 2 plugin
            serum2 = self.load_serum2_plugin()
            
            # Apply preset parameters
            applied_count = self.apply_preset_to_plugin(serum2, parameters)
            print(f"Applied {applied_count}/{len(parameters)} parameters for {preset_json_path.name}")
            
            # Create MIDI sequence
            midi_data = self.generate_midi_sequence()
            
            # Set up the graph
            graph = [
                (serum2, [])
            ]
            self.engine.load_graph(graph)
            
            # Add MIDI events
            for time, event_type, note, velocity in midi_data:
                sample_time = int(time * self.sample_rate)
                if event_type == 'note_on':
                    serum2.add_midi_note(note, velocity, sample_time)
                elif event_type == 'note_off':
                    serum2.add_midi_note(note, 0, sample_time)
            
            # Render audio
            audio = self.engine.render(self.audio_duration)
            
            # Ensure output directory exists
            output_audio_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save audio file
            sf.write(str(output_audio_path), audio.T, self.sample_rate)
            
            return True
            
        except Exception as e:
            print(f"Error rendering {preset_json_path}: {e}")
            return False
    
    def process_preset_directory(self, 
                               json_presets_dir: Path, 
                               output_audio_dir: Path,
                               output_metadata_path: Path = None) -> Dict[str, Any]:
        """Process all JSON presets in a directory and generate audio files."""
        json_files = list(json_presets_dir.rglob("*.json"))
        print(f"Found {len(json_files)} JSON preset files")
        
        metadata = {
            "sample_rate": self.sample_rate,
            "audio_duration": self.audio_duration,
            "total_presets": len(json_files),
            "successful_renders": 0,
            "failed_renders": 0,
            "files": []
        }
        
        for json_file in tqdm(json_files, desc="Rendering presets"):
            # Create corresponding audio file path
            rel_path = json_file.relative_to(json_presets_dir)
            audio_file = output_audio_dir / rel_path.with_suffix('.wav')
            
            # Render audio
            success = self.render_preset_audio(json_file, audio_file)
            
            file_info = {
                "preset_name": json_file.stem,
                "json_path": str(json_file),
                "audio_path": str(audio_file) if success else None,
                "success": success
            }
            
            metadata["files"].append(file_info)
            
            if success:
                metadata["successful_renders"] += 1
            else:
                metadata["failed_renders"] += 1
        
        # Save metadata
        if output_metadata_path:
            with open(output_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return metadata


def main():
    parser = argparse.ArgumentParser(description='Generate audio from Serum 2 JSON presets using DawDreamer')
    parser.add_argument('--presets-dir', '-p',
                       default='/Users/gjb/Datasets/serum2/presets',
                       help='Directory containing JSON preset files')
    parser.add_argument('--output-dir', '-o',
                       default='/Users/gjb/Datasets/serum2/audio',
                       help='Output directory for audio files')
    parser.add_argument('--serum2-plugin', '-s',
                       default='/Library/Audio/Plug-Ins/VST3/Serum2.vst3',
                       help='Path to Serum 2 VST3 plugin')
    parser.add_argument('--sample-rate', '-r',
                       type=int, default=44100,
                       help='Audio sample rate')
    parser.add_argument('--duration', '-d',
                       type=float, default=4.0,
                       help='Audio duration in seconds')
    parser.add_argument('--test-single', '-t',
                       help='Test with a single JSON preset file')
    
    args = parser.parse_args()
    
    # Initialize renderer
    renderer = Serum2DawDreamerRenderer(
        serum2_plugin_path=args.serum2_plugin,
        sample_rate=args.sample_rate,
        audio_duration=args.duration
    )
    
    if args.test_single:
        # Test with single file
        json_path = Path(args.test_single)
        if not json_path.exists():
            print(f"Test file not found: {json_path}")
            return
        
        output_path = Path(args.output_dir) / f"{json_path.stem}.wav"
        success = renderer.render_preset_audio(json_path, output_path)
        
        if success:
            print(f"Successfully rendered: {output_path}")
        else:
            print(f"Failed to render: {json_path}")
    else:
        # Process entire directory
        presets_dir = Path(args.presets_dir)
        output_dir = Path(args.output_dir)
        metadata_path = output_dir / "metadata.json"
        
        metadata = renderer.process_preset_directory(
            presets_dir, output_dir, metadata_path
        )
        
        print(f"\nProcessing complete:")
        print(f"Total presets: {metadata['total_presets']}")
        print(f"Successful renders: {metadata['successful_renders']}")
        print(f"Failed renders: {metadata['failed_renders']}")
        print(f"Success rate: {metadata['successful_renders']/metadata['total_presets']*100:.1f}%")
        print(f"Metadata saved to: {metadata_path}")


if __name__ == '__main__':
    main()