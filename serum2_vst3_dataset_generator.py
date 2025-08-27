#!/usr/bin/env python3
"""
Serum 2 VST3 Dataset Generator
Uses VST3 preset loading instead of parameter mapping to avoid glitchy audio.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy.io import wavfile
from tqdm import tqdm

# Import DDRenderer for cleaner VST3 handling
from code.dd_renderer import DDRenderer

class Serum2VST3Renderer:
    """Serum 2 renderer using VST3 preset loading."""
    
    def __init__(self, plugin_path: str, sample_rate: int = 44100, block_size: int = 512):
        self.plugin_path = plugin_path
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.renderer = None
        
    def startup(self):
        """Initialize the VST3 plugin."""
        self.renderer = DDRenderer(sample_rate=self.sample_rate, block_size=self.block_size)
        
        # Load plugin
        success = self.renderer.load_plugin(self.plugin_path)
        if not success:
            raise RuntimeError(f"Failed to load plugin: {self.plugin_path}")
            
        print(f"  ✅ Loaded Serum 2 VST3 plugin")
        return True
        
    def load_vst3_preset(self, preset_path: str) -> bool:
        """Load VST3 preset file directly."""
        if not self.renderer:
            raise RuntimeError("Renderer not initialized. Call startup() first.")
            
        try:
            # Try loading as VST3 preset
            success = self.renderer.load_vst3_preset(preset_path)
            if success:
                print(f"  ✅ Loaded VST3 preset: {Path(preset_path).name}")
                return True
            else:
                print(f"  ❌ Failed to load VST3 preset: {Path(preset_path).name}")
                return False
        except Exception as e:
            print(f"  ❌ Error loading preset {Path(preset_path).name}: {e}")
            return False
            
    def load_plugin_state(self, state_path: str) -> bool:
        """Load plugin state file."""
        if not self.renderer:
            raise RuntimeError("Renderer not initialized. Call startup() first.")
            
        try:
            success = self.renderer.load_state(state_path)
            if success:
                print(f"  ✅ Loaded plugin state: {Path(state_path).name}")
                return True
            else:
                print(f"  ❌ Failed to load plugin state: {Path(state_path).name}")
                return False
        except Exception as e:
            print(f"  ❌ Error loading state {Path(state_path).name}: {e}")
            return False
            
    def render_audio(self, midi_note: int = 60, velocity: int = 100, 
                    note_duration: float = 3.0, render_duration: float = 4.0) -> np.ndarray:
        """Render audio with current preset."""
        if not self.renderer:
            raise RuntimeError("Renderer not initialized. Call startup() first.")
            
        # Render audio
        audio_stereo = self.renderer.render_patch(
            midi_note=midi_note, 
            velocity=velocity, 
            note_len_sec=note_duration, 
            render_len_sec=render_duration
        )
        
        # Convert to mono
        if len(audio_stereo.shape) > 1:
            audio_mono = audio_stereo[0]  # Take left channel
        else:
            audio_mono = audio_stereo
            
        return audio_mono
        
    def process_preset(self, preset_path: str) -> Tuple[Optional[np.ndarray], dict]:
        """Process a single preset and return audio + metadata."""
        preset_name = Path(preset_path).stem
        
        try:
            # Try loading as VST3 preset first
            success = self.load_vst3_preset(preset_path)
            
            # If VST3 preset loading fails, try as plugin state
            if not success:
                success = self.load_plugin_state(preset_path)
                
            if not success:
                return None, {
                    'preset_name': preset_name,
                    'preset_path': preset_path,
                    'error': 'Failed to load preset/state',
                    'success': False
                }
                
            # Render audio
            audio = self.render_audio()
            
            # Calculate audio statistics
            max_amplitude = float(np.max(np.abs(audio)))
            rms_level = float(np.sqrt(np.mean(audio**2)))
            
            metadata = {
                'preset_name': preset_name,
                'preset_path': preset_path,
                'audio_max_amplitude': max_amplitude,
                'audio_rms_level': rms_level,
                'audio_shape': audio.shape,
                'sample_rate': self.sample_rate,
                'success': True
            }
            
            return audio, metadata
            
        except Exception as e:
            print(f"  ❌ Error processing {preset_name}: {e}")
            return None, {
                'preset_name': preset_name,
                'preset_path': preset_path,
                'error': str(e),
                'success': False
            }

def find_preset_files(preset_dir: str) -> List[str]:
    """Find all potential preset files."""
    preset_dir = Path(preset_dir)
    preset_files = []
    
    # Look for various preset file types
    extensions = ['.vstpreset', '.fxp', '.SerumPreset', '.json', '.state']
    
    for ext in extensions:
        preset_files.extend(list(preset_dir.rglob(f'*{ext}')))
        
    return [str(p) for p in preset_files]

def generate_vst3_dataset(preset_dir: str, output_dir: str, max_presets: int = None):
    """Generate dataset using VST3 preset loading."""
    print("🎵 Serum 2 VST3 Dataset Generator")
    print("=" * 50)
    
    # Setup paths
    plugin_path = "/Library/Audio/Plug-Ins/VST3/Serum2.vst3"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find preset files
    preset_files = find_preset_files(preset_dir)
    if not preset_files:
        print(f"❌ No preset files found in {preset_dir}")
        return
        
    if max_presets:
        preset_files = preset_files[:max_presets]
        
    print(f"📁 Found {len(preset_files)} preset files")
    print(f"📂 Output directory: {output_dir}")
    
    # Initialize renderer
    renderer = Serum2VST3Renderer(plugin_path)
    renderer.startup()
    
    # Process presets
    dataset_metadata = []
    success_count = 0
    
    for i, preset_path in enumerate(tqdm(preset_files, desc="Processing presets")):
        print(f"\n🎛️  Processing preset {i+1}/{len(preset_files)}: {Path(preset_path).name}")
        
        # Process preset
        audio, metadata = renderer.process_preset(preset_path)
        
        if audio is not None:
            # Save audio file
            preset_name = metadata['preset_name']
            audio_filename = f"{preset_name}.wav"
            audio_path = output_dir / audio_filename
            
            # Normalize audio to prevent clipping
            if metadata['audio_max_amplitude'] > 0:
                audio_normalized = audio / metadata['audio_max_amplitude'] * 0.8
            else:
                audio_normalized = audio
                
            wavfile.write(str(audio_path), renderer.sample_rate, audio_normalized.astype(np.float32))
            
            metadata['audio_file'] = audio_filename
            success_count += 1
            print(f"  ✅ Saved: {audio_filename}")
        else:
            print(f"  ❌ Failed to process preset")
            
        dataset_metadata.append(metadata)
        
    # Save metadata
    metadata_file = output_dir / "vst3_dataset_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(dataset_metadata, f, indent=2)
        
    print(f"\n📊 Dataset Generation Complete!")
    print(f"  ✅ Successfully processed: {success_count}/{len(preset_files)} presets")
    print(f"  📁 Audio files saved to: {output_dir}")
    print(f"  📄 Metadata saved to: {metadata_file}")
    
    success_rate = (success_count / len(preset_files)) * 100
    print(f"  📈 Success rate: {success_rate:.1f}%")

if __name__ == "__main__":
    # Test with a small number of presets first
    generate_vst3_dataset(
        preset_dir="converted_presets",
        output_dir="/Users/gjb/Datasets/serum2/vst3_renders",
        max_presets=5  # Start with 5 presets to test
    )