#!/usr/bin/env python3
"""
PolyMAX Dataset Generator for Flow Synthesizer Training

Generates a complete dataset from .vstpreset files following the exact format
expected by flow-synthesizer's SynthesizerDataset class.

Dataset Structure:
- /raw/*.npz files containing: 'param' (dict), 'chars' (metadata), 'audio' (numpy array)
- /mel/*.npy files containing mel spectrograms
- /mfcc/*.npy files containing MFCC features

Usage:
    python polymax_dataset_generator.py --preset-dir "/Library/Audio/Presets/UADx PolyMAX Synth" \
                                       --output-dir "/Users/gjb/Datasets/polymax" \
                                       --plugin-path "/Library/Audio/Plug-Ins/VST3/uaudio_polymax.vst3"
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add the poc-audio-pedalboard directory to path for imports
sys.path.append('/Users/gjb/Projects/poc-audio-pedalboard')

try:
    import pedalboard
    from pedalboard import VST3Plugin, load_plugin
except ImportError:
    print("Error: pedalboard not found. Please install with: pip install pedalboard")
    sys.exit(1)

class PolyMAXDatasetGenerator:
    """
    Generates flow-synthesizer compatible dataset from PolyMAX .vstpreset files.
    """
    
    def __init__(self, plugin_path: str, output_dir: str, sample_rate: int = 44100):
        self.plugin_path = plugin_path
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        
        # Create output directories
        self.raw_dir = self.output_dir / 'raw'
        self.mel_dir = self.output_dir / 'mel'
        self.mfcc_dir = self.output_dir / 'mfcc'
        self.wav_dir = self.output_dir / 'wav'
        
        for dir_path in [self.raw_dir, self.mel_dir, self.mfcc_dir, self.wav_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize plugin
        self.plugin = None
        self.initial_state = None
        self._init_plugin()
        
        # Audio generation parameters
        self.audio_duration = 4.0  # seconds
        self.midi_note = 60  # Middle C
        self.midi_velocity = 100
        
        # Spectral transform parameters (matching flow-synthesizer)
        self.mel_params = {
            'sr': self.sample_rate,  # Use actual sample rate
            'n_fft': 2048,
            'n_mels': 64,
            'hop_length': 1024,
            'fmin': 30,
            'fmax': self.sample_rate // 2  # Nyquist frequency
        }
        
        self.mfcc_params = {
            'sr': self.sample_rate,  # Use actual sample rate
            'n_mfcc': 16,
            'hop_length': 256
        }
        
        # Statistics tracking
        self.stats = {
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'errors': []
        }
    
    def _init_plugin(self):
        """Initialize the PolyMAX VST3 plugin."""
        try:
            self.plugin = load_plugin(self.plugin_path)
            print(f"Successfully loaded PolyMAX plugin from {self.plugin_path}")
            print(f"Plugin parameters: {len(self.plugin.parameters)}")
            
            # Capture initial state for resetting between presets
            self.initial_state = self.plugin.raw_state
            print(f"Captured initial plugin state: {len(self.initial_state)} bytes")
        except Exception as e:
            print(f"Error loading plugin: {e}")
            sys.exit(1)

    def _reset_plugin_state(self):
        """Reset plugin to its initial state."""
        if self.initial_state is not None:
            try:
                self.plugin.raw_state = self.initial_state
                # Small delay to let plugin settle after state reset
                import time
                time.sleep(0.1)
                print("✓ Plugin state reset to initial state")
            except Exception as e:
                print(f"Warning: Failed to reset plugin state: {e}")
        else:
            print("Warning: No initial state captured to reset to")
    
    def _load_preset(self, preset_path: str) -> bool:
        """Load a .vstpreset file using the same strategy as poc_polymax_loader.py."""
        try:
            # Strategy 1: Use pedalboard's built-in load_preset method
            if hasattr(self.plugin, 'load_preset'):
                try:
                    self.plugin.load_preset(preset_path)
                    
                    # Verify the state actually changed
                    new_state = self.plugin.raw_state
                    if new_state != self.initial_state:
                        return True
                    
                except Exception as e:
                    print(f"Warning: load_preset() failed: {e}")
            
            # Strategy 2: Load as raw state (fallback)
            try:
                with open(preset_path, 'rb') as f:
                    preset_data = f.read()
                
                # Set the raw state directly
                initial_state = self.plugin.raw_state
                self.plugin.raw_state = preset_data
                
                # Verify the state actually changed
                new_state = self.plugin.raw_state
                if new_state != initial_state:
                    return True
                    
            except Exception as e:
                print(f"Warning: Raw state loading failed: {e}")
            
            print(f"Error: All loading strategies failed for preset: {preset_path}")
            return False
                
        except Exception as e:
            print(f"Error: Failed to load preset {preset_path}: {e}")
            return False

    def _generate_audio(self) -> np.ndarray:
        """Generate audio using MIDI note sequence."""
        try:
            # MIDI configuration
            midi_note = 60  # Middle C
            midi_velocity = 100
            note_length = 3.0  # Note duration
            render_length = self.audio_duration  # Total render time
            
            # Create MIDI message sequence (using compatibility fallback since MIDIMessage not available)
            class _CompatMidi:
                def __init__(self, status, note, velocity, time):
                    self._data = bytes([status, note, velocity])
                    self.time = time
                def bytes(self):
                    return self._data
            midi_messages = [
                _CompatMidi(0x90, int(midi_note), int(midi_velocity), 0.0),
                _CompatMidi(0x80, int(midi_note), int(midi_velocity), float(note_length)),
            ]
            
            # Render audio using MIDI messages for instrument plugins
            audio_out = self.plugin(
                midi_messages,
                duration=render_length,
                sample_rate=self.sample_rate,
                num_channels=2,
                reset=False
            )
            
            # Ensure audio is in the correct format (mono)
            if audio_out.ndim == 2:
                # Convert stereo to mono by taking left channel
                if audio_out.shape[0] == 2:  # (channels, samples)
                    audio_out = audio_out[0]  # Take left channel
                elif audio_out.shape[1] == 2:  # (samples, channels)
                    audio_out = audio_out[:, 0]  # Take left channel
            
            return audio_out.astype(np.float32)
            
        except Exception as e:
            print(f"Error generating audio: {e}")
            # Return silence if generation fails
            total_samples = int(self.audio_duration * self.sample_rate)
            return np.zeros(total_samples, dtype=np.float32)
    
    def _extract_parameters(self) -> Dict[str, float]:
        """Extract current plugin parameters as a dictionary."""
        params = {}
        for param in self.plugin.parameters:
            try:
                value = param.raw_value
                param_name = param.name
                
                # Handle different parameter types
                if isinstance(value, (int, float)):
                    params[param_name] = float(value)
                elif isinstance(value, str):
                    # Try to extract numeric values from strings
                    import re
                    numeric_match = re.search(r'([+-]?\d*\.?\d+)', value)
                    if numeric_match:
                        params[param_name] = float(numeric_match.group(1))
                    else:
                        # For non-numeric strings, use hash or enumeration
                        params[param_name] = float(hash(value) % 1000) / 1000.0
                elif hasattr(value, '__bool__'):
                    # Boolean-like values
                    params[param_name] = float(bool(value))
                else:
                    # Fallback for other types
                    params[param_name] = 0.0
                    
            except Exception as e:
                # Silently handle extraction errors
                params[param_name if 'param_name' in locals() else f'param_{len(params)}'] = 0.0
        return params
    
    def _generate_metadata(self, preset_name: str) -> np.ndarray:
        """Generate metadata array for semantic characteristics.
        
        This is a simplified version - in practice, you might want to
        analyze the preset name or parameters to determine characteristics.
        """
        # Create a basic metadata array
        # Format: [characteristic_pairs, values, categories]
        # For now, we'll create a simple placeholder
        n_characteristics = 10  # Number of semantic pairs
        metadata = np.zeros((n_characteristics, 3), dtype=np.float32)
        
        # Set some basic characteristics based on preset name
        preset_lower = preset_name.lower()
        
        # Example heuristics (you can expand these)
        if 'bright' in preset_lower or 'sharp' in preset_lower:
            metadata[0, 0] = 1.0  # Bright vs Dark
        elif 'dark' in preset_lower or 'deep' in preset_lower:
            metadata[0, 1] = 1.0
        
        if 'clean' in preset_lower:
            metadata[1, 0] = 1.0  # Clean vs Dirty
        elif 'dirty' in preset_lower or 'distort' in preset_lower:
            metadata[1, 1] = 1.0
        
        if 'modern' in preset_lower:
            metadata[2, 0] = 1.0  # Modern vs Vintage
        elif 'vintage' in preset_lower or 'classic' in preset_lower:
            metadata[2, 1] = 1.0
        
        # Add more heuristics as needed...
        
        return metadata
    
    def _compute_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Compute mel spectrogram following flow-synthesizer format."""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            **self.mel_params
        )
        
        # Crop to expected size (64, 80) as in flow-synthesizer
        mel_spec = mel_spec[:64, :80]
        
        return mel_spec.astype(np.float32)
    
    def _compute_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """Compute MFCC features following flow-synthesizer format."""
        mfcc = librosa.feature.mfcc(
            y=audio,
            **self.mfcc_params
        )
        
        # Crop to expected size (16, 320) as in flow-synthesizer
        mfcc = mfcc[:16, :320]
        
        return mfcc.astype(np.float32)
    
    def process_preset(self, preset_path: str) -> bool:
        """Process a single .vstpreset file and generate dataset entry."""
        preset_name = Path(preset_path).stem
        print(f"Processing preset: {preset_name}")
        
        try:
            # Reset plugin to initial state before loading new preset
            self._reset_plugin_state()
            
            # Load preset using the same strategy as poc_polymax_loader.py
            if not self._load_preset(preset_path):
                print(f"✗ Failed to load preset: {preset_name}")
                return False
            
            # Extract parameters
            params = self._extract_parameters()
            
            # Ensure all parameters are numeric
            numeric_params = {}
            for key, val in params.items():
                if isinstance(val, (int, float)):
                    numeric_params[key] = float(val)
                else:
                    # Convert non-numeric to 0.0 as fallback
                    numeric_params[key] = 0.0
            params = numeric_params
            param_array = np.array(list(params.values()), dtype=np.float32)
            
            # Generate audio
            audio = self._generate_audio()
            
            # Check if audio was generated successfully
            if np.max(np.abs(audio)) < 1e-6:
                print(f"Warning: Very quiet audio generated for {preset_name}")
            
            # Generate metadata
            metadata = self._generate_metadata(preset_name)
            
            # Save raw data (.npz file)
            raw_file = self.raw_dir / f"{preset_name}.npz"
            np.savez_compressed(
                raw_file,
                param=param_array,
                chars=metadata,
                audio=audio
            )
            
            # Compute and save spectral transforms
            mel_spec = self._compute_mel_spectrogram(audio)
            mel_file = self.mel_dir / f"{preset_name}.npy"
            np.save(mel_file, mel_spec)
            
            mfcc = self._compute_mfcc(audio)
            mfcc_file = self.mfcc_dir / f"{preset_name}.npy"
            np.save(mfcc_file, mfcc)
            
            # Save audio as WAV file
            wav_file = self.wav_dir / f"{preset_name}.wav"
            sf.write(wav_file, audio, self.sample_rate)
            
            print(f"✓ Successfully processed {preset_name}")
            print(f"  Audio RMS: {np.sqrt(np.mean(audio**2)):.6f}")
            print(f"  Parameters: {len(param_array)}")
            print(f"  Mel shape: {mel_spec.shape}")
            print(f"  MFCC shape: {mfcc.shape}")
            
            self.stats['successful'] += 1
            return True
            
        except Exception as e:
            error_msg = f"Error processing {preset_name}: {str(e)}"
            print(f"✗ {error_msg}")
            self.stats['errors'].append(error_msg)
            self.stats['failed'] += 1
            return False
        
        finally:
            self.stats['processed'] += 1
    
    def process_preset_directory(self, preset_dir: str) -> Dict:
        """Process all .vstpreset files in a directory (recursively)."""
        preset_files = glob.glob(os.path.join(preset_dir, "**", "*.vstpreset"), recursive=True)
        
        if not preset_files:
            print(f"No .vstpreset files found in {preset_dir}")
            return self.stats
        
        print(f"Found {len(preset_files)} preset files")
        
        for preset_file in sorted(preset_files):
            self.process_preset(preset_file)
        
        return self.stats
    
    def generate_dataset_summary(self) -> Dict:
        """Generate a summary of the created dataset."""
        summary = {
            'dataset_info': {
                'total_presets': self.stats['successful'],
                'output_directory': str(self.output_dir),
                'sample_rate': self.sample_rate,
                'audio_duration': self.audio_duration
            },
            'file_counts': {
                'raw_files': len(list(self.raw_dir.glob('*.npz'))),
                'mel_files': len(list(self.mel_dir.glob('*.npy'))),
                'mfcc_files': len(list(self.mfcc_dir.glob('*.npy'))),
                'wav_files': len(list(self.wav_dir.glob('*.wav')))
            },
            'processing_stats': self.stats
        }
        
        # Save summary
        summary_file = self.output_dir / 'dataset_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary

def main():
    parser = argparse.ArgumentParser(
        description='Generate PolyMAX dataset for flow-synthesizer training'
    )
    parser.add_argument(
        '--preset-dir',
        type=str,
        default='/Library/Audio/Presets/UADx PolyMAX Synth',
        help='Directory containing .vstpreset files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/Users/gjb/Datasets/polymax',
        help='Output directory for dataset'
    )
    parser.add_argument(
        '--plugin-path',
        type=str,
        default='/Library/Audio/Plug-Ins/VST3/uaudio_polymax.vst3',
        help='Path to PolyMAX VST3 plugin'
    )
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=44100,
        help='Sample rate for audio generation'
    )
    parser.add_argument(
        '--single-preset',
        type=str,
        help='Process only a single preset file (for testing)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.plugin_path):
        print(f"Error: Plugin not found at {args.plugin_path}")
        return 1
    
    if args.single_preset:
        if not os.path.exists(args.single_preset):
            print(f"Error: Preset file not found at {args.single_preset}")
            return 1
    else:
        if not os.path.exists(args.preset_dir):
            print(f"Error: Preset directory not found at {args.preset_dir}")
            return 1
    
    # Create generator
    generator = PolyMAXDatasetGenerator(
        plugin_path=args.plugin_path,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate
    )
    
    print(f"=== PolyMAX Dataset Generator ===")
    print(f"Plugin: {args.plugin_path}")
    print(f"Output: {args.output_dir}")
    print(f"Sample rate: {args.sample_rate}")
    print()
    
    # Process presets
    if args.single_preset:
        print(f"Processing single preset: {args.single_preset}")
        generator.process_preset(args.single_preset)
    else:
        print(f"Processing preset directory: {args.preset_dir}")
        generator.process_preset_directory(args.preset_dir)
    
    # Generate summary
    summary = generator.generate_dataset_summary()
    
    print("\n=== Dataset Generation Complete ===")
    print(f"Total processed: {summary['processing_stats']['processed']}")
    print(f"Successful: {summary['processing_stats']['successful']}")
    print(f"Failed: {summary['processing_stats']['failed']}")
    
    if summary['processing_stats']['processed'] > 0:
        success_rate = summary['processing_stats']['successful'] / summary['processing_stats']['processed']
        print(f"Success rate: {success_rate:.1%}")
    
    print(f"\nDataset saved to: {args.output_dir}")
    print(f"Raw files: {summary['file_counts']['raw_files']}")
    print(f"Mel files: {summary['file_counts']['mel_files']}")
    print(f"MFCC files: {summary['file_counts']['mfcc_files']}")
    print(f"WAV files: {summary['file_counts']['wav_files']}")
    
    if summary['processing_stats']['errors']:
        print("\nErrors encountered:")
        for error in summary['processing_stats']['errors'][:5]:  # Show first 5 errors
            print(f"  - {error}")
        if len(summary['processing_stats']['errors']) > 5:
            print(f"  ... and {len(summary['processing_stats']['errors']) - 5} more")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())