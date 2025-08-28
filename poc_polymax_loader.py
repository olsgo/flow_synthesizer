#!/usr/bin/env python3
"""
PolyMAX VST3 Preset Loader for Flow Synthesizer Training

This module provides functionality to load UAD PolyMAX VST3 .vstpreset files,
extract their state, and save binary .npz files for flow-synth training.

Based on patterns from poc_inspired_serum_loader.py but adapted for PolyMAX
and .vstpreset file format.
"""

import os
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import argparse
from dataclasses import dataclass
import logging
from pedalboard import load_plugin, VST3Plugin
from pedalboard.io import AudioFile
from mido import Message

# Try to import MIDI message functionality
try:
    from pedalboard import MIDIMessage
    HAVE_MIDI_MESSAGE = True
except ImportError:
    HAVE_MIDI_MESSAGE = False
    MIDIMessage = None
import time

# Import helper functions
from polymax_helpers import (
    create_audio_filename,
    create_state_filename,
    create_safe_filename,
    find_vstpreset_files,
    estimate_processing_time
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PolyMAXConfig:
    """Configuration for PolyMAX preset loading and rendering."""
    vst3_path: str = '/Library/Audio/Plug-Ins/VST3/uaudio_polymax.vst3'
    presets_dir: str = '/Library/Audio/Presets/UADx PolyMAX Synth'
    output_dir: str = '/Users/gjb/Datasets/polymax/render'
    sample_rate: int = 44100
    num_channels: int = 2
    render_duration: float = 4.0
    midi_note: int = 60  # Middle C
    midi_velocity: int = 100
    note_duration: float = 3.0


class VST3StateManager:
    """Manages VST3 plugin state capture and restoration."""
    
    def __init__(self, plugin: VST3Plugin):
        self.plugin = plugin
        self.initial_state = None
        
    def capture_initial_state(self) -> bytes:
        """Capture the initial raw state of the plugin."""
        self.initial_state = self.plugin.raw_state
        logger.info(f"Captured initial state: {len(self.initial_state)} bytes")
        return self.initial_state
        
    def get_current_state(self) -> bytes:
        """Get the current raw state of the plugin."""
        return self.plugin.raw_state
        
    def restore_initial_state(self):
        """Restore the plugin to its initial state."""
        if self.initial_state is not None:
            self.plugin.raw_state = self.initial_state
            logger.debug("Restored plugin to initial state")
        else:
            logger.warning("No initial state captured to restore")
            
    def set_state(self, state: bytes):
        """Set the plugin to a specific state."""
        self.plugin.raw_state = state
        logger.debug(f"Set plugin state: {len(state)} bytes")


class PluginParameterTracker:
    """Tracks parameter changes in VST3 plugins."""
    
    def __init__(self, plugin: VST3Plugin):
        self.plugin = plugin
        self.initial_params = {}
        
    def capture_initial_parameters(self) -> Dict[str, float]:
        """Capture initial parameter values."""
        self.initial_params = {
            key: param.raw_value 
            for key, param in self.plugin.parameters.items()
        }
        logger.info(f"Captured {len(self.initial_params)} initial parameters")
        return self.initial_params.copy()
        
    def get_current_parameters(self) -> Dict[str, float]:
        """Get current parameter values."""
        return {
            key: param.raw_value 
            for key, param in self.plugin.parameters.items()
        }
        
    def get_parameter_changes(self) -> Dict[str, Dict[str, float]]:
        """Get parameters that have changed from initial values."""
        current_params = self.get_current_parameters()
        changes = {}
        
        for key in self.initial_params:
            if key in current_params:
                initial_val = self.initial_params[key]
                current_val = current_params[key]
                if abs(initial_val - current_val) > 1e-6:  # Small tolerance for float comparison
                    changes[key] = {
                        'initial': initial_val,
                        'current': current_val,
                        'delta': current_val - initial_val
                    }
                    
        return changes





class PolyMAXPresetLoader:
    """Main class for loading PolyMAX presets and generating training data."""
    
    def __init__(self, config: PolyMAXConfig):
        self.config = config
        self.plugin = None
        self.state_manager = None
        self.param_tracker = None
        
    def initialize_plugin(self):
        """Initialize the PolyMAX VST3 plugin."""
        try:
            logger.info(f"Loading PolyMAX plugin from {self.config.vst3_path}")
            self.plugin = load_plugin(self.config.vst3_path)
            
            if not self.plugin.is_instrument:
                logger.warning("PolyMAX plugin is not detected as an instrument")
                
            logger.info(f"Plugin loaded: {self.plugin.name}")
            logger.info(f"Parameters available: {len(self.plugin.parameters)}")
            
            # Initialize managers
            self.state_manager = VST3StateManager(self.plugin)
            self.param_tracker = PluginParameterTracker(self.plugin)
            
            # Capture initial state
            self.state_manager.capture_initial_state()
            self.param_tracker.capture_initial_parameters()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize PolyMAX plugin: {e}")
            return False
            
    def load_preset(self, preset_path: str) -> bool:
        """Load a .vstpreset file into the PolyMAX plugin using pedalboard's native support."""
        try:
            logger.info(f"Loading preset: {preset_path}")
            
            # First, restore to initial state
            self.state_manager.restore_initial_state()
            
            # Strategy 1: Use pedalboard's built-in load_preset method
            if hasattr(self.plugin, 'load_preset'):
                try:
                    self.plugin.load_preset(preset_path)
                    
                    # Verify the state actually changed
                    new_state = self.plugin.raw_state
                    if new_state != self.state_manager.initial_state:
                        logger.info(f"Successfully loaded preset via load_preset(): {Path(preset_path).stem}")
                        return True
                    else:
                        logger.debug("State did not change after load_preset()")
                        
                except Exception as e:
                    logger.debug(f"load_preset() failed: {e}")
            
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
                    logger.info(f"Successfully loaded preset via raw_state: {Path(preset_path).stem}")
                    return True
                else:
                    logger.debug("State did not change after raw_state loading")
                    
            except Exception as e:
                logger.debug(f"Raw state loading failed: {e}")
            
            logger.error(f"All loading strategies failed for preset: {preset_path}")
            return False
                
        except Exception as e:
            logger.error(f"Failed to load preset {preset_path}: {e}")
            return False
            

            
    def render_audio(self, preset_name: str) -> Optional[np.ndarray]:
        """Render audio with the current preset using MIDI input."""
        try:
            # Use single note for cleaner audio generation
            midi_note = self.config.midi_note
            midi_velocity = self.config.midi_velocity
            note_length = self.config.note_duration
            render_length = self.config.render_duration
            
            logger.debug(f"Rendering MIDI note {midi_note} (vel={midi_velocity}) for {note_length}s over {render_length}s")
            
            # Create MIDI message sequence
            if HAVE_MIDI_MESSAGE:
                midi_messages = [
                    MIDIMessage.note_on(note=int(midi_note), velocity=int(midi_velocity), time=0.0),
                    MIDIMessage.note_off(note=int(midi_note), velocity=int(midi_velocity), time=float(note_length)),
                ]
            else:
                # Compatibility fallback: provide minimal objects exposing .bytes()
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
                sample_rate=self.config.sample_rate,
                num_channels=self.config.num_channels,
                reset=False
            )
            
            # Ensure audio is in the correct format (samples, channels)
            if audio_out.ndim == 1:
                # Mono to stereo
                audio_out = np.stack([audio_out, audio_out], axis=1)
            elif audio_out.shape[0] < audio_out.shape[1]:
                # Transpose if needed (channels, samples) -> (samples, channels)
                audio_out = audio_out.T
                
            logger.info(f"Rendered {render_length}s audio for preset: {preset_name}")
            logger.debug(f"Audio stats - Max: {np.max(audio_out):.3f}, Min: {np.min(audio_out):.3f}, RMS: {np.sqrt(np.mean(audio_out**2)):.3f}")
            
            return audio_out
            
        except Exception as e:
            logger.error(f"Failed to render audio for preset {preset_name}: {e}")
            return None
            
    def save_training_data(self, preset_name: str, audio: np.ndarray, state: bytes, 
                          preset_info: Optional[Dict] = None) -> bool:
        """Save audio and state data for flow-synth training."""
        try:
            # Create safe filename using helper function
            safe_name = create_safe_filename(preset_name)
            
            # Ensure output directory exists
            os.makedirs(self.config.output_dir, exist_ok=True)
            
            # Save audio file
            audio_path = os.path.join(self.config.output_dir, f"{safe_name}.wav")
            with AudioFile(audio_path, "w", self.config.sample_rate, self.config.num_channels) as f:
                f.write(audio)
                
            # Save state as compressed .npz file for flow-synth training
            state_path = os.path.join(self.config.output_dir, f"{safe_name}.npz")
            
            # Prepare data for .npz file - include additional metadata for training
            npz_data = {
                'vst_state': state,  # Raw VST3 state bytes
                'preset_name': preset_name,
                'sample_rate': self.config.sample_rate,
                'render_duration': self.config.render_duration,
                'plugin_name': 'UAD PolyMAX',
                'plugin_format': 'VST3',
                'audio_shape': audio.shape,
                'timestamp': np.datetime64('now')
            }
            
            # Add preset information if available
            if preset_info:
                npz_data.update({
                    'preset_info': preset_info,
                    'parameter_count': len(preset_info.get('parameters', {})),
                    'xml_content_available': 'xml_content' in preset_info
                })
                
            # Save compressed .npz file
            np.savez_compressed(state_path, **npz_data)
            
            # Verify file sizes
            audio_size = os.path.getsize(audio_path) / 1024  # KB
            state_size = os.path.getsize(state_path) / 1024  # KB
            
            logger.info(f"Saved training data for {preset_name}:")
            logger.info(f"  Audio: {audio_path} ({audio_size:.1f} KB)")
            logger.info(f"  State: {state_path} ({state_size:.1f} KB)")
            
            # Log additional metadata
            if preset_info:
                param_count = len(preset_info.get('parameters', {}))
                logger.debug(f"  Parameters: {param_count}")
                logger.debug(f"  XML available: {'xml_content' in preset_info}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save training data for {preset_name}: {e}")
            return False
            
    def process_preset(self, preset_path: str) -> bool:
        """Process a single preset: load, render, and save."""
        preset_name = Path(preset_path).stem
        
        try:
            # Create basic preset info
            preset_info = {
                'file_path': preset_path,
                'preset_name': preset_name,
                'file_size': Path(preset_path).stat().st_size
            }
            
            # Load the preset
            if not self.load_preset(preset_path):
                return False
                
            # Small delay to let plugin settle
            time.sleep(0.1)
            
            # Capture the current state
            current_state = self.state_manager.get_current_state()
            
            # Render audio
            audio = self.render_audio(preset_name)
            if audio is None:
                return False
                
            # Save training data with preset info
            return self.save_training_data(preset_name, audio, current_state, preset_info)
            
        except Exception as e:
            logger.error(f"Failed to process preset {preset_path}: {e}")
            return False
            
    def batch_process_presets(self, preset_dir: Optional[str] = None) -> Dict[str, bool]:
        """Process all .vstpreset files in a directory."""
        if preset_dir is None:
            preset_dir = self.config.presets_dir
            
        results = {}
        
        try:
            # Use helper function to find preset files
            preset_files = find_vstpreset_files(preset_dir, recursive=True)
            
            if not preset_files:
                logger.warning(f"No .vstpreset files found in {preset_dir}")
                return results
                
            logger.info(f"Found {len(preset_files)} preset files in {preset_dir}")
            
            # Estimate processing time
            time_estimate = estimate_processing_time(len(preset_files), self.config.render_duration)
            logger.info(f"Estimated processing time: {time_estimate['total_time_minutes']:.1f} minutes")
            
            # Process each preset
            successful_count = 0
            for i, preset_path in enumerate(preset_files, 1):
                preset_name = Path(preset_path).name
                logger.info(f"Processing preset {i}/{len(preset_files)}: {preset_name}")
                
                try:
                    success = self.process_preset(preset_path)
                    results[preset_path] = success
                    
                    if success:
                        successful_count += 1
                        logger.info(f"✓ Successfully processed: {preset_name}")
                    else:
                        logger.warning(f"✗ Failed to process: {preset_name}")
                        
                except Exception as e:
                    logger.error(f"✗ Error processing {preset_name}: {e}")
                    results[preset_path] = False
                    
                # Restore to initial state between presets
                self.state_manager.restore_initial_state()
                
                # Progress update every 10 presets
                if i % 10 == 0:
                    progress = (i / len(preset_files)) * 100
                    logger.info(f"Progress: {progress:.1f}% ({successful_count}/{i} successful)")
                
            logger.info(f"Batch processing complete: {successful_count}/{len(preset_files)} successful")
            
            # Summary statistics
            success_rate = (successful_count / len(preset_files)) * 100 if preset_files else 0
            logger.info(f"Success rate: {success_rate:.1f}%")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to batch process presets: {e}")
            return results


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="PolyMAX Preset Loader for Flow Synthesizer Training")
    parser.add_argument('--vst3-path', default='/Library/Audio/Plug-Ins/VST3/uaudio_polymax.vst3',
                       help='Path to PolyMAX VST3 plugin')
    parser.add_argument('--presets-dir', default='/Library/Audio/Presets/UADx PolyMAX Synth',
                       help='Directory containing .vstpreset files')
    parser.add_argument('--output-dir', default='/Users/gjb/Datasets/polymax/render',
                       help='Output directory for rendered audio and state files')
    parser.add_argument('--preset-file', help='Process a single preset file')
    parser.add_argument('--sample-rate', type=int, default=44100, help='Audio sample rate')
    parser.add_argument('--duration', type=float, default=4.0, help='Render duration in seconds')
    parser.add_argument('--midi-note', type=int, default=60, help='MIDI note to play')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Create configuration
    config = PolyMAXConfig(
        vst3_path=args.vst3_path,
        presets_dir=args.presets_dir,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        render_duration=args.duration,
        midi_note=args.midi_note
    )
    
    # Initialize loader
    loader = PolyMAXPresetLoader(config)
    
    if not loader.initialize_plugin():
        logger.error("Failed to initialize plugin")
        return 1
        
    try:
        if args.preset_file:
            # Process single preset
            success = loader.process_preset(args.preset_file)
            return 0 if success else 1
        else:
            # Batch process all presets
            results = loader.batch_process_presets()
            failed = sum(1 for success in results.values() if not success)
            return 0 if failed == 0 else 1
            
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == '__main__':
    exit(main())