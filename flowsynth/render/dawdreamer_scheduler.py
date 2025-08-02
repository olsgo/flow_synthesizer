"""
DawDreamer-based MIDI scheduler for polyphonic synthesis.

This module extends the existing dd_renderer.py to support:
- MIDI note scheduling with pitch bends and CC events
- BPM control and beat-quantized rendering
- Multiple synthesis modes (note-only vs full MIDI)
- Plugin registry and preset/state management
"""

import os
import json
import tempfile
import numpy as np
import dawdreamer as daw
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
from pathlib import Path

from ..transcription.midi_tools import MIDINote

logger = logging.getLogger(__name__)


class PluginRegistry:
    """Manages plugin paths and presets."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.plugins = {}
        self.presets = {}
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def load_config(self, config_path: str):
        """Load plugin configuration from YAML or JSON file."""
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    import yaml
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)
                    
            self.plugins = config.get('plugins', {})
            self.presets = config.get('presets', {})
            
        except Exception as e:
            logger.error(f"Failed to load plugin config: {e}")
    
    def register_plugin(self, name: str, path: str, presets: Optional[Dict[str, str]] = None):
        """Register a plugin with optional preset paths."""
        self.plugins[name] = path
        if presets:
            self.presets[name] = presets
    
    def get_plugin_path(self, name: str) -> Optional[str]:
        """Get plugin path by name."""
        return self.plugins.get(name)
    
    def get_preset_path(self, plugin_name: str, preset_name: str) -> Optional[str]:
        """Get preset path for a plugin."""
        plugin_presets = self.presets.get(plugin_name, {})
        return plugin_presets.get(preset_name)


class DawDreamerScheduler:
    """Enhanced MIDI scheduler built on DawDreamer."""
    
    def __init__(self, 
                 sample_rate: int = 22050, 
                 block_size: int = 512,
                 plugin_registry: Optional[PluginRegistry] = None):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.engine = daw.RenderEngine(sample_rate, block_size)
        self.plugin_registry = plugin_registry or PluginRegistry()
        
        # Current state
        self.current_plugin = None
        self.current_bpm = 120.0
        self.loaded_presets = {}
        
    def load_plugin(self, plugin_name_or_path: str, processor_name: str = "synth") -> bool:
        """
        Load a plugin by name (from registry) or direct path.
        
        Args:
            plugin_name_or_path: Plugin name in registry or direct file path
            processor_name: Name for the processor in DawDreamer
            
        Returns:
            True if successfully loaded
        """
        try:
            # Try to get from registry first
            plugin_path = self.plugin_registry.get_plugin_path(plugin_name_or_path)
            if plugin_path is None:
                # Assume it's a direct path
                plugin_path = plugin_name_or_path
                
            if not os.path.exists(plugin_path):
                logger.error(f"Plugin not found: {plugin_path}")
                return False
                
            self.current_plugin = self.engine.make_plugin_processor(processor_name, plugin_path)
            logger.info(f"Loaded plugin: {plugin_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_name_or_path}: {e}")
            return False
    
    def load_preset(self, preset_path_or_name: str, plugin_name: Optional[str] = None) -> bool:
        """
        Load a preset by path or name from registry.
        
        Args:
            preset_path_or_name: Direct path or preset name
            plugin_name: Plugin name for registry lookup (if preset_path_or_name is a name)
            
        Returns:
            True if successfully loaded
        """
        if self.current_plugin is None:
            logger.error("No plugin loaded. Call load_plugin() first.")
            return False
            
        try:
            # Try to get from registry
            preset_path = None
            if plugin_name:
                preset_path = self.plugin_registry.get_preset_path(plugin_name, preset_path_or_name)
                
            if preset_path is None:
                # Assume it's a direct path
                preset_path = preset_path_or_name
                
            if not os.path.exists(preset_path):
                logger.error(f"Preset not found: {preset_path}")
                return False
                
            # Try different preset loading methods
            if preset_path.endswith('.vst3preset'):
                return self.current_plugin.load_vst3_preset(preset_path)
            else:
                return self.current_plugin.load_state(preset_path)
                
        except Exception as e:
            logger.error(f"Failed to load preset {preset_path_or_name}: {e}")
            return False
    
    def set_bpm(self, bpm: float):
        """Set the BPM for beat-quantized rendering."""
        self.current_bpm = bpm
        self.engine.set_bpm(bpm)
    
    def render_notes_only(self, notes: List[MIDINote], duration_seconds: float) -> np.ndarray:
        """
        Render MIDI notes using add_midi_note (no pitch bends or CC).
        
        Args:
            notes: List of MIDI notes to render
            duration_seconds: Total render duration
            
        Returns:
            Audio array (channels, samples)
        """
        if self.current_plugin is None:
            raise RuntimeError("No plugin loaded. Call load_plugin() first.")
            
        # Clear existing MIDI
        self.current_plugin.clear_midi()
        
        # Add notes grouped by channel
        for note in notes:
            self.current_plugin.add_midi_note(
                note.pitch,
                note.velocity, 
                note.start_time,
                note.duration
            )
            
        # Render
        self.engine.load_graph([(self.current_plugin, [])])
        self.engine.render(duration_seconds)
        
        return self.engine.get_audio()
    
    def render_with_midi_file(self, notes: List[MIDINote], duration_seconds: float,
                             temp_dir: Optional[str] = None) -> np.ndarray:
        """
        Render MIDI notes by creating a MIDI file and loading it (supports bends/CC).
        
        Args:
            notes: List of MIDI notes with potential pitch bends
            duration_seconds: Total render duration
            temp_dir: Directory for temporary MIDI file
            
        Returns:
            Audio array (channels, samples)
        """
        if self.current_plugin is None:
            raise RuntimeError("No plugin loaded. Call load_plugin() first.")
            
        # Create temporary MIDI file
        midi_data = self._create_midi_file(notes, temp_dir)
        
        try:
            # Load MIDI file
            self.current_plugin.load_midi(midi_data)
            
            # Render
            self.engine.load_graph([(self.current_plugin, [])])
            self.engine.render(duration_seconds)
            
            return self.engine.get_audio()
            
        finally:
            # Clean up temporary file
            if os.path.exists(midi_data):
                os.unlink(midi_data)
    
    def render_beats(self, notes: List[MIDINote], duration_beats: float) -> np.ndarray:
        """
        Render with beat-based timing.
        
        Args:
            notes: List of MIDI notes
            duration_beats: Duration in beats
            
        Returns:
            Audio array (channels, samples)
        """
        if self.current_plugin is None:
            raise RuntimeError("No plugin loaded. Call load_plugin() first.")
            
        # For beat-based rendering, we'll use MIDI file approach for full feature support
        midi_data = self._create_midi_file(notes)
        
        try:
            self.current_plugin.load_midi(midi_data)
            self.engine.load_graph([(self.current_plugin, [])])
            self.engine.render(duration_beats, beats=True)
            
            return self.engine.get_audio()
            
        finally:
            if os.path.exists(midi_data):
                os.unlink(midi_data)
    
    def _create_midi_file(self, notes: List[MIDINote], temp_dir: Optional[str] = None) -> str:
        """
        Create a MIDI file from MIDINote objects.
        
        Args:
            notes: List of MIDI notes
            temp_dir: Directory for temporary file
            
        Returns:
            Path to created MIDI file
        """
        try:
            import pretty_midi
        except ImportError:
            logger.error("pretty_midi not available for MIDI file creation")
            raise RuntimeError("pretty_midi required for MIDI file rendering")
            
        # Create MIDI object
        midi = pretty_midi.PrettyMIDI(initial_tempo=self.current_bpm)
        
        # Group notes by channel
        channel_notes = {}
        for note in notes:
            if note.channel not in channel_notes:
                channel_notes[note.channel] = []
            channel_notes[note.channel].append(note)
        
        # Create instrument for each channel
        for channel, channel_note_list in channel_notes.items():
            instrument = pretty_midi.Instrument(
                program=0,  # Acoustic Grand Piano (will be overridden by synth)
                is_drum=False,
                name=f"Channel_{channel}"
            )
            
            # Add notes
            for note in channel_note_list:
                midi_note = pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=note.start_time,
                    end=note.end_time
                )
                instrument.notes.append(midi_note)
                
                # Add pitch bends if present
                for bend_time, bend_value in note.pitch_bends:
                    # Convert bend value to MIDI pitch bend range (-8192 to 8191)
                    midi_bend_value = max(-8192, min(8191, int(bend_value)))
                    pitch_bend = pretty_midi.PitchBend(midi_bend_value, bend_time)
                    instrument.pitch_bends.append(pitch_bend)
            
            midi.instruments.append(instrument)
        
        # Write to temporary file
        if temp_dir is None:
            temp_dir = tempfile.gettempdir()
            
        temp_path = os.path.join(temp_dir, f"flowsynth_temp_{os.getpid()}.mid")
        midi.write(temp_path)
        
        return temp_path
    
    def get_plugin_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded plugin."""
        if self.current_plugin is None:
            return {}
            
        try:
            info = {
                'parameter_count': self.current_plugin.get_plugin_parameter_size(),
                'parameters': self.current_plugin.get_parameters_description(),
            }
            return info
        except Exception as e:
            logger.error(f"Failed to get plugin info: {e}")
            return {}
    
    def get_parameters(self) -> List[Tuple[int, float]]:
        """Get current parameter values."""
        if self.current_plugin is None:
            return []
            
        try:
            param_count = self.current_plugin.get_plugin_parameter_size()
            return [(i, self.current_plugin.get_parameter(i)) for i in range(param_count)]
        except Exception as e:
            logger.error(f"Failed to get parameters: {e}")
            return []
    
    def set_parameters(self, parameters: List[Tuple[int, float]]):
        """Set plugin parameters."""
        if self.current_plugin is None:
            raise RuntimeError("No plugin loaded. Call load_plugin() first.")
            
        for param_idx, value in parameters:
            try:
                self.current_plugin.set_parameter(int(param_idx), float(value))
            except Exception as e:
                logger.warning(f"Failed to set parameter {param_idx}: {e}")


def create_default_registry() -> PluginRegistry:
    """Create a default plugin registry with common plugin paths."""
    registry = PluginRegistry()
    
    # Common macOS plugin paths
    macos_plugins = {
        'diva': '/Library/Audio/Plug-Ins/VST/u-he/Diva.vst',
        'diva_vst3': '/Library/Audio/Plug-Ins/VST3/Diva.vst3',
        'massive_x': '/Library/Audio/Plug-Ins/VST3/Massive X.vst3',
        'fm8': '/Library/Audio/Plug-Ins/Components/FM8.component',
        'polymax': '/Library/Audio/Plug-Ins/VST3/PolyMAX.vst3',
    }
    
    # Register plugins that exist
    for name, path in macos_plugins.items():
        if os.path.exists(path):
            registry.register_plugin(name, path)
            
    return registry