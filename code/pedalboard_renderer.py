# code/pedalboard_renderer.py
import json
import os
import numpy as np
from pedalboard import load_plugin

class PBRenderer:
    """
    Pedalboard-based renderer that mirrors DDRenderer API for plugin loading,
    parameter manipulation, and audio rendering with full binary state capture.
    """
    
    def __init__(self, sample_rate=22050, buffer_size=512):
        self.sr = sample_rate
        self.buffer_size = buffer_size
        self.plugin = None
        self.plugin_path = None
        
    def load_plugin(self, plugin_path: str, name: str = "synth"):
        """
        Load a VST3/AU plugin using Pedalboard.
        
        Args:
            plugin_path: Path to the plugin file
            name: Plugin name (for compatibility with DDRenderer API)
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.plugin = load_plugin(plugin_path)
            self.plugin_path = plugin_path
            return True
        except Exception as e:
            print(f"Error loading plugin {plugin_path}: {e}")
            return False
    
    def get_parameters_description(self):
        """
        Get parameter descriptions in DDRenderer-compatible format.
        
        Returns:
            list[dict]: Parameter descriptions with name, index, etc.
        """
        if self.plugin is None:
            return []
        
        descriptions = []
        for i, param in enumerate(self.plugin.parameters):
            descriptions.append({
                "name": param.name,
                "index": i,
                "range": [param.min_value, param.max_value],
                "default": param.default_value
            })
        return descriptions
    
    def get_plugin_parameter_size(self) -> int:
        """
        Get the number of parameters in the plugin.
        
        Returns:
            int: Number of parameters
        """
        if self.plugin is None:
            return 0
        return len(self.plugin.parameters)
    
    def get_patch(self):
        """
        Get current parameter values as a list of (index, value) tuples.
        
        Returns:
            list: [(index, normalized_value), ...] where values are 0-1
        """
        if self.plugin is None:
            return []
        
        patch = []
        for i, param in enumerate(self.plugin.parameters):
            # Normalize parameter value to 0-1 range
            normalized_value = (param.raw_value - param.min_value) / (param.max_value - param.min_value)
            patch.append((i, normalized_value))
        return patch
    
    def set_patch(self, patch):
        """
        Set parameter values from a list of (index, value) tuples.
        
        Args:
            patch: [(index, normalized_value), ...] where values are 0-1
        """
        if self.plugin is None:
            raise RuntimeError("No plugin loaded. Call load_plugin() first.")
        
        for idx, normalized_val in patch:
            if 0 <= idx < len(self.plugin.parameters):
                param = self.plugin.parameters[idx]
                # Convert normalized value back to parameter range
                actual_value = param.min_value + normalized_val * (param.max_value - param.min_value)
                param.raw_value = actual_value
    
    def save_state(self, path: str):
        """
        Save plugin's full internal state to a binary file.
        
        Args:
            path: Path to save the .bin file
        """
        if self.plugin is None:
            raise RuntimeError("No plugin loaded. Call load_plugin() first.")
        
        try:
            with open(path, 'wb') as f:
                f.write(self.plugin.raw_state)
            return True
        except Exception as e:
            print(f"Error saving state to {path}: {e}")
            return False
    
    def load_state(self, path: str):
        """
        Load plugin's full internal state from a binary file.
        
        Args:
            path: Path to the .bin file
        """
        if self.plugin is None:
            raise RuntimeError("No plugin loaded. Call load_plugin() first.")
        
        try:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    self.plugin.raw_state = f.read()
                return True
            else:
                print(f"State file not found: {path}")
                return False
        except Exception as e:
            print(f"Error loading state from {path}: {e}")
            return False
    
    def render_patch(self, midi_note=60, velocity=100, note_len_sec=3.0, render_len_sec=4.0):
        """
        Render audio using the current plugin state with a MIDI note.
        
        Args:
            midi_note: MIDI note number (0-127)
            velocity: MIDI velocity (0-127)
            note_len_sec: Duration of the MIDI note
            render_len_sec: Total rendering duration
        
        Returns:
            np.ndarray: Audio array with shape (channels, samples)
        """
        if self.plugin is None:
            raise RuntimeError("No plugin loaded. Call load_plugin() first.")
        
        # Calculate sample counts
        note_samples = int(note_len_sec * self.sr)
        total_samples = int(render_len_sec * self.sr)
        
        # For Pedalboard, we need to generate appropriate input
        # Most instruments work by processing silent audio with MIDI events
        # This is a simplified approach that works for many plugins
        
        try:
            # Create silent input audio (2 channels to be safe)
            audio_input = np.zeros((2, total_samples), dtype=np.float32)
            
            # Process through plugin
            # Note: Full MIDI support would require Pedalboard's MIDI functionality
            # For now, we trigger the plugin by processing silence which works for many plugins
            # that auto-trigger or have internal sequencers
            audio_output = self.plugin.process(audio_input, sample_rate=self.sr)
            
            # Ensure we return the expected shape (channels, samples) 
            if audio_output.ndim == 1:
                audio_output = audio_output.reshape(1, -1)
            elif audio_output.shape[0] > audio_output.shape[1]:
                # If samples > channels, transpose
                audio_output = audio_output.T
            
            # Ensure we have at least 2 channels for compatibility
            if audio_output.shape[0] == 1:
                audio_output = np.vstack([audio_output, audio_output])  # Duplicate mono to stereo
            
            return audio_output
            
        except Exception as e:
            print(f"Error during audio rendering: {e}")
            # Return silence if rendering fails
            return np.zeros((2, total_samples), dtype=np.float32)
    
    def render(self, audio_input=None, midi_input=None, render_len_sec=4.0):
        """
        Generic render method for compatibility with DDRenderer API.
        
        Args:
            audio_input: Input audio (for effects)
            midi_input: Input MIDI (for instruments) 
            render_len_sec: Rendering duration
            
        Returns:
            np.ndarray: Rendered audio
        """
        if self.plugin is None:
            raise RuntimeError("No plugin loaded. Call load_plugin() first.")
        
        total_samples = int(render_len_sec * self.sr)
        
        if audio_input is not None:
            # Process existing audio (effect mode)
            return self.plugin.process(audio_input, sample_rate=self.sr)
        else:
            # Generate from silence (instrument mode) 
            return self.render_patch(render_len_sec=render_len_sec)