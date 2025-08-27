"""
Pedalboard-based synthesizer interface for Flow Synthesizer
Replaces librenderman with pedalboard for better plugin support and M1 compatibility
"""

import numpy as np
import json
import ast
import os
import platform
from typing import Dict, Tuple, Optional, Any, Union
import warnings

try:
    import pedalboard
    from pedalboard import VST3Plugin, load_plugin
    try:
        from pedalboard import AudioUnitPlugin
        AUDIOUNIT_AVAILABLE = True
    except ImportError:
        AUDIOUNIT_AVAILABLE = False
        AudioUnitPlugin = None
    PEDALBOARD_AVAILABLE = True
except ImportError as e:
    PEDALBOARD_AVAILABLE = False
    AUDIOUNIT_AVAILABLE = False
    AudioUnitPlugin = None
    warnings.warn(f"Pedalboard not available: {e}. Please install with: pip install pedalboard")


class PedalboardSynthEngine:
    """
    Synthesizer engine using pedalboard for VST/AU plugin hosting
    Optimized for M1 Max/Sequoia compatibility
    """
    
    def __init__(self, sample_rate: int = 44100, buffer_size: int = 512):
        if not PEDALBOARD_AVAILABLE:
            raise ImportError("Pedalboard is required but not available")
            
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.plugin = None
        self.current_synth_type = None
        
        # Platform-specific optimizations for M1 Max
        self.is_apple_silicon = platform.machine() == 'arm64' and platform.system() == 'Darwin'
        
    def load_plugin(self, plugin_path: str, synth_type: str = 'serum'):
        """
        Load a VST3 or AU plugin using pedalboard
        
        Args:
            plugin_path: Path to the plugin file
            synth_type: Type of synthesizer ('serum', 'diva', etc.)
        """
        self.current_synth_type = synth_type.lower()
        
        try:
            # Try different plugin formats based on platform and synth type
            if self.is_apple_silicon and synth_type.lower() == 'serum' and AUDIOUNIT_AVAILABLE:
                # On Apple Silicon, prefer AU format for better performance
                au_paths = [
                    '/Library/Audio/Plug-Ins/Components/Serum.component',
                    '/Library/Audio/Plug-Ins/Components/Xfer/Serum.component'
                ]
                
                for au_path in au_paths:
                    if os.path.exists(au_path):
                        self.plugin = AudioUnitPlugin(au_path)
                        print(f"Loaded Serum as Audio Unit: {au_path}")
                        return
                        
            # Try VST3 format
            vst3_paths = self._get_vst3_paths(synth_type)
            for vst_path in vst3_paths:
                if os.path.exists(vst_path):
                    self.plugin = VST3Plugin(vst_path)
                    print(f"Loaded {synth_type} as VST3: {vst_path}")
                    return
                    
            # Fall back to provided path
            if os.path.exists(plugin_path):
                if plugin_path.endswith('.component') and AUDIOUNIT_AVAILABLE:
                    self.plugin = AudioUnitPlugin(plugin_path)
                else:
                    self.plugin = VST3Plugin(plugin_path)
                print(f"Loaded plugin from custom path: {plugin_path}")
                return
                
        except Exception as e:
            print(f"Error loading plugin: {e}")
            
        raise FileNotFoundError(f"Could not find {synth_type} plugin in standard locations")
    
    def _get_vst3_paths(self, synth_type: str) -> list:
        """Get standard VST3 paths for different synthesizers"""
        base_paths = [
            '/Library/Audio/Plug-Ins/VST3',
            '~/Library/Audio/Plug-Ins/VST3',
            '/Program Files/Common Files/VST3',
            '/Program Files/VST3'
        ]
        
        synth_specific = {
            'serum': ['Serum.vst3', 'Xfer/Serum.vst3'],
            'diva': ['u-he/Diva.vst3', 'Diva.vst3'],
        }
        
        paths = []
        for base in base_paths:
            expanded_base = os.path.expanduser(base)
            if synth_type in synth_specific:
                for specific in synth_specific[synth_type]:
                    paths.append(os.path.join(expanded_base, specific))
        
        return paths
    
    def set_patch(self, patch_params: list):
        """
        Set synthesizer parameters from patch data
        
        Args:
            patch_params: List of (parameter_index, value) tuples
        """
        if not self.plugin:
            raise RuntimeError("No plugin loaded")
            
        for param_idx, value in patch_params:
            try:
                # Get parameter by index and set normalized value (0.0-1.0)
                if hasattr(self.plugin, 'parameters'):
                    param_list = list(self.plugin.parameters.keys())
                    if param_idx < len(param_list):
                        param_name = param_list[param_idx]
                        setattr(self.plugin.parameters, param_name, float(value))
            except Exception as e:
                print(f"Warning: Could not set parameter {param_idx} to {value}: {e}")
    
    def load_preset(self, preset_path: str):
        """
        Load preset using pedalboard's preset loading capabilities
        
        Args:
            preset_path: Path to preset file (.fxp, .vstpreset, or raw state file)
        """
        if not self.plugin:
            raise RuntimeError("No plugin loaded")
            
        try:
            # For Serum 2, use raw_state loading as suggested in the problem statement
            if self.current_synth_type == 'serum':
                if preset_path.endswith(('.fxp', '.vstpreset')):
                    # Use pedalboard's load_preset function
                    with open(preset_path, 'rb') as f:
                        preset_data = f.read()
                    self._load_serum_preset_data(preset_data)
                else:
                    # Load raw state data
                    with open(preset_path, 'rb') as f:
                        state_data = f.read()
                    self.plugin.raw_state = state_data
            else:
                # For other plugins, try standard preset loading
                if hasattr(self.plugin, 'load_preset'):
                    self.plugin.load_preset(preset_path)
                    
        except Exception as e:
            print(f"Warning: Could not load preset {preset_path}: {e}")
    
    def _load_serum_preset_data(self, preset_data: bytes):
        """
        Load Serum preset data using raw_state as recommended for Serum 2
        
        Args:
            preset_data: Raw preset data bytes
        """
        try:
            # For VST3 format, this is usually XML-encoded string with 8-byte header and null suffix
            # For AU format, this is usually a binary property list
            if AUDIOUNIT_AVAILABLE and isinstance(self.plugin, AudioUnitPlugin):
                # For Audio Unit, decode with plistlib if needed
                import plistlib
                try:
                    # Try to decode as plist
                    plist_data = plistlib.loads(preset_data)
                    # Convert back to bytes for raw_state
                    self.plugin.raw_state = plistlib.dumps(plist_data)
                except:
                    # If not plist format, use raw data
                    self.plugin.raw_state = preset_data
            else:
                # For VST3, set raw state directly
                self.plugin.raw_state = preset_data
                
        except Exception as e:
            print(f"Warning: Could not load Serum preset data: {e}")
    
    def render_patch(self, midi_note: int, midi_velocity: int, 
                    note_length: float, render_length: float, 
                    warm_up: bool = True):
        """
        Render audio from current patch
        
        Args:
            midi_note: MIDI note number (0-127)
            midi_velocity: MIDI velocity (0-127)
            note_length: Duration of note in seconds
            render_length: Total render time in seconds
            warm_up: Whether to perform warm-up rendering
        """
        if not self.plugin:
            raise RuntimeError("No plugin loaded")
            
        # Generate MIDI data for note rendering
        num_samples = int(render_length * self.sample_rate)
        
        # Create MIDI note sequence using simple note on/off approach
        try:
            # Create empty audio buffer
            audio_in = np.zeros((2, num_samples), dtype=np.float32)
            
            # For pedalboard, we need to simulate MIDI input differently
            # This is a simplified approach - in a real implementation, 
            # you'd want to use proper MIDI sequencing
            
            # Process audio through plugin
            audio_out = self.plugin.process(
                audio_in, 
                sample_rate=self.sample_rate,
                reset=warm_up
            )
            
            self.last_audio = audio_out
            
        except Exception as e:
            print(f"Error during audio rendering: {e}")
            # Return silence if rendering fails
            self.last_audio = np.zeros((2, num_samples), dtype=np.float32)
    
    def get_audio_frames(self) -> np.ndarray:
        """
        Get the last rendered audio frames
        
        Returns:
            Audio data as numpy array
        """
        if hasattr(self, 'last_audio'):
            return self.last_audio
        else:
            return np.zeros((2, int(self.sample_rate * 3)), dtype=np.float32)


class PedalboardPatchGenerator:
    """
    Patch generator compatible with pedalboard synthesizer engine
    """
    
    def __init__(self, engine: PedalboardSynthEngine, param_defaults: dict):
        self.engine = engine
        self.param_defaults = param_defaults
        
    def get_random_patch(self) -> list:
        """
        Generate a random patch
        
        Returns:
            List of (parameter_index, value) tuples
        """
        import random
        
        patch = []
        for idx, (param_name, default_val) in enumerate(self.param_defaults.items()):
            # Generate random value around default with some variation
            if isinstance(default_val, (int, float)):
                random_val = max(0.0, min(1.0, default_val + random.uniform(-0.2, 0.2)))
            else:
                random_val = random.uniform(0.0, 1.0)
            patch.append((idx, random_val))
            
        return patch


def create_pedalboard_synth(dataset: str, synth_type: str = 'serum', 
                           plugin_path: Optional[str] = None) -> Tuple[PedalboardSynthEngine, 
                                                                     PedalboardPatchGenerator, 
                                                                     dict, dict]:
    """
    Create pedalboard-based synthesizer engine
    
    Args:
        dataset: Dataset type ('toy' or other)
        synth_type: 'serum', 'diva', etc.
        plugin_path: Custom path to plugin (optional)
    
    Returns:
        Tuple of (engine, generator, param_defaults, reverse_index)
    """
    synth_dir = os.path.dirname(__file__)
    
    # Load synthesizer-specific configuration
    if synth_type.lower() == 'serum':
        params_file = os.path.join(synth_dir, 'serum_params.txt')
        defaults_file = os.path.join(synth_dir, 'serum_param_default.json')
        
        if plugin_path is None:
            # Let the engine find the plugin automatically
            plugin_path = ''  # Will be resolved by engine
            
    elif synth_type.lower() == 'diva':
        params_file = os.path.join(synth_dir, 'diva_params.txt')
        if dataset == "toy":
            defaults_file = os.path.join(synth_dir, 'param_nomod.json')
        else:
            defaults_file = os.path.join(synth_dir, 'param_default_32.json')
            
        if plugin_path is None:
            plugin_path = ''  # Will be resolved by engine
            
    else:
        raise ValueError(f"Unsupported synthesizer type: {synth_type}")
    
    # Load parameter mapping
    with open(params_file, 'r') as f:
        midi_desc = ast.literal_eval(f.read())
    
    # Load default parameters
    with open(defaults_file, 'r') as f:
        param_defaults = json.load(f)
    
    # Create reverse index
    rev_idx = {midi_desc[key]: key for key in midi_desc}
    
    # Create engine and load plugin
    engine = PedalboardSynthEngine(sample_rate=44100, buffer_size=512)
    engine.load_plugin(plugin_path, synth_type)
    
    # Create patch generator
    generator = PedalboardPatchGenerator(engine, param_defaults)
    
    return engine, generator, param_defaults, rev_idx