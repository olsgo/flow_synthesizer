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
    PEDALBOARD_AVAILABLE = True
    # Optional AudioUnit support
    try:
        from pedalboard import AudioUnitPlugin
        AUDIOUNIT_AVAILABLE = True
    except Exception:
        AUDIOUNIT_AVAILABLE = False
        AudioUnitPlugin = None
    # Optional MIDIMessage (older pedalboard may not expose this)
    try:
        from pedalboard import MIDIMessage
        HAVE_MIDI_MESSAGE = True
    except Exception:
        HAVE_MIDI_MESSAGE = False
except ImportError as e:
    PEDALBOARD_AVAILABLE = False
    AUDIOUNIT_AVAILABLE = False
    AudioUnitPlugin = None
    HAVE_MIDI_MESSAGE = False
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
        # Store the last requested plugin path for optional resets
        self.plugin_path = plugin_path
        
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
                    # Handle Serum 2 which has multiple plugins in one file
                    if synth_type.lower() == 'serum' and 'Serum2.vst3' in plugin_path:
                        self.plugin = VST3Plugin(plugin_path, plugin_name="Serum 2")
                    else:
                        self.plugin = VST3Plugin(plugin_path)
                print(f"Loaded plugin from custom path: {plugin_path}")
                return
                
        except Exception as e:
            print(f"Error loading plugin: {e}")
            
        raise FileNotFoundError(f"Could not find {synth_type} plugin in standard locations")

    def reset_plugin(self):
        """Reload the current plugin instance to clear internal state."""
        if not self.current_synth_type:
            raise RuntimeError("Synth type not set; call load_plugin first")
        # Prefer reloading from the same resolved path when possible
        path = getattr(self, 'plugin_path', '') or ''
        self.load_plugin(path, self.current_synth_type)
    
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
            # UAD PolyMAX official VST3 filename on macOS
            'polymax': ['uaudio_polymax.vst3', 'UAD/uaudio_polymax.vst3', 'UADx PolyMAX Synth.vst3']
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
            
        # Get the actual parameter names from the plugin
        actual_param_names = list(self.plugin.parameters.keys())
        
        for i, (param_idx, value) in enumerate(patch_params):
            # Use the actual parameter name directly by index
            if param_idx < len(actual_param_names):
                param_name = actual_param_names[param_idx]
                
                try:
                    # Prefer setting normalized floats directly
                    setattr(self.plugin, param_name, float(value))
                except Exception as e:
                    # Try to coerce to allowed domain from error message
                    msg = str(e)
                    # 1) Boolean parameters
                    if 'should be a boolean' in msg:
                        try:
                            setattr(self.plugin, param_name, bool(float(value) >= 0.5))
                            continue
                        except Exception:
                            pass
                    # 2) Enum choices
                    if 'not in list of valid values' in msg:
                        try:
                            import re
                            m = re.search(r"\[(.*)\]", msg)
                            if m:
                                choices_str = m.group(1)
                                # split on comma while stripping quotes and spaces
                                raw = [c.strip() for c in choices_str.split(',')]
                                choices = [c.strip("'\"") for c in raw]
                                if choices:
                                    idx = int(round(max(0.0, min(1.0, float(value))) * (len(choices) - 1)))
                                    setattr(self.plugin, param_name, choices[idx])
                                    continue
                        except Exception:
                            pass
                    # 3) Numeric range parsing (e.g., [100.0Hz, 10900.0Hz])
                    if 'out of range [' in msg:
                        try:
                            import re
                            m = re.search(r"out of range \[(.*),(.*)\]", msg)
                            if m:
                                def _to_num(s):
                                    s = s.strip()
                                    # strip units like Hz, k, ct
                                    s = s.replace('Hz','').replace('ct','')
                                    s = s.replace('k','000')
                                    try:
                                        return float(s)
                                    except Exception:
                                        return None
                                lo = _to_num(m.group(1))
                                hi = _to_num(m.group(2))
                                if lo is not None and hi is not None and hi > lo:
                                    v = lo + (hi - lo) * max(0.0, min(1.0, float(value)))
                                    setattr(self.plugin, param_name, v)
                                    continue
                        except Exception:
                            pass
                    # 4) Last resort: set raw_value if available (normalized 0..1)
                    try:
                        p = self.plugin.parameters[param_name]
                        p.raw_value = float(value)
                        continue
                    except Exception:
                        pass
                    print(f"Error setting parameter {param_idx} ({param_name}): {e}")
            else:
                print(f"Warning: Parameter index {param_idx} out of range (max: {len(actual_param_names)-1})")
    
    def load_preset(self, preset_path: str):
        """
        Load preset using pedalboard's preset loading capabilities
        
        Args:
            preset_path: Path to preset file (.fxp, .vstpreset, or raw state file)
        """
        if not self.plugin:
            raise RuntimeError("No plugin loaded")
            
        try:
            # First, prefer native plugin preset loader if available
            if hasattr(self.plugin, 'load_preset'):
                try:
                    self.plugin.load_preset(preset_path)
                    return
                except Exception:
                    pass

            # Serum-specific fallback: set raw_state bytes (works for some state formats)
            if self.current_synth_type == 'serum':
                with open(preset_path, 'rb') as f:
                    preset_data = f.read()
                self._load_serum_preset_data(preset_data)
                return

            # Generic fallback: try raw state
            with open(preset_path, 'rb') as f:
                state_data = f.read()
            self.plugin.raw_state = state_data

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
                    warm_up: bool = False):
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
            
        try:
            # Calculate timing for MIDI messages
            note_off_time = note_length
            
            # Create MIDI message sequence
            if HAVE_MIDI_MESSAGE:
                midi_messages = [
                    MIDIMessage.note_on(note=int(midi_note), velocity=int(midi_velocity), time=0.0),
                    MIDIMessage.note_off(note=int(midi_note), velocity=int(midi_velocity), time=float(note_off_time)),
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
                    _CompatMidi(0x80, int(midi_note), int(midi_velocity), float(note_off_time)),
                ]
            
            # Render audio using MIDI messages for instrument plugins
            audio_out = self.plugin(
                midi_messages,
                duration=render_length,
                sample_rate=self.sample_rate,
                num_channels=2,
                reset=bool(warm_up)
            )
            
            # Ensure audio is in the correct format (channels, samples)
            if audio_out.ndim == 1:
                # Mono to stereo
                audio_out = np.stack([audio_out, audio_out])
            elif audio_out.shape[0] > audio_out.shape[1]:
                # Transpose if needed (samples, channels) -> (channels, samples)
                audio_out = audio_out.T
                
            self.last_audio = audio_out
            
        except Exception as e:
            print(f"Error during audio rendering: {e}")
            # Return silence if rendering fails
            num_samples = int(render_length * self.sample_rate)
            self.last_audio = np.zeros((2, num_samples), dtype=np.float32)

    def render_midi(self, midi_messages, render_length: float, warm_up: bool = False):
        """
        Render audio from an arbitrary MIDI message sequence.

        Args:
            midi_messages: Iterable of pedalboard.MIDIMessage or objects with .bytes() and .time
            render_length: Total render time in seconds
            warm_up: Whether to reset the plugin before rendering
        """
        if not self.plugin:
            raise RuntimeError("No plugin loaded")
        try:
            audio_out = self.plugin(
                midi_messages,
                duration=render_length,
                sample_rate=self.sample_rate,
                num_channels=2,
                reset=bool(warm_up)
            )
            if audio_out.ndim == 1:
                audio_out = np.stack([audio_out, audio_out])
            elif audio_out.shape[0] > audio_out.shape[1]:
                audio_out = audio_out.T
            self.last_audio = audio_out
        except Exception as e:
            print(f"Error during MIDI rendering: {e}")
            num_samples = int(render_length * self.sample_rate)
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
            
    elif synth_type.lower() == 'polymax':
        # For PolyMAX, use the parameter mapping that matches the training dataset
        params_file = os.path.join(synth_dir, 'polymax_params.txt')
        defaults_file = None  # Will be discovered from plugin

        # Default VST3 location from assignment/context
        if plugin_path is None:
            plugin_path = '/Library/Audio/Plug-Ins/VST3/uaudio_polymax.vst3'
            if not os.path.exists(plugin_path):
                plugin_path = ''
        
    else:
        raise ValueError(f"Unsupported synthesizer type: {synth_type}")
    
    # Create engine and load plugin
    engine = PedalboardSynthEngine(sample_rate=44100, buffer_size=512)
    engine.load_plugin(plugin_path, synth_type)

    # Build parameter mapping/defaults
    if synth_type.lower() == 'polymax':
        # Load parameter mapping from file (matches training dataset)
        with open(params_file, 'r') as f:
            midi_desc = ast.literal_eval(f.read())
        
        # Create mapping from dataset parameter names to synthesizer parameter names
        # The dataset uses names from params_schema.json, but synthesizer expects generic names
        import json
        schema_path = os.path.join(os.path.dirname(params_file), '..', '..', 'params_schema.json')
        if os.path.exists(schema_path):
            with open(schema_path, 'r') as f:
                schema = json.load(f)
            dataset_param_names = schema.get('parameter_order', [])
        else:
            # Fallback to generic names if schema not found
            dataset_param_names = [f'param_{i}' for i in range(66)]
        
        # Create mapping: dataset_name -> generic_name (param_0, param_1, etc.)
        dataset_to_generic = {}
        for i, dataset_name in enumerate(dataset_param_names):
            dataset_to_generic[dataset_name] = f'param_{i}'
        
        # Update midi_desc to use dataset parameter names as keys
        new_midi_desc = {}
        for idx, synth_name in midi_desc.items():
            if idx < len(dataset_param_names):
                dataset_name = dataset_param_names[idx]
                new_midi_desc[dataset_name] = idx
        midi_desc = new_midi_desc
        
        # Discover default values from the loaded plugin
        param_defaults = {}
        plugin_params = list(engine.plugin.parameters.keys())
        for dataset_name, idx in midi_desc.items():
            try:
                # Use dataset parameter name as key
                param_defaults[dataset_name] = 0.5  # Default value
            except Exception:
                param_defaults[dataset_name] = 0.5
    else:
        # Load parameter mapping from files for other synths
        with open(params_file, 'r') as f:
            midi_desc = ast.literal_eval(f.read())
        with open(defaults_file, 'r') as f:
            param_defaults = json.load(f)

    # For pedalboard usage, return name->index mapping
    rev_idx = midi_desc
    
    # Create patch generator
    generator = PedalboardPatchGenerator(engine, param_defaults)
    
    return engine, generator, param_defaults, rev_idx
