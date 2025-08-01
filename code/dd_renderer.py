# code/dd_renderer.py
import json
import os
from pathlib import Path
import dawdreamer as daw

# Try relative import first, fall back to absolute import
try:
    from .plugin_config import get_config
except ImportError:
    from plugin_config import get_config

class DDRenderer:
    def __init__(self, sample_rate=22050, block_size=512):
        self.sr = sample_rate
        self.engine = daw.RenderEngine(self.sr, block_size)
        self.inst = None

    @classmethod
    def from_config(cls):
        """Create a DDRenderer with settings from the configuration file."""
        config = get_config()
        settings = config.get_audio_settings()
        return cls(sample_rate=settings["sample_rate"], block_size=settings["block_size"])

    def load_plugin_by_name(self, plugin_name: str, name: str = "synth") -> bool:
        """
        Load a plugin by its configured name instead of explicit path.
        
        Args:
            plugin_name: Name of the plugin as configured in plugin_config.yml
            name: Internal name for the plugin processor
            
        Returns:
            bool: True if plugin loaded successfully, False otherwise
        """
        config = get_config()
        plugin_path = config.get_preferred_plugin_path(plugin_name)
        
        if not plugin_path:
            print(f"Warning: No configured path found for plugin '{plugin_name}'")
            print(f"Available configured plugins: {config.list_configured_plugins()}")
            return False
        
        print(f"Loading {plugin_name} from: {plugin_path}")
        return self.load_plugin(plugin_path, name)

    def load_plugin(self, plugin_path: str, name: str = "synth"):
        """
        Load a plugin from the given path.
        
        Args:
            plugin_path: Path to the plugin file (.vst3, .component, etc.)
            name: Internal name for the plugin processor
            
        Returns:
            bool: True if plugin loaded successfully, False otherwise
        """
        try:
            self.inst = self.engine.make_plugin_processor(name, plugin_path)
            # Test if plugin was actually loaded by trying to access its parameters
            if self.inst is None:
                return False
            # Try to get parameter count to verify plugin is functional
            _ = self.inst.get_plugin_parameter_size()
            return True
        except Exception as e:
            print(f"Warning: Failed to load plugin '{plugin_path}': {e}")
            self.inst = None
            return False

    def is_plugin_loaded(self) -> bool:
        """Check if a plugin is currently loaded and functional."""
        return self.inst is not None

    def get_plugin_info(self) -> dict:
        """Get information about the currently loaded plugin."""
        if not self.is_plugin_loaded():
            return {"loaded": False, "error": "No plugin loaded"}
        
        try:
            param_count = self.get_plugin_parameter_size()
            param_desc = self.get_parameters_description()
            return {
                "loaded": True,
                "parameter_count": param_count,
                "parameter_names": [p.get("name", f"param_{i}") for i, p in enumerate(param_desc[:10])],  # First 10 only
                "total_parameters": len(param_desc)
            }
        except Exception as e:
            return {"loaded": False, "error": f"Plugin loaded but not functional: {e}"}

    @staticmethod
    def validate_plugin_path(plugin_path: str) -> dict:
        """
        Validate if a plugin path exists and appears to be a valid plugin.
        
        Returns:
            dict: {"valid": bool, "path": str, "type": str, "message": str}
        """
        if not plugin_path:
            return {"valid": False, "path": "", "type": "unknown", "message": "Empty path provided"}
        
        path = Path(plugin_path).expanduser().resolve()
        
        if not path.exists():
            return {"valid": False, "path": str(path), "type": "unknown", "message": "Path does not exist"}
        
        # Determine plugin type based on extension/structure
        plugin_type = "unknown"
        if path.suffix.lower() == ".vst3":
            plugin_type = "VST3"
        elif path.suffix.lower() == ".component":
            plugin_type = "Audio Unit"
        elif path.suffix.lower() in [".vst", ".dll"]:
            plugin_type = "VST2"
        elif path.is_dir() and path.suffix.lower() == ".vst":
            plugin_type = "VST2 Bundle"
        elif path.is_dir() and path.suffix.lower() == ".component":
            plugin_type = "Audio Unit Bundle"
        
        return {
            "valid": True, 
            "path": str(path), 
            "type": plugin_type,
            "message": f"Valid {plugin_type} plugin found"
        }

    @staticmethod
    def find_plugin_paths(plugin_name: str) -> list:
        """
        Search for plugin paths on the system for a given plugin name.
        
        Args:
            plugin_name: Name of the plugin to search for
            
        Returns:
            list: List of potential plugin paths found
        """
        search_paths = []
        
        # macOS paths
        mac_paths = [
            f"/Library/Audio/Plug-Ins/VST3/{plugin_name}.vst3",
            f"/Library/Audio/Plug-Ins/Components/{plugin_name}.component",
            f"/Library/Audio/Plug-Ins/VST/{plugin_name}.vst",
            f"~/Library/Audio/Plug-Ins/VST3/{plugin_name}.vst3",
            f"~/Library/Audio/Plug-Ins/Components/{plugin_name}.component",
            f"~/Library/Audio/Plug-Ins/VST/{plugin_name}.vst",
        ]
        
        # Windows paths
        win_paths = [
            f"C:/Program Files/Common Files/VST3/{plugin_name}.vst3",
            f"C:/Program Files (x86)/Common Files/VST3/{plugin_name}.vst3",
            f"C:/Program Files/VSTPlugins/{plugin_name}.dll",
            f"C:/Program Files (x86)/VSTPlugins/{plugin_name}.dll",
        ]
        
        # Linux paths
        linux_paths = [
            f"~/.vst3/{plugin_name}.vst3",
            f"/usr/lib/vst3/{plugin_name}.vst3",
            f"~/.vst/{plugin_name}.so",
            f"/usr/lib/vst/{plugin_name}.so",
        ]
        
        # Check all paths based on the current OS
        import platform
        if platform.system() == "Darwin":
            search_paths.extend(mac_paths)
        elif platform.system() == "Windows":
            search_paths.extend(win_paths)
        else:
            search_paths.extend(linux_paths)
        
        # Also add some cross-platform searches
        search_paths.extend(mac_paths + win_paths + linux_paths)
        
        found_paths = []
        for path_str in search_paths:
            path = Path(path_str).expanduser()
            if path.exists():
                found_paths.append(str(path))
        
        return found_paths

    # --- parameter I/O ---
    def get_parameters_description(self):  # list[dict]
        if self.inst is None:
            return []
        return self.inst.get_parameters_description()

    def get_plugin_parameter_size(self) -> int:
        if self.inst is None:
            return 0
        return self.inst.get_plugin_parameter_size()

    def get_patch(self):
        if self.inst is None:
            return []
        n = self.get_plugin_parameter_size()
        return [(i, self.inst.get_parameter(i)) for i in range(n)]

    def set_patch(self, patch):  # [(index, value_0_1), ...]
        if self.inst is None:
            raise RuntimeError("No plugin loaded. Call load_plugin() first.")
        for idx, val in patch:
            self.inst.set_parameter(int(idx), float(val))

    # --- rendering ---
    def render_patch(self, midi_note=60, velocity=100, note_len_sec=3.0, render_len_sec=4.0):
        if self.inst is None:
            raise RuntimeError("No plugin loaded. Call load_plugin() first.")
        self.inst.clear_midi()
        self.inst.add_midi_note(midi_note, velocity, 0.0, note_len_sec)
        self.engine.load_graph([(self.inst, [])])
        self.engine.render(render_len_sec)
        return self.engine.get_audio()  # (channels, samples)

    # optional helpers for presets/state
    def load_vst3_preset(self, preset_path: str):
        if self.inst is None:
            raise RuntimeError("No plugin loaded. Call load_plugin() first.")
        return self.inst.load_vst3_preset(preset_path)

    def load_state(self, path: str):
        if self.inst is None:
            raise RuntimeError("No plugin loaded. Call load_plugin() first.")
        return self.inst.load_state(path)