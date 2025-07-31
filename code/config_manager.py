"""
Configuration manager for Flow Synthesizer
Handles synthesizer selection, plugin paths, and other settings
"""
import json
import platform
import os
from typing import Dict, Any, Optional


class ConfigManager:
    """Manages configuration for the Flow Synthesizer"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self._config = self._load_config()
        self._platform = self._get_platform()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Configuration file {self.config_path} not found. Using defaults.")
            return self._get_default_config()
        except json.JSONDecodeError as e:
            print(f"Error parsing configuration file: {e}. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file is missing or invalid"""
        return {
            "synthesizer": {
                "default_type": "diva",
                "plugin_paths": {
                    "diva": {
                        "mac": "/Library/Audio/Plug-Ins/VST/u-he/Diva.vst",
                        "linux": "synth/diva.64.so",
                        "windows": "synth/diva.dll"
                    },
                    "massive_x": {
                        "mac": "/Library/Audio/Plug-Ins/VST/Native Instruments/Massive X.vst",
                        "linux": "synth/massive_x.64.so",
                        "windows": "synth/massive_x.dll"
                    }
                }
            },
            "audio": {
                "sample_rate": 44100,
                "buffer_size": 512
            },
            "osc": {
                "default_in_port": 1232,
                "default_out_port": 1233,
                "default_ip": "127.0.0.1"
            }
        }
    
    def _get_platform(self) -> str:
        """Get current platform for plugin path selection"""
        system = platform.system().lower()
        if system == 'darwin':
            return 'mac'
        elif system == 'linux':
            return 'linux'
        elif system == 'windows':
            return 'windows'
        else:
            print(f"Unknown platform: {system}. Defaulting to linux.")
            return 'linux'
    
    def get_synthesizer_type(self) -> str:
        """Get the default synthesizer type"""
        return self._config.get("synthesizer", {}).get("default_type", "diva")
    
    def get_plugin_path(self, synth_type: Optional[str] = None, platform: Optional[str] = None) -> str:
        """
        Get plugin path for specified synthesizer and platform
        
        Args:
            synth_type: Synthesizer type ('diva' or 'massive_x'). If None, uses default.
            platform: Target platform ('mac', 'linux', 'windows'). If None, uses current.
        
        Returns:
            Path to plugin file
        """
        if synth_type is None:
            synth_type = self.get_synthesizer_type()
        
        if platform is None:
            platform = self._platform
        
        plugin_paths = self._config.get("synthesizer", {}).get("plugin_paths", {})
        synth_paths = plugin_paths.get(synth_type, {})
        
        if platform not in synth_paths:
            # Fallback to linux path if platform not found
            platform = 'linux'
        
        path = synth_paths.get(platform, f"synth/{synth_type}.64.so")
        
        # Check if file exists, if not warn user
        if not os.path.exists(path):
            print(f"Warning: Plugin file not found at {path}")
            print(f"Please ensure {synth_type} is installed and update the path in {self.config_path}")
        
        return path
    
    def get_parameter_file(self, synth_type: Optional[str] = None, dataset: str = "default") -> str:
        """
        Get parameter file path for specified synthesizer and dataset
        
        Args:
            synth_type: Synthesizer type. If None, uses default.
            dataset: Dataset type ('default', 'toy', etc.)
        
        Returns:
            Path to parameter file
        """
        if synth_type is None:
            synth_type = self.get_synthesizer_type()
        
        param_sets = self._config.get("synthesizer", {}).get("parameter_sets", {})
        synth_params = param_sets.get(synth_type, {})
        
        return synth_params.get(dataset, synth_params.get("default", f"synth/{synth_type}_default.json"))
    
    def get_audio_config(self) -> Dict[str, int]:
        """Get audio configuration"""
        audio_config = self._config.get("audio", {})
        return {
            "sample_rate": audio_config.get("sample_rate", 44100),
            "buffer_size": audio_config.get("buffer_size", 512)
        }
    
    def get_osc_config(self) -> Dict[str, Any]:
        """Get OSC configuration"""
        osc_config = self._config.get("osc", {})
        return {
            "default_in_port": osc_config.get("default_in_port", 1232),
            "default_out_port": osc_config.get("default_out_port", 1233),
            "default_ip": osc_config.get("default_ip", "127.0.0.1")
        }
    
    def set_synthesizer_type(self, synth_type: str):
        """Set the default synthesizer type"""
        if "synthesizer" not in self._config:
            self._config["synthesizer"] = {}
        self._config["synthesizer"]["default_type"] = synth_type
        self._save_config()
    
    def _save_config(self):
        """Save current configuration to file"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self._config, f, indent=2)
        except Exception as e:
            print(f"Error saving configuration: {e}")


# Global configuration instance
config = ConfigManager()