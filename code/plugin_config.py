"""
Configuration management for Flow Synthesizer plugin paths and settings.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, List, Optional, Any


class PluginConfig:
    """Manages plugin configuration and path resolution."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize plugin configuration.
        
        Args:
            config_path: Path to configuration file. If None, searches for default locations.
        """
        self.config = {}
        self.load_config(config_path)
    
    def load_config(self, config_path: Optional[str] = None):
        """Load configuration from file."""
        search_paths = []
        
        if config_path:
            search_paths.append(config_path)
        
        # Add default search locations
        search_paths.extend([
            "./plugin_config.yml",
            "./config/plugin_config.yml",
            "~/.flow_synthesizer/config.yml",
            "~/.config/flow_synthesizer/config.yml"
        ])
        
        config_found = False
        for path_str in search_paths:
            path = Path(path_str).expanduser()
            if path.exists():
                try:
                    with open(path, 'r') as f:
                        self.config = yaml.safe_load(f) or {}
                    print(f"Loaded plugin configuration from: {path}")
                    config_found = True
                    break
                except Exception as e:
                    print(f"Warning: Error loading config from {path}: {e}")
        
        if not config_found:
            print("No configuration file found, using default settings")
            self.config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration when no config file is found."""
        return {
            "defaults": {
                "preferred_format": "vst3",
                "sample_rate": 22050,
                "block_size": 512,
                "midi_note": 60,
                "note_duration": 3.0,
                "render_duration": 4.0
            }
        }
    
    def get_plugin_paths(self, plugin_name: str) -> Dict[str, str]:
        """
        Get all configured paths for a plugin.
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            dict: Dictionary mapping format -> path
        """
        paths = {}
        
        # Check standard plugins
        plugins = self.config.get("plugins", {})
        if plugin_name in plugins:
            paths.update(plugins[plugin_name])
        
        # Check custom plugins
        custom = self.config.get("custom_plugins", {})
        if custom and plugin_name in custom:  # Handle None case
            paths.update(custom[plugin_name])
        
        return paths
    
    def get_preferred_plugin_path(self, plugin_name: str) -> Optional[str]:
        """
        Get the preferred plugin path for a plugin based on configuration.
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            str or None: Path to preferred plugin format, or None if not found
        """
        paths = self.get_plugin_paths(plugin_name)
        if not paths:
            return None
        
        # Get preferred format from config
        preferred_format = self.config.get("defaults", {}).get("preferred_format", "vst3")
        
        # Try preferred format first
        if preferred_format in paths:
            return paths[preferred_format]
        
        # Fall back to any available format
        format_priority = ["vst3", "au", "vst", "dll"]
        for fmt in format_priority:
            if fmt in paths:
                return paths[fmt]
        
        # If none of the priority formats exist, return the first available
        return list(paths.values())[0]
    
    def list_configured_plugins(self) -> List[str]:
        """Get list of all configured plugin names."""
        plugins = set()
        plugins.update(self.config.get("plugins", {}).keys())
        
        custom_plugins = self.config.get("custom_plugins", {})
        if custom_plugins:  # Handle None case
            plugins.update(custom_plugins.keys())
        
        return sorted(list(plugins))
    
    def get_audio_settings(self) -> Dict[str, Any]:
        """Get audio rendering settings from configuration."""
        defaults = self.config.get("defaults", {})
        return {
            "sample_rate": defaults.get("sample_rate", 22050),
            "block_size": defaults.get("block_size", 512),
            "midi_note": defaults.get("midi_note", 60),
            "note_duration": defaults.get("note_duration", 3.0),
            "render_duration": defaults.get("render_duration", 4.0)
        }
    
    def add_plugin_path(self, plugin_name: str, format_type: str, path: str, save: bool = False):
        """
        Add a new plugin path to the configuration.
        
        Args:
            plugin_name: Name of the plugin
            format_type: Format type (vst3, au, vst, dll)
            path: Path to the plugin
            save: Whether to save the configuration to file
        """
        if "custom_plugins" not in self.config:
            self.config["custom_plugins"] = {}
        
        if plugin_name not in self.config["custom_plugins"]:
            self.config["custom_plugins"][plugin_name] = {}
        
        self.config["custom_plugins"][plugin_name][format_type] = path
        
        if save:
            self.save_config()
    
    def save_config(self, path: Optional[str] = None):
        """Save current configuration to file."""
        if not path:
            path = "./plugin_config.yml"
        
        try:
            with open(path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, sort_keys=True)
            print(f"Configuration saved to: {path}")
        except Exception as e:
            print(f"Error saving configuration: {e}")


# Global configuration instance
_global_config = None

def get_config() -> PluginConfig:
    """Get the global plugin configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = PluginConfig()
    return _global_config

def reload_config(config_path: Optional[str] = None):
    """Reload the global configuration."""
    global _global_config
    _global_config = PluginConfig(config_path)