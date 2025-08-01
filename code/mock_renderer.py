"""
Mock plugin system for testing environments where real plugins are not available.

This module provides a MockDDRenderer that simulates plugin behavior for testing
and development purposes, especially useful in CI/CD environments.
"""

import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

# Handle imports for both standalone and package usage
try:
    from code.plugin_config import get_config
except ImportError:
    try:
        from plugin_config import get_config
    except ImportError:
        from .plugin_config import get_config


class MockPlugin:
    """Mock plugin that simulates real plugin behavior."""
    
    def __init__(self, plugin_name: str, param_count: int = 64):
        self.plugin_name = plugin_name
        self.param_count = param_count
        self.parameters = {f"param_{i}": 0.5 for i in range(param_count)}
        self.parameter_descriptions = [
            {"name": f"param_{i}", "min": 0.0, "max": 1.0, "default": 0.5}
            for i in range(param_count)
        ]
    
    def get_plugin_parameter_size(self):
        """Get number of parameters."""
        return self.param_count
    
    def get_parameters_description(self):
        """Get parameter descriptions."""
        return self.parameter_descriptions
    
    def get_parameter(self, index: int) -> float:
        """Get parameter value by index."""
        if 0 <= index < self.param_count:
            return list(self.parameters.values())[index]
        return 0.0
    
    def set_parameter(self, index: int, value: float):
        """Set parameter value by index."""
        if 0 <= index < self.param_count:
            param_name = f"param_{index}"
            self.parameters[param_name] = max(0.0, min(1.0, value))
    
    def clear_midi(self):
        """Clear MIDI events (mock implementation)."""
        pass
    
    def add_midi_note(self, note: int, velocity: int, start_time: float, duration: float):
        """Add MIDI note (mock implementation)."""
        pass


class MockEngine:
    """Mock audio engine for testing."""
    
    def __init__(self, sample_rate: int, block_size: int):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.audio_data = None
    
    def make_plugin_processor(self, name: str, plugin_path: str):
        """Create a mock plugin processor."""
        # Extract plugin name from path
        plugin_name = plugin_path.split("/")[-1].replace(".vst3", "").replace(".component", "").replace(".vst", "")
        
        # Return different mock plugins based on the path
        if "diva" in plugin_name.lower():
            return MockPlugin("Diva", param_count=64)
        elif "massive" in plugin_name.lower():
            return MockPlugin("Massive X", param_count=128)
        elif "fm8" in plugin_name.lower():
            return MockPlugin("FM8", param_count=96)
        elif "polymax" in plugin_name.lower():
            return MockPlugin("Polymax", param_count=48)
        elif "serum" in plugin_name.lower():
            return MockPlugin("Serum", param_count=256)
        else:
            return MockPlugin("Generic Synth", param_count=64)
    
    def load_graph(self, graph):
        """Load processing graph (mock implementation)."""
        pass
    
    def render(self, duration: float):
        """Render audio (mock implementation)."""
        samples = int(duration * self.sample_rate)
        # Generate simple synthetic audio for testing
        t = np.linspace(0, duration, samples)
        frequency = 440.0  # A4
        audio = 0.3 * np.sin(2 * np.pi * frequency * t)
        
        # Add some noise and envelope
        noise = 0.05 * np.random.randn(samples)
        envelope = np.exp(-3 * t)  # Exponential decay
        audio = (audio + noise) * envelope
        
        # Convert to stereo
        self.audio_data = np.array([audio, audio])
    
    def get_audio(self):
        """Get rendered audio data."""
        if self.audio_data is None:
            # Return silence if no audio was rendered
            return np.zeros((2, int(4.0 * self.sample_rate)))
        return self.audio_data


class MockDDRenderer:
    """
    Mock DDRenderer that simulates real plugin behavior for testing.
    
    This class provides the same interface as DDRenderer but uses mock plugins
    instead of real audio plugins. Useful for testing and CI environments.
    """
    
    def __init__(self, sample_rate=22050, block_size=512):
        self.sr = sample_rate
        self.engine = MockEngine(self.sr, block_size)
        self.inst = None
        self.mock_mode = True
    
    @classmethod
    def from_config(cls):
        """Create a MockDDRenderer with settings from configuration."""
        config = get_config()
        settings = config.get_audio_settings()
        return cls(sample_rate=settings["sample_rate"], block_size=settings["block_size"])
    
    def load_plugin(self, plugin_path: str, name: str = "synth") -> bool:
        """
        Load a mock plugin that simulates the real plugin.
        
        Args:
            plugin_path: Path to the plugin (used to determine mock behavior)
            name: Internal name for the plugin processor
            
        Returns:
            bool: Always True for mock plugins
        """
        try:
            self.inst = self.engine.make_plugin_processor(name, plugin_path)
            print(f"Mock plugin loaded: {self.inst.plugin_name} (simulated)")
            return True
        except Exception as e:
            print(f"Warning: Failed to create mock plugin for '{plugin_path}': {e}")
            self.inst = None
            return False
    
    def load_plugin_by_name(self, plugin_name: str, name: str = "synth") -> bool:
        """Load a mock plugin by configured name."""
        config = get_config()
        plugin_path = config.get_preferred_plugin_path(plugin_name)
        
        if not plugin_path:
            # Create a mock path for unknown plugins
            plugin_path = f"/mock/plugins/{plugin_name}.vst3"
        
        print(f"Loading mock {plugin_name}")
        return self.load_plugin(plugin_path, name)
    
    def is_plugin_loaded(self) -> bool:
        """Check if a plugin is currently loaded."""
        return self.inst is not None
    
    def get_plugin_info(self) -> dict:
        """Get information about the currently loaded mock plugin."""
        if not self.is_plugin_loaded():
            return {"loaded": False, "error": "No plugin loaded"}
        
        try:
            param_count = self.get_plugin_parameter_size()
            param_desc = self.get_parameters_description()
            return {
                "loaded": True,
                "parameter_count": param_count,
                "parameter_names": [p.get("name", f"param_{i}") for i, p in enumerate(param_desc[:10])],
                "total_parameters": len(param_desc),
                "mock": True,
                "plugin_name": self.inst.plugin_name
            }
        except Exception as e:
            return {"loaded": False, "error": f"Mock plugin error: {e}"}
    
    # --- parameter I/O ---
    def get_parameters_description(self):
        """Get parameter descriptions from mock plugin."""
        if self.inst is None:
            return []
        return self.inst.get_parameters_description()
    
    def get_plugin_parameter_size(self) -> int:
        """Get parameter count from mock plugin."""
        if self.inst is None:
            return 0
        return self.inst.get_plugin_parameter_size()
    
    def get_patch(self):
        """Get current parameter values from mock plugin."""
        if self.inst is None:
            return []
        n = self.get_plugin_parameter_size()
        return [(i, self.inst.get_parameter(i)) for i in range(n)]
    
    def set_patch(self, patch):
        """Set parameter values on mock plugin."""
        if self.inst is None:
            raise RuntimeError("No plugin loaded. Call load_plugin() first.")
        for idx, val in patch:
            self.inst.set_parameter(int(idx), float(val))
    
    # --- rendering ---
    def render_patch(self, midi_note=60, velocity=100, note_len_sec=3.0, render_len_sec=4.0):
        """Render audio with mock plugin."""
        if self.inst is None:
            raise RuntimeError("No plugin loaded. Call load_plugin() first.")
        
        print(f"Mock rendering: note={midi_note}, velocity={velocity}, duration={render_len_sec}s")
        
        self.inst.clear_midi()
        self.inst.add_midi_note(midi_note, velocity, 0.0, note_len_sec)
        self.engine.load_graph([(self.inst, [])])
        self.engine.render(render_len_sec)
        return self.engine.get_audio()
    
    # Mock implementations of preset/state methods
    def load_vst3_preset(self, preset_path: str):
        """Mock VST3 preset loading."""
        if self.inst is None:
            raise RuntimeError("No plugin loaded. Call load_plugin() first.")
        print(f"Mock: Loading VST3 preset from {preset_path}")
        return True
    
    def load_state(self, path: str):
        """Mock state loading."""
        if self.inst is None:
            raise RuntimeError("No plugin loaded. Call load_plugin() first.")
        print(f"Mock: Loading state from {path}")
        return True
    
    # Utility methods that work the same as real DDRenderer
    @staticmethod
    def validate_plugin_path(plugin_path: str) -> dict:
        """
        Mock plugin path validation - always returns valid for testing.
        
        Returns:
            dict: Always indicates valid for mock testing
        """
        return {
            "valid": True,
            "path": plugin_path,
            "type": "Mock Plugin",
            "message": "Mock plugin - always valid for testing"
        }
    
    @staticmethod
    def find_plugin_paths(plugin_name: str) -> list:
        """
        Mock plugin search - returns simulated paths.
        
        Returns:
            list: Mock plugin paths for testing
        """
        return [
            f"/mock/vst3/{plugin_name}.vst3",
            f"/mock/au/{plugin_name}.component",
            f"/mock/vst/{plugin_name}.vst"
        ]


def create_mock_renderer(use_mock: bool = None) -> Any:
    """
    Factory function to create either a real or mock renderer.
    
    Args:
        use_mock: If True, creates MockDDRenderer. If False, creates real DDRenderer.
                 If None, automatically detects based on environment.
    
    Returns:
        DDRenderer or MockDDRenderer instance
    """
    if use_mock is None:
        # Auto-detect: use mock if real plugins are not available
        try:
            from code.dd_renderer import DDRenderer
            test_renderer = DDRenderer(22050, 512)
            
            # Try to load a common plugin to test if real plugins work
            config = get_config()
            plugins = config.list_configured_plugins()
            
            real_plugins_available = False
            if plugins:
                for plugin_name in plugins[:3]:  # Test first 3 plugins
                    if test_renderer.load_plugin_by_name(plugin_name):
                        real_plugins_available = True
                        break
            
            use_mock = not real_plugins_available
            
        except Exception:
            use_mock = True
    
    if use_mock:
        print("Using MockDDRenderer for testing (no real plugins available)")
        return MockDDRenderer.from_config()
    else:
        print("Using real DDRenderer")
        from code.dd_renderer import DDRenderer
        return DDRenderer.from_config()


# Auto-detection convenience function
def get_renderer(prefer_real: bool = True):
    """
    Get the best available renderer (real or mock).
    
    Args:
        prefer_real: If True, tries real renderer first, falls back to mock
                    If False, uses mock renderer
    
    Returns:
        Renderer instance (real or mock)
    """
    if not prefer_real:
        return MockDDRenderer.from_config()
    
    return create_mock_renderer(use_mock=None)