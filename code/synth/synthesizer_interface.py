"""
Synthesizer abstraction layer for Flow Synthesizer
Supports multiple synthesizers including Diva VST and Massive X
"""
import json
import ast
try:
    import librenderman as rm
    LIBRENDERMAN_AVAILABLE = True
except ImportError:
    LIBRENDERMAN_AVAILABLE = False
    print("Warning: librenderman not available. Synthesizer engine initialization will be disabled.")

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List


class SynthesizerInterface(ABC):
    """Abstract base class for synthesizer interfaces"""
    
    def __init__(self, plugin_path: str, sample_rate: int = 44100, buffer_size: int = 512):
        self.plugin_path = plugin_path
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.engine = None
        self.generator = None
        
    @abstractmethod
    def load_parameter_mapping(self) -> Dict[int, str]:
        """Load parameter index to name mapping"""
        pass
    
    @abstractmethod
    def load_default_parameters(self, dataset: str = "default") -> Dict[str, float]:
        """Load default parameter values"""
        pass
    
    @abstractmethod
    def get_preset_path(self) -> str:
        """Get path to reset preset file"""
        pass
    
    def initialize_engine(self):
        """Initialize the RenderMan engine"""
        if not LIBRENDERMAN_AVAILABLE:
            print("Warning: Cannot initialize engine - librenderman not available")
            return
        self.engine = rm.RenderEngine(self.sample_rate, self.buffer_size, self.buffer_size)
        self.engine.load_plugin(self.plugin_path)
        self.generator = rm.PatchGenerator(self.engine)
        
    def create_reverse_mapping(self, param_mapping: Dict[int, str]) -> Dict[str, int]:
        """Create reverse mapping from parameter names to indices"""
        return {param_mapping[key]: key for key in param_mapping}


class DivaInterface(SynthesizerInterface):
    """Interface for u-he Diva VST synthesizer"""
    
    def load_parameter_mapping(self) -> Dict[int, str]:
        """Load Diva parameter mapping"""
        with open("synth/diva_params.txt") as f:
            return ast.literal_eval(f.read())
    
    def load_default_parameters(self, dataset: str = "default") -> Dict[str, float]:
        """Load Diva default parameters"""
        if dataset == "toy":
            with open("synth/param_nomod.json") as f:
                return json.load(f)
        else:
            with open("synth/param_default_32.json") as f:
                return json.load(f)
    
    def get_preset_path(self) -> str:
        """Get path to Diva reset preset"""
        return "synth/osc_reset.fxb"


class MassiveXInterface(SynthesizerInterface):
    """Interface for Native Instruments Massive X synthesizer"""
    
    def load_parameter_mapping(self) -> Dict[int, str]:
        """Load Massive X parameter mapping"""
        with open("synth/massive_x_params.txt") as f:
            return ast.literal_eval(f.read())
    
    def load_default_parameters(self, dataset: str = "default") -> Dict[str, float]:
        """Load Massive X default parameters"""
        # For now, we use the same defaults regardless of dataset
        # In future, we could have different presets for different purposes
        with open("synth/massive_x_default.json") as f:
            return json.load(f)
    
    def get_preset_path(self) -> str:
        """Get path to Massive X reset preset"""
        # Note: This would need to be created for Massive X
        # For now, we'll use the same reset mechanism
        return "synth/osc_reset.fxb"


class SynthesizerFactory:
    """Factory class for creating synthesizer interfaces"""
    
    _interfaces = {
        'diva': DivaInterface,
        'massive_x': MassiveXInterface,
        'massivex': MassiveXInterface,  # Alternative name
    }
    
    @classmethod
    def create_synthesizer(cls, synth_type: str, plugin_path: str, **kwargs) -> SynthesizerInterface:
        """Create a synthesizer interface based on type"""
        synth_type_lower = synth_type.lower()
        if synth_type_lower not in cls._interfaces:
            raise ValueError(f"Unsupported synthesizer type: {synth_type}. "
                           f"Supported types: {list(cls._interfaces.keys())}")
        
        interface_class = cls._interfaces[synth_type_lower]
        return interface_class(plugin_path, **kwargs)
    
    @classmethod
    def get_supported_synthesizers(cls) -> List[str]:
        """Get list of supported synthesizer types"""
        return list(cls._interfaces.keys())


def create_synth(synth_type: str = None, plugin_path: str = None, dataset: str = "default"):
    """
    Create and initialize a synthesizer interface
    
    Args:
        synth_type: Type of synthesizer ('diva' or 'massive_x'). If None, uses config default.
        plugin_path: Path to the VST/AU plugin. If None, uses config default.
        dataset: Dataset type for parameter selection
        
    Returns:
        Tuple of (engine, generator, param_defaults, reverse_mapping)
    """
    # Import here to avoid circular imports
    from config_manager import config
    
    # Use configuration defaults if not provided
    if synth_type is None:
        synth_type = config.get_synthesizer_type()
    
    if plugin_path is None:
        plugin_path = config.get_plugin_path(synth_type)
    
    # Create synthesizer interface
    synth = SynthesizerFactory.create_synthesizer(synth_type, plugin_path)
    
    # Load parameter mapping and defaults
    param_mapping = synth.load_parameter_mapping()
    param_defaults = synth.load_default_parameters(dataset)
    reverse_mapping = synth.create_reverse_mapping(param_mapping)
    
    # Initialize engine
    synth.initialize_engine()
    
    return synth.engine, synth.generator, param_defaults, reverse_mapping


# Maintain backward compatibility with existing code
def create_diva_synth(dataset: str = "default", path: str = 'synth/diva.64.so'):
    """Backward compatibility function for creating Diva synthesizer"""
    return create_synth('diva', path, dataset)