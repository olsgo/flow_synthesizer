#!/usr/bin/env python3
"""
Test script for synthesizer interface
Verifies that both Diva and Massive X interfaces can be created
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from synth.synthesizer_interface import SynthesizerFactory, create_synth
from config_manager import config

def test_synthesizer_creation():
    """Test creating synthesizer interfaces"""
    print("Testing synthesizer interface creation...")
    
    # Test getting supported synthesizers
    supported = SynthesizerFactory.get_supported_synthesizers()
    print(f"Supported synthesizers: {supported}")
    
    # Test creating Diva interface (without actual plugin)
    try:
        print("\nTesting Diva interface creation...")
        diva_interface = SynthesizerFactory.create_synthesizer('diva', 'dummy_path.so')
        param_mapping = diva_interface.load_parameter_mapping()
        defaults = diva_interface.load_default_parameters('default')
        print(f"✓ Diva interface created successfully")
        print(f"  - Parameters: {len(param_mapping)}")
        print(f"  - Defaults: {len(defaults)}")
        print(f"  - Sample parameters: {list(param_mapping.values())[:5]}")
    except Exception as e:
        print(f"✗ Error creating Diva interface: {e}")
    
    # Test creating Massive X interface (without actual plugin)
    try:
        print("\nTesting Massive X interface creation...")
        massive_interface = SynthesizerFactory.create_synthesizer('massive_x', 'dummy_path.so')
        param_mapping = massive_interface.load_parameter_mapping()
        defaults = massive_interface.load_default_parameters('default')
        print(f"✓ Massive X interface created successfully")
        print(f"  - Parameters: {len(param_mapping)}")
        print(f"  - Defaults: {len(defaults)}")
        print(f"  - Sample parameters: {list(param_mapping.values())[:5]}")
    except Exception as e:
        print(f"✗ Error creating Massive X interface: {e}")
    
    # Test configuration manager
    try:
        print("\nTesting configuration manager...")
        default_synth = config.get_synthesizer_type()
        diva_path = config.get_plugin_path('diva')
        massive_path = config.get_plugin_path('massive_x')
        audio_config = config.get_audio_config()
        print(f"✓ Configuration loaded successfully")
        print(f"  - Default synthesizer: {default_synth}")
        print(f"  - Diva path: {diva_path}")
        print(f"  - Massive X path: {massive_path}")
        print(f"  - Audio config: {audio_config}")
    except Exception as e:
        print(f"✗ Error with configuration: {e}")

def test_parameter_compatibility():
    """Test parameter compatibility between synthesizers"""
    print("\n" + "="*50)
    print("Testing parameter compatibility...")
    
    try:
        # Load Diva parameters
        diva_interface = SynthesizerFactory.create_synthesizer('diva', 'dummy.so')
        diva_params = diva_interface.load_parameter_mapping()
        diva_defaults = diva_interface.load_default_parameters()
        
        # Load Massive X parameters
        massive_interface = SynthesizerFactory.create_synthesizer('massive_x', 'dummy.so')
        massive_params = massive_interface.load_parameter_mapping()
        massive_defaults = massive_interface.load_default_parameters()
        
        print(f"Diva parameters: {len(diva_params)}")
        print(f"Massive X parameters: {len(massive_params)}")
        
        # Check for common parameter types
        diva_names = set(diva_params.values())
        massive_names = set(massive_params.values())
        
        # Look for similar parameter categories
        diva_categories = set(name.split(':')[0] for name in diva_names if ':' in name)
        massive_categories = set(name.split(':')[0] for name in massive_names if ':' in name)
        
        common_categories = diva_categories.intersection(massive_categories)
        print(f"Common parameter categories: {common_categories}")
        
        print("✓ Parameter compatibility check completed")
        
    except Exception as e:
        print(f"✗ Error in parameter compatibility test: {e}")

if __name__ == "__main__":
    print("Flow Synthesizer - Synthesizer Interface Test")
    print("=" * 50)
    
    test_synthesizer_creation()
    test_parameter_compatibility()
    
    print("\n" + "="*50)
    print("Test completed!")