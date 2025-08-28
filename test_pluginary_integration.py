#!/usr/bin/env python3
"""
Test Pedalboard-Pluginary Integration

This script tests the basic integration of pedalboard-pluginary with Serum 2
to verify that the library can help solve the preset loading issues.
"""

import os
import sys
from pathlib import Path

# Add the code directory to Python path
sys.path.append(str(Path(__file__).parent / 'code'))

try:
    from pedalboard_pluginary import PedalboardPluginary
    PLUGINARY_AVAILABLE = True
    print("✓ pedalboard-pluginary is available")
except ImportError as e:
    print(f"✗ pedalboard-pluginary not available: {e}")
    PLUGINARY_AVAILABLE = False

try:
    import pedalboard
    from pedalboard import VST3Plugin
    PEDALBOARD_AVAILABLE = True
    print("✓ pedalboard is available")
except ImportError as e:
    print(f"✗ pedalboard not available: {e}")
    PEDALBOARD_AVAILABLE = False
    sys.exit(1)


def test_plugin_loading():
    """Test basic plugin loading"""
    plugin_path = '/Library/Audio/Plug-Ins/VST3/Serum2.vst3'
    
    print(f"\n=== Testing Plugin Loading ===")
    print(f"Plugin path: {plugin_path}")
    print(f"Plugin exists: {os.path.exists(plugin_path)}")
    
    if not os.path.exists(plugin_path):
        print("✗ Serum 2 plugin not found")
        return None
    
    try:
        # Load Serum 2 plugin
        plugin = VST3Plugin(plugin_path, plugin_name="Serum 2")
        print("✓ Successfully loaded Serum 2 VST3")
        
        # Get basic plugin info
        print(f"Plugin name: {getattr(plugin, 'name', 'Unknown')}")
        
        if hasattr(plugin, 'parameters'):
            param_count = len(plugin.parameters)
            print(f"Parameter count: {param_count}")
            
            # Show first few parameters
            if param_count > 0:
                print("First 5 parameters:")
                for i, (name, param) in enumerate(plugin.parameters.items()):
                    if i >= 5:
                        break
                    try:
                        value = getattr(plugin, name)
                        print(f"  {name}: {value}")
                    except Exception as e:
                        print(f"  {name}: <error reading value: {e}>")
        
        return plugin
        
    except Exception as e:
        print(f"✗ Failed to load plugin: {e}")
        return None


def test_pluginary_with_plugin(plugin):
    """Test pedalboard-pluginary with the loaded plugin"""
    if not PLUGINARY_AVAILABLE or not plugin:
        print("\n=== Skipping Pluginary Tests (not available or no plugin) ===")
        return
    
    print(f"\n=== Testing Pedalboard-Pluginary Integration ===")
    
    try:
        # Initialize pluginary
        pp = PedalboardPluginary()
        print("✓ PedalboardPluginary initialized")
        
        # Test plugin info retrieval
        try:
            plugin_info = pp.get_plugin_info('/Library/Audio/Plug-Ins/VST3/Serum2.vst3')
            print(f"✓ Plugin info retrieved: {type(plugin_info)}")
            
            # Try to get some basic info
            if hasattr(plugin_info, 'name'):
                print(f"  Plugin name: {plugin_info.name}")
            if hasattr(plugin_info, 'parameters'):
                print(f"  Parameters detected: {len(plugin_info.parameters) if plugin_info.parameters else 0}")
                
        except Exception as e:
            print(f"⚠ Plugin info retrieval failed: {e}")
        
        # Test plugin state retrieval
        try:
            current_state = pp.get_plugin_state(plugin)
            print(f"✓ Plugin state retrieved: {type(current_state)}")
            
            if hasattr(current_state, 'data'):
                print(f"  State data size: {len(current_state.data)} bytes")
            if hasattr(current_state, 'format'):
                print(f"  State format: {current_state.format}")
                
        except Exception as e:
            print(f"⚠ Plugin state retrieval failed: {e}")
        
        # Test parameter extraction capabilities
        try:
            # This might not work without an actual preset file, but let's try
            default_params = pp.get_default_parameter_values(plugin)
            print(f"✓ Default parameters retrieved: {len(default_params) if default_params else 0}")
            
        except Exception as e:
            print(f"⚠ Default parameter retrieval failed: {e}")
            
    except Exception as e:
        print(f"✗ Pluginary integration test failed: {e}")


def test_plugin_state_comparison(plugin):
    """Test plugin state comparison capabilities"""
    if not plugin:
        return
        
    print(f"\n=== Testing Plugin State Comparison ===")
    
    try:
        # Get initial raw state
        initial_raw_state = getattr(plugin, 'raw_state', None)
        print(f"Initial raw state: {len(initial_raw_state) if initial_raw_state else 0} bytes")
        
        # Get initial parameter snapshot
        initial_params = {}
        if hasattr(plugin, 'parameters'):
            for name in list(plugin.parameters.keys())[:10]:  # First 10 params
                try:
                    value = getattr(plugin, name)
                    initial_params[name] = value
                except Exception:
                    pass
        
        print(f"Initial parameter snapshot: {len(initial_params)} parameters")
        
        # Try to modify a parameter to test state change detection
        if initial_params:
            param_name = list(initial_params.keys())[0]
            original_value = initial_params[param_name]
            
            try:
                # Try to change the parameter
                if isinstance(original_value, (int, float)):
                    new_value = original_value + 0.1 if original_value < 0.9 else original_value - 0.1
                    setattr(plugin, param_name, new_value)
                    
                    # Check if it actually changed
                    current_value = getattr(plugin, param_name)
                    if abs(current_value - original_value) > 1e-6:
                        print(f"✓ Parameter change detected: {param_name} {original_value} -> {current_value}")
                        
                        # Check if raw state changed
                        new_raw_state = getattr(plugin, 'raw_state', None)
                        if new_raw_state and initial_raw_state:
                            if new_raw_state != initial_raw_state:
                                print(f"✓ Raw state changed: {len(initial_raw_state)} -> {len(new_raw_state)} bytes")
                            else:
                                print(f"⚠ Raw state unchanged despite parameter change")
                        
                        # Restore original value
                        setattr(plugin, param_name, original_value)
                    else:
                        print(f"⚠ Parameter change not detected")
                        
            except Exception as e:
                print(f"⚠ Parameter modification test failed: {e}")
        
    except Exception as e:
        print(f"✗ State comparison test failed: {e}")


def main():
    print("Pedalboard-Pluginary Integration Test")
    print("=====================================")
    
    # Test 1: Basic plugin loading
    plugin = test_plugin_loading()
    
    # Test 2: Pluginary integration
    test_pluginary_with_plugin(plugin)
    
    # Test 3: Plugin state comparison
    test_plugin_state_comparison(plugin)
    
    print(f"\n=== Summary ===")
    print(f"Pedalboard available: {PEDALBOARD_AVAILABLE}")
    print(f"Pluginary available: {PLUGINARY_AVAILABLE}")
    print(f"Plugin loaded: {plugin is not None}")
    
    if plugin and PLUGINARY_AVAILABLE:
        print("\n✓ Ready to test enhanced preset loading!")
        print("\nNext steps:")
        print("1. Find some .SerumPreset files to test with")
        print("2. Run enhanced_serum_preset_loader.py with actual presets")
        print("3. Verify that presets actually change plugin state")
        print("4. Integrate into dataset generation pipeline")
    elif plugin:
        print("\n⚠ Plugin loaded but pluginary not available")
        print("  - Can still test basic preset loading with raw_state")
        print("  - Limited debugging capabilities")
    else:
        print("\n✗ Cannot proceed without plugin")


if __name__ == '__main__':
    main()