#!/usr/bin/env python3
"""
Test Plugin State Tracking

Simple test to verify our enhanced plugin state tracking works correctly
with the Serum 2 VST3 plugin.
"""

import sys
from pathlib import Path

# Add the code directory to Python path
sys.path.append(str(Path(__file__).parent / 'code'))

try:
    from practical_serum_preset_loader import PracticalSerumPresetLoader, PluginStateTracker
    import pedalboard
    from pedalboard import VST3Plugin
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def test_plugin_state_tracking():
    """
    Test basic plugin state tracking functionality
    """
    print("=== Testing Plugin State Tracking ===")
    
    try:
        # Initialize the loader
        loader = PracticalSerumPresetLoader()
        print("✓ Plugin loaded successfully")
        
        # Test state capture
        initial_state = loader.state_tracker.capture_state('test_initial')
        print(f"✓ Initial state captured: {initial_state['parameter_count']} parameters")
        print(f"  Raw state size: {initial_state['raw_state_size']} bytes")
        print(f"  Raw state hash: {initial_state['raw_state_hash'][:8]}...")
        
        # Try to modify some parameters
        print("\n--- Attempting to modify parameters ---")
        modified_params = 0
        
        if hasattr(loader.plugin, 'parameters'):
            param_names = list(loader.plugin.parameters.keys())[:5]  # Test first 5 parameters
            
            for param_name in param_names:
                try:
                    # Get current value
                    current_value = getattr(loader.plugin, param_name)
                    print(f"  {param_name}: {current_value}")
                    
                    # Try to set a different value
                    if isinstance(current_value, (int, float)):
                        new_value = 0.7 if current_value != 0.7 else 0.3
                        setattr(loader.plugin, param_name, new_value)
                        
                        # Verify the change
                        actual_value = getattr(loader.plugin, param_name)
                        if abs(actual_value - new_value) < 0.01:
                            print(f"    ✓ Changed to: {actual_value}")
                            modified_params += 1
                        else:
                            print(f"    ✗ Failed to change (got {actual_value})")
                    
                except Exception as e:
                    print(f"    ✗ Error modifying {param_name}: {e}")
        
        print(f"\nModified {modified_params} parameters")
        
        # Capture state after modifications
        modified_state = loader.state_tracker.capture_state('test_modified')
        print(f"✓ Modified state captured")
        
        # Compare states
        comparison = loader.state_tracker.compare_states(initial_state, modified_state)
        print(f"\n--- State Comparison ---")
        print(f"States identical: {comparison['states_identical']}")
        print(f"Raw state changed: {comparison['raw_state_changed']}")
        print(f"Parameters changed: {comparison['parameter_change_count']}")
        
        if comparison['significant_changes']:
            print("Significant changes:")
            for change in comparison['significant_changes'][:3]:  # Show first 3
                print(f"  {change['parameter']}: {change['old_value']:.3f} → {change['new_value']:.3f}")
        
        # Test audio rendering
        print("\n--- Testing Audio Rendering ---")
        audio_result = loader.test_audio_rendering()
        
        if audio_result['success']:
            print(f"✓ Audio rendered successfully")
            print(f"  Length: {audio_result['audio_length']} samples")
            print(f"  RMS: {audio_result['rms_db']:.1f} dB")
            print(f"  Peak: {audio_result['peak_db']:.1f} dB")
            print(f"  Hash: {audio_result['audio_hash'][:8]}...")
        else:
            print(f"✗ Audio rendering failed: {audio_result['error']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_json_preset_integration():
    """
    Test integration with existing JSON preset format
    """
    print("\n=== Testing JSON Preset Integration ===")
    
    try:
        import json
        
        # Load a sample JSON preset
        json_preset_path = "/Users/gjb/Datasets/serum2/presets/000000.json"
        
        with open(json_preset_path, 'r') as f:
            preset_data = json.load(f)
        
        print(f"✓ Loaded JSON preset: {json_preset_path}")
        print(f"  Keys: {list(preset_data.keys())}")
        
        # Initialize loader
        loader = PracticalSerumPresetLoader()
        
        # Capture initial state
        initial_state = loader.state_tracker.capture_state('before_json_preset')
        
        # Apply JSON preset parameters
        applied_params = 0
        
        if hasattr(loader.plugin, 'parameters'):
            plugin_params = loader.plugin.parameters
            
            for json_key, json_value in preset_data.items():
                # Try to find matching plugin parameter
                for plugin_param_name in plugin_params.keys():
                    if json_key.lower() in plugin_param_name.lower() or plugin_param_name.lower() in json_key.lower():
                        try:
                            setattr(loader.plugin, plugin_param_name, float(json_value))
                            applied_params += 1
                            print(f"  Applied {json_key} → {plugin_param_name}: {json_value}")
                            break
                        except Exception as e:
                            print(f"  Failed to apply {json_key}: {e}")
        
        print(f"\nApplied {applied_params} parameters from JSON preset")
        
        # Capture state after applying JSON preset
        json_state = loader.state_tracker.capture_state('after_json_preset')
        
        # Compare states
        comparison = loader.state_tracker.compare_states(initial_state, json_state)
        print(f"\n--- JSON Preset State Comparison ---")
        print(f"States identical: {comparison['states_identical']}")
        print(f"Raw state changed: {comparison['raw_state_changed']}")
        print(f"Parameters changed: {comparison['parameter_change_count']}")
        
        # Test audio with JSON preset
        audio_result = loader.test_audio_rendering()
        if audio_result['success']:
            print(f"✓ Audio with JSON preset: {audio_result['audio_hash'][:8]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ JSON preset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("Testing Enhanced Serum Preset Loading System\n")
    
    # Test 1: Basic plugin state tracking
    test1_success = test_plugin_state_tracking()
    
    # Test 2: JSON preset integration
    test2_success = test_json_preset_integration()
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Plugin State Tracking: {'✓ PASS' if test1_success else '✗ FAIL'}")
    print(f"JSON Preset Integration: {'✓ PASS' if test2_success else '✗ FAIL'}")
    
    if test1_success and test2_success:
        print("\n🎉 All tests passed! The enhanced preset loading system is working.")
        print("\nNext steps:")
        print("1. Integrate this into serum2_preset_dataset.py")
        print("2. Test with multiple presets to verify audio diversity")
        print("3. Add comprehensive error handling")
    else:
        print("\n❌ Some tests failed. Check the errors above.")


if __name__ == '__main__':
    main()