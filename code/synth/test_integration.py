#!/usr/bin/env python3
"""
Simple test script to verify pedalboard integration works correctly
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from synth.synthesize import create_synth, PEDALBOARD_AVAILABLE, LIBRENDERMAN_AVAILABLE

def test_backend_detection():
    """Test that backend detection works correctly"""
    print("=== Backend Detection Test ===")
    print(f"Pedalboard available: {PEDALBOARD_AVAILABLE}")
    print(f"Librenderman available: {LIBRENDERMAN_AVAILABLE}")
    assert PEDALBOARD_AVAILABLE or LIBRENDERMAN_AVAILABLE, "At least one backend should be available"
    print("✓ Backend detection passed")
    return True

def test_backend_auto_selection():
    """Test automatic backend selection"""
    print("\n=== Backend Auto-Selection Test ===")
    
    # Test with 'auto' backend - should not crash even without plugins
    try:
        engine, generator, defaults, rev_idx = create_synth('toy', 'diva', backend='auto')
        print("✓ Auto backend selection worked (unexpected - no plugins available)")
        return True
    except FileNotFoundError:
        print("✓ Auto backend selection failed gracefully (expected - no plugins)")
        return True
    except Exception as e:
        print(f"✗ Unexpected error in auto backend selection: {e}")
        return False

def test_parameter_loading():
    """Test parameter loading for different synthesizers"""
    print("\n=== Parameter Loading Test ===")
    
    # Test Serum parameters
    try:
        synth_dir = os.path.dirname(__file__)
        serum_params_file = os.path.join(synth_dir, 'serum_params.txt')
        serum_defaults_file = os.path.join(synth_dir, 'serum_param_default.json')
        
        if os.path.exists(serum_params_file) and os.path.exists(serum_defaults_file):
            import ast
            import json
            
            with open(serum_params_file, 'r') as f:
                serum_params = ast.literal_eval(f.read())
            
            with open(serum_defaults_file, 'r') as f:
                serum_defaults = json.load(f)
            
            print(f"✓ Serum parameters loaded: {len(serum_params)} params")
            print(f"✓ Serum defaults loaded: {len(serum_defaults)} params")
            
            # Verify some key Serum parameters exist
            key_params = ['OSC A: Volume', 'FILTER: Cutoff', 'ENV1: Attack']
            for param in key_params:
                assert param in serum_defaults, f"Missing key parameter: {param}"
            
            print("✓ Key Serum parameters verified")
        else:
            print("⚠ Serum parameter files not found")
            
    except Exception as e:
        print(f"✗ Error loading Serum parameters: {e}")
        return False
    
    return True

def test_pedalboard_import():
    """Test pedalboard-specific functionality"""
    print("\n=== Pedalboard Import Test ===")
    
    if PEDALBOARD_AVAILABLE:
        try:
            from synth.pedalboard_synth import PedalboardSynthEngine, PedalboardPatchGenerator
            print("✓ Pedalboard classes imported successfully")
            
            # Test engine creation (without plugin loading)
            engine = PedalboardSynthEngine()
            print("✓ PedalboardSynthEngine created successfully")
            
            return True
        except Exception as e:
            print(f"✗ Error testing pedalboard functionality: {e}")
            return False
    else:
        print("⚠ Pedalboard not available, skipping test")
        return True

def run_all_tests():
    """Run all tests"""
    print("Flow Synthesizer Pedalboard Integration Test")
    print("=" * 50)
    
    tests = [
        test_backend_detection,
        test_backend_auto_selection,
        test_parameter_loading,
        test_pedalboard_import
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Tests passed: {passed}")
    print(f"Tests failed: {failed}")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)