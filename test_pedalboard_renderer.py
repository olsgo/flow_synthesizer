#!/usr/bin/env python3
"""
Test Pedalboard renderer functionality with basic plugin loading and state management.
This test doesn't require specific plugins and focuses on the API compatibility.
"""

import os
import sys
import tempfile
import numpy as np

# Add the code directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))

from pedalboard_renderer import PBRenderer

def test_pbrenderer_basic():
    """Test basic PBRenderer functionality without requiring specific plugins."""
    
    print("Testing PBRenderer basic functionality...")
    
    # Test initialization
    renderer = PBRenderer(sample_rate=22050, buffer_size=512)
    assert renderer.sr == 22050
    assert renderer.buffer_size == 512
    assert renderer.plugin is None
    print("✓ Initialization successful")
    
    # Test methods work with no plugin loaded
    assert renderer.get_plugin_parameter_size() == 0
    assert renderer.get_parameters_description() == []
    assert renderer.get_patch() == []
    print("✓ No-plugin state methods work correctly")
    
    # Test error handling for operations requiring a plugin
    try:
        renderer.set_patch([(0, 0.5)])
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "No plugin loaded" in str(e)
        print("✓ set_patch error handling works")
    
    try:
        renderer.save_state("/tmp/test.bin")
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "No plugin loaded" in str(e)
        print("✓ save_state error handling works")
    
    try:
        renderer.render_patch()
        assert False, "Should have raised RuntimeError" 
    except RuntimeError as e:
        assert "No plugin loaded" in str(e)
        print("✓ render_patch error handling works")
    
    print("All basic tests passed!")

def test_pbrenderer_state_file_handling():
    """Test state file operations without actual plugins."""
    
    print("\nTesting state file handling...")
    
    renderer = PBRenderer()
    
    # Test load_state with no plugin loaded - should raise RuntimeError
    try:
        renderer.load_state("/nonexistent/path.bin")
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "No plugin loaded" in str(e)
        print("✓ load_state error handling works")
    
    print("State file handling tests completed!")

if __name__ == "__main__":
    test_pbrenderer_basic()
    test_pbrenderer_state_file_handling()
    print("\n🎉 All PBRenderer tests completed successfully!")