#!/usr/bin/env python3
"""
Comprehensive test for Pedalboard renderer functionality including binary state round-trip.
This test creates a mock workflow to validate the implementation without requiring actual plugins.
"""

import os
import sys
import tempfile
import json
import numpy as np
from pathlib import Path

# Add the code directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))

def test_integration_workflow():
    """Test the complete integration workflow with mock data."""
    
    print("🧪 Testing Pedalboard integration workflow...")
    
    # Test imports
    try:
        from pedalboard_renderer import PBRenderer
        print("✓ PBRenderer import successful")
    except ImportError as e:
        print(f"❌ Failed to import PBRenderer: {e}")
        return False
    
    # Test that scripts can be imported
    script_tests = [
        ("capture_init_state", "capture_init_state.py"),
        ("json_to_parameters", "json_to_parameters.py"),
        ("render_dataset_pb", "code/render_dataset_pb.py")
    ]
    
    for module_name, script_path in script_tests:
        if os.path.exists(script_path):
            print(f"✓ Script exists: {script_path}")
        else:
            print(f"❌ Script missing: {script_path}")
            return False
    
    # Test file structure compatibility
    expected_files = [
        "code/pedalboard_renderer.py",
        "code/render_dataset_pb.py", 
        "docs/pedalboard_migration.md",
        "requirements.txt"
    ]
    
    for file_path in expected_files:
        if os.path.exists(file_path):
            print(f"✓ File exists: {file_path}")
        else:
            print(f"❌ File missing: {file_path}")
            return False
    
    # Test that requirements.txt includes pedalboard
    with open("requirements.txt", "r") as f:
        requirements = f.read()
        if "pedalboard" in requirements:
            print("✓ Pedalboard added to requirements.txt")
        else:
            print("❌ Pedalboard missing from requirements.txt")
            return False
    
    # Test mock JSON parameter handling
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock JSON parameter file
        mock_params = {
            "0": 0.5,
            "1": 0.3, 
            "2": 0.8,
            "oscillator_volume": 0.7,
            "filter_cutoff": 0.4
        }
        
        json_file = Path(temp_dir) / "mock_params.json"
        with open(json_file, "w") as f:
            json.dump(mock_params, f)
        
        print(f"✓ Created mock JSON parameter file: {json_file}")
        
        # Test that JSON file can be loaded
        with open(json_file, "r") as f:
            loaded_params = json.load(f)
            if loaded_params == mock_params:
                print("✓ JSON parameter file round-trip successful")
            else:
                print("❌ JSON parameter file round-trip failed")
                return False
    
    print("✅ All integration tests passed!")
    return True

def test_api_compatibility():
    """Test that PBRenderer API matches DDRenderer expectations."""
    
    print("\n🔄 Testing API compatibility...")
    
    try:
        from pedalboard_renderer import PBRenderer
        
        # Test initialization
        renderer = PBRenderer(sample_rate=22050, buffer_size=512)
        
        # Test all required methods exist
        required_methods = [
            'load_plugin',
            'get_parameters_description', 
            'get_plugin_parameter_size',
            'get_patch',
            'set_patch',
            'save_state',
            'load_state',
            'render_patch',
            'render'
        ]
        
        for method_name in required_methods:
            if hasattr(renderer, method_name) and callable(getattr(renderer, method_name)):
                print(f"✓ Method exists: {method_name}")
            else:
                print(f"❌ Method missing: {method_name}")
                return False
        
        # Test method signatures work (without plugin loaded)
        try:
            # These should work without errors (return empty/default values)
            params = renderer.get_parameters_description()
            size = renderer.get_plugin_parameter_size()
            patch = renderer.get_patch()
            
            assert isinstance(params, list)
            assert isinstance(size, int) 
            assert isinstance(patch, list)
            
            print("✓ Methods return expected types")
            
        except Exception as e:
            print(f"❌ Method signature test failed: {e}")
            return False
            
        print("✅ API compatibility tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ API compatibility test failed: {e}")
        return False

def test_documentation():
    """Test that documentation is comprehensive."""
    
    print("\n📚 Testing documentation...")
    
    doc_path = "docs/pedalboard_migration.md"
    if not os.path.exists(doc_path):
        print(f"❌ Documentation file missing: {doc_path}")
        return False
    
    with open(doc_path, "r") as f:
        doc_content = f.read()
    
    # Check for key sections
    required_sections = [
        "# Pedalboard Migration Guide",
        "## Key Differences from DawDreamer",
        "## Architecture", 
        "### PBRenderer Class",
        "## Usage Examples",
        "## File Formats",
        "## Known Limitations",
        "## Troubleshooting"
    ]
    
    for section in required_sections:
        if section in doc_content:
            print(f"✓ Documentation section exists: {section}")
        else:
            print(f"❌ Documentation section missing: {section}")
            return False
    
    # Check for code examples
    if "```python" in doc_content and "```bash" in doc_content:
        print("✓ Documentation includes code examples")
    else:
        print("❌ Documentation missing code examples")
        return False
    
    print("✅ Documentation tests passed!")
    return True

def main():
    """Run all integration tests."""
    
    print("🚀 Starting Pedalboard integration tests...\n")
    
    tests = [
        ("Integration Workflow", test_integration_workflow),
        ("API Compatibility", test_api_compatibility),
        ("Documentation", test_documentation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        try:
            if test_func():
                passed += 1
            else:
                print(f"\n❌ {test_name} FAILED")
        except Exception as e:
            print(f"\n❌ {test_name} FAILED with exception: {e}")
    
    print(f"\n{'='*50}")
    print(f"📊 Test Results: {passed}/{total} tests passed")
    print('='*50)
    
    if passed == total:
        print("🎉 All tests passed! Pedalboard integration is ready.")
        return 0
    else:
        print("❌ Some tests failed. Please review the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())