#!/usr/bin/env python3
"""
Test framework for audio plugin loading with graceful fallbacks.
This test file demonstrates best practices for plugin testing.
"""

import sys
from pathlib import Path
from code.dd_renderer import DDRenderer
from code.plugin_config import get_config


def test_diva_plugin():
    """Test Diva plugin loading with proper error handling and fallbacks."""
    print("=== Testing Diva Plugin ===")
    
    config = get_config()
    renderer = DDRenderer.from_config()
    
    # Method 1: Try loading from configuration
    print("Method 1: Loading from configuration...")
    success = renderer.load_plugin_by_name("diva")
    
    if success:
        print("✅ Diva loaded successfully from configuration!")
        info = renderer.get_plugin_info()
        print(f"   Parameter count: {info['parameter_count']}")
        return True
    
    # Method 2: Search for Diva plugins on the system
    print("Method 2: Searching for Diva plugins on system...")
    found_paths = DDRenderer.find_plugin_paths("Diva")
    
    for path in found_paths:
        print(f"   Trying: {path}")
        success = renderer.load_plugin(path)
        if success:
            print("✅ Diva loaded successfully from auto-detected path!")
            info = renderer.get_plugin_info()
            print(f"   Parameter count: {info['parameter_count']}")
            return True
    
    # Method 3: Try common macOS paths with different variations
    print("Method 3: Trying common paths...")
    common_paths = [
        "/Library/Audio/Plug-Ins/VST3/Diva.vst3",
        "/Library/Audio/Plug-Ins/Components/Diva.component", 
        "/Library/Audio/Plug-Ins/VST/u-he/Diva.vst",
        "~/Library/Audio/Plug-Ins/VST3/Diva.vst3",
        "~/Library/Audio/Plug-Ins/Components/Diva.component",
        "~/Library/Audio/Plug-Ins/VST/u-he/Diva.vst",
    ]
    
    for path in common_paths:
        expanded_path = str(Path(path).expanduser())
        if Path(expanded_path).exists():
            print(f"   Trying: {expanded_path}")
            success = renderer.load_plugin(expanded_path)
            if success:
                print("✅ Diva loaded successfully from common path!")
                info = renderer.get_plugin_info()
                print(f"   Parameter count: {info['parameter_count']}")
                return True
    
    # Fallback: Mock/simulation mode
    print("Method 4: Plugin not available - entering simulation mode")
    print("⚠️  Diva plugin not found on system")
    print("   This is normal in CI/testing environments")
    print("   To use Diva:")
    print("   1. Install Diva VST from u-he.com")
    print("   2. Update plugin_config.yml with correct paths")
    print("   3. Run: python plugin_debug_tool.py --test-plugin diva")
    
    return False


def test_basic_dd_renderer():
    """Test basic DDRenderer functionality without specific plugins."""
    print("\n=== Testing DDRenderer Basic Functionality ===")
    
    try:
        # Test initialization
        renderer = DDRenderer(sample_rate=22050, block_size=512)
        print("✅ DDRenderer initialization successful")
        
        # Test configuration-based initialization
        renderer_config = DDRenderer.from_config()
        print("✅ DDRenderer config-based initialization successful")
        
        # Test utility functions
        validation = DDRenderer.validate_plugin_path("/nonexistent/path")
        assert not validation['valid'], "Should detect invalid path"
        print("✅ Path validation working correctly")
        
        # Test search functionality (shouldn't crash even if no plugins found)
        found_paths = DDRenderer.find_plugin_paths("NonexistentPlugin")
        assert isinstance(found_paths, list), "Should return list even if empty"
        print("✅ Plugin search functionality working")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic DDRenderer test failed: {e}")
        return False


def test_configuration_system():
    """Test the plugin configuration system."""
    print("\n=== Testing Configuration System ===")
    
    try:
        config = get_config()
        print("✅ Configuration loading successful")
        
        # Test plugin listing
        plugins = config.list_configured_plugins()
        print(f"✅ Found {len(plugins)} configured plugins")
        
        # Test audio settings
        settings = config.get_audio_settings()
        assert 'sample_rate' in settings, "Should have sample_rate setting"
        print("✅ Audio settings loading successful")
        
        # Test plugin path resolution
        if plugins:
            test_plugin = plugins[0]
            paths = config.get_plugin_paths(test_plugin)
            preferred = config.get_preferred_plugin_path(test_plugin)
            print(f"✅ Plugin path resolution working for {test_plugin}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration system test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and provide a summary."""
    print("Flow Synthesizer Plugin Test Suite")
    print("==================================")
    
    results = {}
    
    # Run individual tests
    results['basic_renderer'] = test_basic_dd_renderer()
    results['configuration'] = test_configuration_system()
    results['diva_plugin'] = test_diva_plugin()
    
    # Summary
    print("\n=== Test Summary ===")
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("⚠️  Some tests failed - this may be normal in environments without plugins")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)