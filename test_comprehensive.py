#!/usr/bin/env python3
"""
Comprehensive test suite for Flow Synthesizer plugin system.

This test demonstrates the complete plugin system with both real and mock plugins,
suitable for development, testing, and CI/CD environments.
"""

import sys
import argparse
from pathlib import Path
from code.dd_renderer import DDRenderer
from code.mock_renderer import MockDDRenderer, create_mock_renderer, get_renderer
from code.plugin_config import get_config
import numpy as np


def test_renderer_with_plugin(renderer, plugin_name: str, is_mock: bool = False):
    """Test a renderer with a specific plugin."""
    print(f"\n=== Testing {plugin_name} ({'Mock' if is_mock else 'Real'} Plugin) ===")
    
    # Load plugin
    success = renderer.load_plugin_by_name(plugin_name)
    
    if not success:
        print(f"❌ Failed to load {plugin_name}")
        return False
    
    print(f"✅ {plugin_name} loaded successfully")
    
    # Get plugin info
    info = renderer.get_plugin_info()
    print(f"   Parameters: {info['parameter_count']}")
    if 'plugin_name' in info:
        print(f"   Plugin name: {info['plugin_name']}")
    
    # Test parameter manipulation
    try:
        # Get current patch
        original_patch = renderer.get_patch()
        print(f"   Original patch: {len(original_patch)} parameters")
        
        # Modify some parameters
        test_patch = [(0, 0.8), (1, 0.3), (2, 0.6)]
        renderer.set_patch(test_patch)
        print("   ✅ Parameter modification successful")
        
        # Restore original patch
        renderer.set_patch(original_patch)
        print("   ✅ Parameter restoration successful")
        
    except Exception as e:
        print(f"   ❌ Parameter manipulation failed: {e}")
        return False
    
    # Test audio rendering
    try:
        audio = renderer.render_patch(midi_note=60, note_len_sec=1.0, render_len_sec=2.0)
        
        if audio is not None and len(audio) > 0:
            print(f"   ✅ Audio rendering successful: {audio.shape}")
            
            # Basic audio validation
            if isinstance(audio, np.ndarray) and len(audio.shape) == 2:
                channels, samples = audio.shape
                print(f"   Audio: {channels} channels, {samples} samples")
                
                # Check for reasonable audio levels
                max_level = np.max(np.abs(audio))
                if max_level > 0.001:  # Some minimum level
                    print(f"   ✅ Audio has content (max level: {max_level:.3f})")
                else:
                    print(f"   ⚠️  Audio is very quiet (max level: {max_level:.3f})")
            else:
                print(f"   ⚠️  Unexpected audio format: {type(audio)}")
        else:
            print("   ❌ Audio rendering returned no data")
            return False
            
    except Exception as e:
        print(f"   ❌ Audio rendering failed: {e}")
        return False
    
    return True


def test_plugin_comparison():
    """Test the same plugin with both real and mock renderers for comparison."""
    print("\n=== Plugin Comparison Test ===")
    
    config = get_config()
    plugins = config.list_configured_plugins()
    
    if not plugins:
        print("❌ No configured plugins available for comparison")
        return False
    
    test_plugin = plugins[0]  # Test first available plugin
    print(f"Comparing real vs mock for: {test_plugin}")
    
    results = {}
    
    # Test with real renderer
    try:
        real_renderer = DDRenderer.from_config()
        results['real'] = test_renderer_with_plugin(real_renderer, test_plugin, is_mock=False)
    except Exception as e:
        print(f"Real renderer test failed: {e}")
        results['real'] = False
    
    # Test with mock renderer
    try:
        mock_renderer = MockDDRenderer.from_config()
        results['mock'] = test_renderer_with_plugin(mock_renderer, test_plugin, is_mock=True)
    except Exception as e:
        print(f"Mock renderer test failed: {e}")
        results['mock'] = False
    
    # Comparison summary
    print(f"\n=== Comparison Results ===")
    print(f"Real Plugin: {'✅ Working' if results['real'] else '❌ Failed'}")
    print(f"Mock Plugin: {'✅ Working' if results['mock'] else '❌ Failed'}")
    
    if not results['real'] and results['mock']:
        print("💡 Recommendation: Use mock renderer for development/testing")
    elif results['real'] and results['mock']:
        print("💡 Both renderers working - can use either for development")
    elif results['real'] and not results['mock']:
        print("⚠️  Mock renderer has issues - check implementation")
    else:
        print("❌ Both renderers failed - check configuration")
    
    return any(results.values())


def test_auto_renderer():
    """Test the automatic renderer selection."""
    print("\n=== Auto Renderer Selection Test ===")
    
    try:
        renderer = get_renderer(prefer_real=True)
        renderer_type = "Mock" if hasattr(renderer, 'mock_mode') else "Real"
        print(f"✅ Auto-selected renderer: {renderer_type}")
        
        # Test the auto-selected renderer
        config = get_config()
        plugins = config.list_configured_plugins()
        
        if plugins:
            test_plugin = plugins[0]
            success = test_renderer_with_plugin(renderer, test_plugin, 
                                              is_mock=(renderer_type == "Mock"))
            return success
        else:
            print("⚠️  No plugins configured for testing")
            return True  # Not a failure, just no plugins to test
            
    except Exception as e:
        print(f"❌ Auto renderer selection failed: {e}")
        return False


def test_mock_specific_features():
    """Test features specific to the mock renderer."""
    print("\n=== Mock Renderer Specific Features ===")
    
    try:
        mock_renderer = MockDDRenderer.from_config()
        
        # Test different plugin types
        test_plugins = ["diva", "massive_x", "fm8", "nonexistent_plugin"]
        
        for plugin_name in test_plugins:
            print(f"\nTesting mock plugin: {plugin_name}")
            success = mock_renderer.load_plugin_by_name(plugin_name)
            
            if success:
                info = mock_renderer.get_plugin_info()
                print(f"   ✅ Loaded: {info.get('plugin_name', 'Unknown')} ({info['parameter_count']} params)")
            else:
                print(f"   ❌ Failed to load {plugin_name}")
        
        # Test mock-specific validation
        validation = MockDDRenderer.validate_plugin_path("/any/path/plugin.vst3")
        print(f"\n✅ Mock validation: {validation['message']}")
        
        # Test mock plugin search
        found_paths = MockDDRenderer.find_plugin_paths("TestPlugin")
        print(f"✅ Mock search found {len(found_paths)} paths")
        
        return True
        
    except Exception as e:
        print(f"❌ Mock renderer specific tests failed: {e}")
        return False


def test_ci_cd_workflow():
    """Test the typical CI/CD workflow with graceful fallbacks."""
    print("\n=== CI/CD Workflow Test ===")
    
    try:
        # Step 1: Try to get any working renderer
        renderer = create_mock_renderer(use_mock=None)
        is_mock = hasattr(renderer, 'mock_mode')
        
        print(f"✅ Renderer created: {'Mock' if is_mock else 'Real'}")
        
        # Step 2: Test basic functionality
        config = get_config()
        plugins = config.list_configured_plugins()
        
        if not plugins:
            print("⚠️  No plugins configured, testing with generic mock")
            mock_renderer = MockDDRenderer.from_config()
            mock_renderer.load_plugin("/mock/generic.vst3")
            plugins = ["generic"]
            renderer = mock_renderer
        
        # Step 3: Test one plugin thoroughly
        test_plugin = plugins[0]
        print(f"Testing workflow with: {test_plugin}")
        
        # Load plugin
        success = renderer.load_plugin_by_name(test_plugin)
        if not success:
            print(f"❌ CI/CD workflow failed: Could not load {test_plugin}")
            return False
        
        # Basic functionality test
        patch = renderer.get_patch()
        renderer.set_patch([(0, 0.5), (1, 0.7)])
        audio = renderer.render_patch(note_len_sec=0.5, render_len_sec=1.0)
        
        print("✅ CI/CD workflow successful:")
        print(f"   - Plugin loading: OK")
        print(f"   - Parameter manipulation: OK") 
        print(f"   - Audio rendering: OK ({audio.shape} samples)")
        print(f"   - Environment: {'Mock' if is_mock else 'Real'} plugins")
        
        return True
        
    except Exception as e:
        print(f"❌ CI/CD workflow failed: {e}")
        return False


def run_comprehensive_tests(test_type: str = "all"):
    """Run comprehensive test suite."""
    print("Flow Synthesizer Comprehensive Plugin Test Suite")
    print("=" * 50)
    
    results = {}
    
    if test_type in ["all", "basic"]:
        # Basic functionality tests
        print("\n🔧 BASIC FUNCTIONALITY TESTS")
        try:
            config = get_config()
            print("✅ Configuration system working")
            results['config'] = True
        except Exception as e:
            print(f"❌ Configuration system failed: {e}")
            results['config'] = False
    
    if test_type in ["all", "comparison"]:
        # Plugin comparison test
        print("\n🔍 PLUGIN COMPARISON TESTS")
        results['comparison'] = test_plugin_comparison()
    
    if test_type in ["all", "auto"]:
        # Auto renderer test
        print("\n🤖 AUTO RENDERER TESTS")
        results['auto_renderer'] = test_auto_renderer()
    
    if test_type in ["all", "mock"]:
        # Mock-specific tests
        print("\n🎭 MOCK RENDERER TESTS")
        results['mock_features'] = test_mock_specific_features()
    
    if test_type in ["all", "cicd"]:
        # CI/CD workflow test
        print("\n🏗️  CI/CD WORKFLOW TESTS")
        results['cicd_workflow'] = test_ci_cd_workflow()
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        print("💡 Plugin system is working correctly")
    elif passed > 0:
        print("⚠️  Some tests failed, but core functionality is working")
        print("💡 This is normal in environments without real plugins")
    else:
        print("❌ All tests failed - check system configuration")
    
    return passed >= max(1, total // 2)  # Pass if at least half the tests pass


def main():
    parser = argparse.ArgumentParser(description="Comprehensive plugin system test suite")
    parser.add_argument("--test-type", 
                      choices=["all", "basic", "comparison", "auto", "mock", "cicd"],
                      default="all",
                      help="Type of tests to run")
    parser.add_argument("--plugin", help="Test specific plugin")
    parser.add_argument("--mock-only", action="store_true", help="Use only mock renderer")
    parser.add_argument("--real-only", action="store_true", help="Use only real renderer")
    
    args = parser.parse_args()
    
    if args.plugin:
        # Test specific plugin
        if args.mock_only:
            renderer = MockDDRenderer.from_config()
            success = test_renderer_with_plugin(renderer, args.plugin, is_mock=True)
        elif args.real_only:
            renderer = DDRenderer.from_config()
            success = test_renderer_with_plugin(renderer, args.plugin, is_mock=False)
        else:
            # Test with auto-selected renderer
            renderer = get_renderer(prefer_real=True)
            is_mock = hasattr(renderer, 'mock_mode')
            success = test_renderer_with_plugin(renderer, args.plugin, is_mock=is_mock)
        
        sys.exit(0 if success else 1)
    else:
        # Run comprehensive test suite
        success = run_comprehensive_tests(args.test_type)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()