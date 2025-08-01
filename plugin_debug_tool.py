#!/usr/bin/env python3
"""
Plugin Debug Tool for Flow Synthesizer

This tool helps debug plugin loading issues and provides information about
available plugins on the system.
"""

import sys
import argparse
from pathlib import Path
from code.dd_renderer import DDRenderer
from code.plugin_config import get_config, reload_config


def test_plugin_path(plugin_path: str, verbose: bool = False):
    """Test loading a specific plugin path."""
    print(f"\n=== Testing Plugin: {plugin_path} ===")
    
    # Validate path first
    validation = DDRenderer.validate_plugin_path(plugin_path)
    print(f"Path validation: {validation['message']}")
    print(f"Plugin type: {validation['type']}")
    print(f"Resolved path: {validation['path']}")
    
    if not validation['valid']:
        print("❌ Path validation failed - cannot proceed with loading test")
        return False
    
    # Try to load the plugin
    print("\nAttempting to load plugin...")
    renderer = DDRenderer(sample_rate=22050, block_size=512)
    success = renderer.load_plugin(plugin_path)
    
    if success:
        print("✅ Plugin loaded successfully!")
        
        # Get plugin info
        info = renderer.get_plugin_info()
        print(f"Parameter count: {info['parameter_count']}")
        print(f"Total parameters: {info['total_parameters']}")
        
        if verbose and info['parameter_names']:
            print("First 10 parameters:")
            for i, name in enumerate(info['parameter_names']):
                print(f"  {i}: {name}")
        
        # Test basic functionality
        try:
            patch = renderer.get_patch()
            print(f"✅ Parameter access working (got {len(patch)} parameter values)")
        except Exception as e:
            print(f"❌ Parameter access failed: {e}")
        
        return True
    else:
        print("❌ Plugin loading failed")
        return False


def search_plugins(plugin_name: str):
    """Search for plugins by name."""
    print(f"\n=== Searching for plugin: {plugin_name} ===")
    
    found_paths = DDRenderer.find_plugin_paths(plugin_name)
    
    if not found_paths:
        print("❌ No plugins found")
        return []
    
    print(f"✅ Found {len(found_paths)} potential plugin(s):")
    for i, path in enumerate(found_paths):
        validation = DDRenderer.validate_plugin_path(path)
        status = "✅" if validation['valid'] else "❌"
        print(f"  {i+1}. {status} {path} ({validation['type']})")
    
    return found_paths


def test_configured_plugin(plugin_name: str, verbose: bool = False):
    """Test loading a plugin using the configuration system."""
    print(f"\n=== Testing Configured Plugin: {plugin_name} ===")
    
    config = get_config()
    paths = config.get_plugin_paths(plugin_name)
    
    if not paths:
        print(f"❌ Plugin '{plugin_name}' not found in configuration")
        print(f"Available configured plugins: {config.list_configured_plugins()}")
        return False
    
    print(f"Found {len(paths)} configured path(s):")
    for format_type, path in paths.items():
        validation = DDRenderer.validate_plugin_path(path)
        status = "✅" if validation['valid'] else "❌"
        print(f"  {format_type}: {status} {path}")
    
    # Try to load using the preferred path
    renderer = DDRenderer.from_config()
    success = renderer.load_plugin_by_name(plugin_name)
    
    if success:
        print("✅ Plugin loaded successfully using configuration!")
        
        # Get plugin info
        info = renderer.get_plugin_info()
        print(f"Parameter count: {info['parameter_count']}")
        print(f"Total parameters: {info['total_parameters']}")
        
        if verbose and info['parameter_names']:
            print("First 10 parameters:")
            for i, name in enumerate(info['parameter_names']):
                print(f"  {i}: {name}")
        
        return True
    else:
        print("❌ Failed to load plugin using configuration")
        return False


def list_configured_plugins():
    """List all plugins configured in the system."""
    print("\n=== Configured Plugins ===")
    
    config = get_config()
    plugins = config.list_configured_plugins()
    
    if not plugins:
        print("❌ No plugins configured")
        print("Create a plugin_config.yml file to configure plugin paths")
        return
    
    print(f"Found {len(plugins)} configured plugin(s):")
    for plugin_name in plugins:
        paths = config.get_plugin_paths(plugin_name)
        preferred = config.get_preferred_plugin_path(plugin_name)
        
        print(f"\n{plugin_name}:")
        for format_type, path in paths.items():
            validation = DDRenderer.validate_plugin_path(path)
            status = "✅" if validation['valid'] else "❌"
            preferred_mark = " (preferred)" if path == preferred else ""
            print(f"  {format_type}: {status} {path}{preferred_mark}")


def test_common_plugins():
    """Test loading of commonly used plugins."""
    common_plugins = [
        "Diva",
        "Massive X", 
        "FM8",
        "uaudio_polymax",
        "Serum",
        "Sylenth1"
    ]
    
    print("\n=== Testing Common Plugins ===")
    
    results = {}
    for plugin_name in common_plugins:
        print(f"\nSearching for {plugin_name}...")
        found_paths = DDRenderer.find_plugin_paths(plugin_name)
        
        if found_paths:
            # Test the first found path
            success = test_plugin_path(found_paths[0], verbose=False)
            results[plugin_name] = {"found": True, "loadable": success, "path": found_paths[0]}
        else:
            print(f"❌ {plugin_name} not found on system")
            results[plugin_name] = {"found": False, "loadable": False, "path": None}
    
    # Summary
    print("\n=== Summary ===")
    for plugin_name, result in results.items():
        if result['found'] and result['loadable']:
            status = "✅ Available"
        elif result['found']:
            status = "⚠️  Found but not loadable"
        else:
            status = "❌ Not found"
        print(f"{plugin_name:15} {status}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Debug audio plugin loading issues")
    parser.add_argument("--test-path", help="Test loading a specific plugin path")
    parser.add_argument("--test-plugin", help="Test loading a configured plugin by name")
    parser.add_argument("--search", help="Search for plugins by name")
    parser.add_argument("--list-configured", action="store_true", help="List all configured plugins")
    parser.add_argument("--test-common", action="store_true", help="Test common plugins")
    parser.add_argument("--config", help="Use specific configuration file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Load configuration if specified
    if args.config:
        reload_config(args.config)
    
    if not any([args.test_path, args.test_plugin, args.search, args.list_configured, args.test_common]):
        # Default: list configured plugins and run common tests
        args.list_configured = True
        args.test_common = True
    
    if args.list_configured:
        list_configured_plugins()
    
    if args.test_plugin:
        test_configured_plugin(args.test_plugin, args.verbose)
    
    if args.test_path:
        test_plugin_path(args.test_path, args.verbose)
    
    if args.search:
        found_paths = search_plugins(args.search)
        if found_paths and args.verbose:
            for path in found_paths:
                test_plugin_path(path, args.verbose)
    
    if args.test_common:
        test_common_plugins()


if __name__ == "__main__":
    main()