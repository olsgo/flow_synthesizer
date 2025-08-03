#!/usr/bin/env python3
"""
Capture initial plugin states for a list of plugins.

Usage:
    python capture_init_state.py --plugins Serum.vst3 Diva.vst3
    python capture_init_state.py --plugin_dir /Library/Audio/Plug-Ins/VST3/
"""

import argparse
import os
import sys
from pathlib import Path

# Add the code directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))

from pedalboard_renderer import PBRenderer

def find_plugin_path(plugin_name, search_dirs=None):
    """
    Find the full path to a plugin by searching common directories.
    
    Args:
        plugin_name: Name of the plugin (e.g., "Serum.vst3")
        search_dirs: List of directories to search
    
    Returns:
        str: Full path to plugin or None if not found
    """
    if search_dirs is None:
        # Common plugin directories on different platforms
        search_dirs = [
            "/Library/Audio/Plug-Ins/VST3",          # macOS VST3
            "/Library/Audio/Plug-Ins/Components",    # macOS AU
            "/usr/lib/vst3",                         # Linux VST3
            "/usr/local/lib/vst3",                   # Linux VST3
            "C:\\Program Files\\Common Files\\VST3", # Windows VST3
            "C:\\Program Files\\Steinberg\\VstPlugins", # Windows VST2
        ]
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for root, dirs, files in os.walk(search_dir):
                for item in dirs + files:
                    if item.lower() == plugin_name.lower():
                        return os.path.join(root, item)
    
    return None

def capture_init_state(plugin_path, output_dir="."):
    """
    Capture the initial state of a plugin and save it as a .bin file.
    
    Args:
        plugin_path: Path to the plugin
        output_dir: Directory to save the init state file
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"Processing plugin: {plugin_path}")
    
    # Extract plugin name for output file
    plugin_name = Path(plugin_path).stem
    output_file = Path(output_dir) / f"{plugin_name}_init.bin"
    
    try:
        # Create renderer and load plugin
        renderer = PBRenderer(sample_rate=22050, buffer_size=512)
        
        if not renderer.load_plugin(plugin_path):
            print(f"❌ Failed to load plugin: {plugin_path}")
            return False
        
        print(f"✓ Loaded plugin: {plugin_name}")
        
        # Get parameter info
        params = renderer.get_parameters_description()
        print(f"  Parameters: {len(params)}")
        
        # Save initial state
        if renderer.save_state(str(output_file)):
            print(f"✓ Saved initial state to: {output_file}")
            return True
        else:
            print(f"❌ Failed to save state to: {output_file}")
            return False
            
    except Exception as e:
        print(f"❌ Error processing {plugin_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Capture initial states of audio plugins")
    parser.add_argument("--plugins", nargs="+", help="List of plugin names to process")
    parser.add_argument("--plugin_dir", help="Directory containing plugins to process")
    parser.add_argument("--search_dirs", nargs="+", help="Directories to search for plugins")
    parser.add_argument("--output_dir", default=".", help="Output directory for .bin files")
    
    args = parser.parse_args()
    
    if not args.plugins and not args.plugin_dir:
        print("Error: Must specify either --plugins or --plugin_dir")
        parser.print_help()
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    total_count = 0
    
    if args.plugins:
        # Process specific plugins by name
        for plugin_name in args.plugins:
            total_count += 1
            
            # Check if it's already a full path
            if os.path.exists(plugin_name):
                plugin_path = plugin_name
            else:
                # Search for the plugin
                plugin_path = find_plugin_path(plugin_name, args.search_dirs)
                if not plugin_path:
                    print(f"❌ Plugin not found: {plugin_name}")
                    continue
            
            if capture_init_state(plugin_path, output_dir):
                success_count += 1
    
    if args.plugin_dir:
        # Process all plugins in a directory
        plugin_dir = Path(args.plugin_dir)
        if not plugin_dir.exists():
            print(f"❌ Plugin directory not found: {plugin_dir}")
            return 1
        
        # Find all plugin files
        plugin_extensions = [".vst3", ".component", ".vst", ".dll"]
        
        for plugin_file in plugin_dir.rglob("*"):
            if plugin_file.is_file() or (plugin_file.is_dir() and plugin_file.suffix in plugin_extensions):
                total_count += 1
                if capture_init_state(str(plugin_file), output_dir):
                    success_count += 1
    
    print(f"\n📊 Summary: {success_count}/{total_count} plugins processed successfully")
    
    if success_count > 0:
        print(f"✓ Initial state files saved to: {output_dir}")
    
    return 0 if success_count > 0 else 1

if __name__ == "__main__":
    sys.exit(main())