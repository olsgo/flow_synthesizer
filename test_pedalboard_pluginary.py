#!/usr/bin/env python3
"""
Test script for pedalboard-pluginary to scan for audio plugins including Serum 2.
"""

import os
from pedalboard_pluginary import PedalboardPluginary

def main():
    print("Initializing Pedalboard Pluginary...")
    
    try:
        # Initialize the pluginary
        pluginary = PedalboardPluginary()
        
        print("Scanning for plugins...")
        # Scan for plugins
        plugins = pluginary.scan()
        
        print(f"Found {len(plugins)} plugins:")
        print("-" * 50)
        
        # Look specifically for Serum 2
        serum_plugins = []
        
        for plugin in plugins:
            plugin_name = plugin.get('name', 'Unknown')
            plugin_path = plugin.get('path', 'Unknown')
            plugin_format = plugin.get('format', 'Unknown')
            
            print(f"Name: {plugin_name}")
            print(f"Path: {plugin_path}")
            print(f"Format: {plugin_format}")
            print("-" * 30)
            
            # Check if this is Serum 2
            if 'serum' in plugin_name.lower() or 'serum2' in plugin_name.lower():
                serum_plugins.append(plugin)
        
        if serum_plugins:
            print(f"\n🎉 Found {len(serum_plugins)} Serum plugin(s):")
            for serum in serum_plugins:
                print(f"  - {serum.get('name', 'Unknown')} at {serum.get('path', 'Unknown')}")
        else:
            print("\n❌ No Serum plugins found.")
            print("Expected location: /Library/Audio/Plug-Ins/VST3/Serum2.vst3")
            
            # Check if Serum 2 exists at expected location
            serum2_path = "/Library/Audio/Plug-Ins/VST3/Serum2.vst3"
            if os.path.exists(serum2_path):
                print(f"✅ Serum 2 VST3 found at: {serum2_path}")
            else:
                print(f"❌ Serum 2 VST3 not found at: {serum2_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTrying alternative approach...")
        
        # Alternative: Just check if Serum 2 exists
        serum2_path = "/Library/Audio/Plug-Ins/VST3/Serum2.vst3"
        if os.path.exists(serum2_path):
            print(f"✅ Serum 2 VST3 found at: {serum2_path}")
        else:
            print(f"❌ Serum 2 VST3 not found at: {serum2_path}")

if __name__ == "__main__":
    main()