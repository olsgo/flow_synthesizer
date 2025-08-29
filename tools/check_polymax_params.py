#!/usr/bin/env python3
"""
Check actual parameter count exposed by PolyMAX VST3 plugin
"""

from pedalboard import load_plugin

def main():
    try:
        plugin = load_plugin('/Library/Audio/Plug-Ins/VST3/uaudio_polymax.vst3')
        
        print(f"Total parameters exposed by PolyMAX: {len(plugin.parameters)}")
        print("\nFirst 20 parameters:")
        
        for i, (name, param) in enumerate(list(plugin.parameters.items())[:20]):
            print(f"  {i:3d}: {name}")
            
        if len(plugin.parameters) > 20:
            print(f"\n... and {len(plugin.parameters) - 20} more parameters")
            
        print(f"\nLast 10 parameters:")
        for i, (name, param) in enumerate(list(plugin.parameters.items())[-10:]):
            actual_index = len(plugin.parameters) - 10 + i
            print(f"  {actual_index:3d}: {name}")
            
    except Exception as e:
        print(f"Error loading plugin: {e}")

if __name__ == "__main__":
    main()