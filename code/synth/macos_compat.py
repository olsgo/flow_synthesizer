#!/usr/bin/env python3
"""
macOS compatibility utilities for Apple Silicon (M1/M2/M3) and Intel Macs.
"""

import os
import platform

def setup_macos_library_paths():
    """
    Setup library paths for macOS, handling both Intel and Apple Silicon architectures.
    """
    if platform.system() != 'Darwin':
        return  # Not macOS, skip
    
    # Define potential library paths
    library_paths = []
    
    # Apple Silicon (M1/M2/M3) - Homebrew typically installs to /opt/homebrew
    if platform.machine() == 'arm64':
        library_paths.extend([
            '/opt/homebrew/lib',
            '/opt/homebrew/local/lib'
        ])
    
    # Intel Mac - Homebrew typically installs to /usr/local
    library_paths.extend([
        '/usr/local/lib',
        '/usr/local/local/lib'
    ])
    
    # Add paths that actually exist
    existing_paths = [path for path in library_paths if os.path.exists(path)]
    
    # Get current DYLD_LIBRARY_PATH
    current_path = os.environ.get('DYLD_LIBRARY_PATH', '')
    
    # Add existing current path to the list
    if current_path:
        existing_paths.append(current_path)
    
    # Set the updated DYLD_LIBRARY_PATH
    if existing_paths:
        os.environ['DYLD_LIBRARY_PATH'] = ':'.join(existing_paths)
        
def get_architecture_info():
    """
    Get information about the current macOS architecture.
    """
    if platform.system() != 'Darwin':
        return None
    
    return {
        'system': platform.system(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'is_apple_silicon': platform.machine() == 'arm64',
        'is_intel': platform.machine() == 'x86_64'
    }

def print_environment_info():
    """
    Print environment information for debugging.
    """
    print("=== macOS Environment Information ===")
    arch_info = get_architecture_info()
    if arch_info:
        print(f"System: {arch_info['system']}")
        print(f"Machine: {arch_info['machine']}")
        print(f"Processor: {arch_info['processor']}")
        print(f"Apple Silicon: {arch_info['is_apple_silicon']}")
        print(f"Intel: {arch_info['is_intel']}")
    else:
        print("Not running on macOS")
    
    print(f"DYLD_LIBRARY_PATH: {os.environ.get('DYLD_LIBRARY_PATH', 'Not set')}")
    print("=====================================")

if __name__ == "__main__":
    print_environment_info()
    setup_macos_library_paths()
    print("\nAfter setup:")
    print_environment_info()