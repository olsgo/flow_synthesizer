#!/usr/bin/env python3
"""
Copy parameter JSON files to the render folder to organize them alongside audio files.
"""

import os
import shutil
import json
from pathlib import Path

def copy_params_to_render():
    """Copy parameter JSON files from params/ to render folder."""
    
    # Define paths
    params_dir = Path('/Users/gjb/Projects/flow_synthesizer/params')
    render_dir = Path('/Users/gjb/Datasets/polymax/render')
    
    # Create render directory if it doesn't exist
    render_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all parameter files
    param_files = list(params_dir.glob('*.json'))
    
    print(f"Found {len(param_files)} parameter files to copy")
    
    copied_count = 0
    for param_file in param_files:
        try:
            # Copy to render directory
            dest_path = render_dir / param_file.name
            shutil.copy2(param_file, dest_path)
            copied_count += 1
            
            if copied_count <= 5 or copied_count % 50 == 0:
                print(f"Copied: {param_file.name}")
                
        except Exception as e:
            print(f"Error copying {param_file.name}: {e}")
    
    print(f"\nSuccessfully copied {copied_count} parameter files to {render_dir}")
    
    # Verify the copy operation
    render_param_files = list(render_dir.glob('*.json'))
    print(f"Verification: {len(render_param_files)} parameter files now in render directory")
    
    # Show a sample of what's in the render directory
    print("\nSample files in render directory:")
    all_files = list(render_dir.iterdir())
    for i, file_path in enumerate(sorted(all_files)[:10]):
        file_type = "audio" if file_path.suffix == ".wav" else "params"
        print(f"  {file_path.name} ({file_type})")
    
    if len(all_files) > 10:
        print(f"  ... and {len(all_files) - 10} more files")

if __name__ == "__main__":
    copy_params_to_render()