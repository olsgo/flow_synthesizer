#!/usr/bin/env python3
"""
Check the actual dimensions of mel spectrograms in the PolyMAX dataset
"""

import numpy as np
from pathlib import Path

# Load a sample mel spectrogram to check dimensions
mel_dir = Path('/Users/gjb/Projects/flow_synthesizer/datasets/polymax_dataset/mel')
mel_files = list(mel_dir.glob('*.npy'))

if mel_files:
    sample_mel = np.load(mel_files[0])
    print(f"Sample mel spectrogram shape: {sample_mel.shape}")
    print(f"Total elements: {sample_mel.size}")
    
    # Check a few more files to ensure consistency
    print("\nChecking consistency across files:")
    for i, mel_file in enumerate(mel_files[:5]):
        mel_data = np.load(mel_file)
        print(f"{mel_file.name}: {mel_data.shape}")
        
    # Calculate what the flattened size would be
    flattened_size = sample_mel.size
    print(f"\nFlattened size: {flattened_size}")
    print(f"Expected CNN output for this size: {flattened_size}")
else:
    print("No mel files found!")