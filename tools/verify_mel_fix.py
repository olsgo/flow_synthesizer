#!/usr/bin/env python3
"""
Verify that the regenerated mel spectrograms no longer contain NaN values
"""

import numpy as np
import glob
from pathlib import Path

# Check a few mel spectrogram files
mel_dir = Path('datasets/polymax_dataset/mel')
mel_files = list(mel_dir.glob('*.npy'))[:10]  # Check first 10 files

print(f"Checking {len(mel_files)} mel spectrogram files...")

for mel_file in mel_files:
    mel_data = np.load(mel_file)
    
    # Check for NaN and Inf values
    nan_count = np.isnan(mel_data).sum()
    inf_count = np.isinf(mel_data).sum()
    
    # Check data range
    min_val = np.min(mel_data)
    max_val = np.max(mel_data)
    
    print(f"{mel_file.name}:")
    print(f"  Shape: {mel_data.shape}")
    print(f"  NaN count: {nan_count}")
    print(f"  Inf count: {inf_count}")
    print(f"  Min/Max: {min_val:.6f}/{max_val:.6f}")
    print(f"  All finite: {np.isfinite(mel_data).all()}")
    print()

print("Verification complete!")