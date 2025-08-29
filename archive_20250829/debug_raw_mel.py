#!/usr/bin/env python3

import numpy as np
import torch
import os

def debug_raw_mel_files():
    print("=== Debugging Raw Mel Spectrogram Files ===")
    
    mel_dir = "datasets/polymax_dataset/mel"
    
    # Check a few mel files directly
    mel_files = sorted(os.listdir(mel_dir))[:5]
    
    for i, filename in enumerate(mel_files):
        if filename.endswith('.npy'):
            filepath = os.path.join(mel_dir, filename)
            print(f"\nFile {i}: {filename}")
            
            # Load raw numpy array
            data = np.load(filepath, allow_pickle=True)
            print(f"  Raw shape: {data.shape}")
            print(f"  Raw dtype: {data.dtype}")
            print(f"  Raw min/max: {np.nanmin(data):.6f}/{np.nanmax(data):.6f}")
            print(f"  Raw mean/std: {np.nanmean(data):.6f}/{np.nanstd(data):.6f}")
            print(f"  NaN count: {np.isnan(data).sum()}")
            print(f"  Inf count: {np.isinf(data).sum()}")
            print(f"  Zero count: {(data == 0).sum()}")
            print(f"  Negative count: {(data < 0).sum()}")
            
            # Check if there are any finite values
            finite_mask = np.isfinite(data)
            finite_count = finite_mask.sum()
            print(f"  Finite values count: {finite_count} / {data.size}")
            
            if finite_count > 0:
                finite_data = data[finite_mask]
                print(f"  Finite min/max: {finite_data.min():.6f}/{finite_data.max():.6f}")
                print(f"  Finite mean/std: {finite_data.mean():.6f}/{finite_data.std():.6f}")
            
            # Check specific locations
            print(f"  First few values: {data.flat[:10]}")
            print(f"  Last few values: {data.flat[-10:]}")
            
            # Check if the entire array is NaN
            if np.isnan(data).all():
                print(f"  WARNING: Entire array is NaN!")
            elif np.isnan(data).any():
                print(f"  WARNING: Array contains some NaN values!")
                # Find where NaNs start
                nan_indices = np.where(np.isnan(data))
                print(f"  First NaN at: {nan_indices[0][0] if len(nan_indices[0]) > 0 else 'None'}, {nan_indices[1][0] if len(nan_indices[1]) > 0 else 'None'}")

if __name__ == "__main__":
    debug_raw_mel_files()