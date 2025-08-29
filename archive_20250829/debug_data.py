#!/usr/bin/env python3

import torch
import numpy as np
from utils.data import load_dataset
import argparse

# Create minimal args for testing
args = argparse.Namespace(
    path='datasets',
    dataset='polymax_dataset',
    data='mel',
    train_type='fixed',
    nbworkers=0,
    device='cpu',
    batch_size=4,
    kernel=5,
    dilation=3
)

print("Loading dataset...")
train_loader, valid_loader, test_loader, args = load_dataset(args)

print("\nChecking first 5 batches for NaN/Inf values...")
for batch_idx, (x, y, _, _) in enumerate(train_loader):
    if batch_idx >= 5:
        break
    
    print(f"\nBatch {batch_idx}:")
    print(f"  Input shape: {x.shape}")
    print(f"  Input min/max: {x.min().item():.6f}/{x.max().item():.6f}")
    print(f"  Input NaN count: {torch.isnan(x).sum().item()}")
    print(f"  Input Inf count: {torch.isinf(x).sum().item()}")
    print(f"  Target shape: {y.shape}")
    print(f"  Target min/max: {y.min().item():.6f}/{y.max().item():.6f}")
    print(f"  Target NaN count: {torch.isnan(y).sum().item()}")
    print(f"  Target Inf count: {torch.isinf(y).sum().item()}")
    
    if torch.isnan(x).any():
        # Find which samples have NaN
        nan_samples = torch.isnan(x).any(dim=(1,2,3))
        print(f"  Samples with NaN: {nan_samples.nonzero().flatten().tolist()}")
        
        # Check the first sample with NaN
        first_nan_idx = nan_samples.nonzero()[0].item()
        sample = x[first_nan_idx]
        print(f"  Sample {first_nan_idx} NaN locations: {torch.isnan(sample).sum().item()} out of {sample.numel()} values")
        
        # Check if entire sample is NaN or just parts
        if torch.isnan(sample).all():
            print(f"  Sample {first_nan_idx}: ENTIRE sample is NaN")
        else:
            print(f"  Sample {first_nan_idx}: Partial NaN - {torch.isnan(sample).sum().item()}/{sample.numel()} values")

print("\nChecking raw data files directly...")
# Load a few raw mel files to check if NaN is in the source data
import os
mel_dir = 'datasets/polymax_dataset/mel'
mel_files = [f for f in os.listdir(mel_dir) if f.endswith('.npy')][:5]

for mel_file in mel_files:
    mel_path = os.path.join(mel_dir, mel_file)
    mel_data = np.load(mel_path)
    
    print(f"\nFile: {mel_file}")
    print(f"  Shape: {mel_data.shape}")
    print(f"  Min/Max: {mel_data.min():.6f}/{mel_data.max():.6f}")
    print(f"  NaN count: {np.isnan(mel_data).sum()}")
    print(f"  Inf count: {np.isinf(mel_data).sum()}")
    
    if np.isnan(mel_data).any():
        print(f"  *** NaN found in raw file {mel_file}! ***")
    if np.isinf(mel_data).any():
        print(f"  *** Inf found in raw file {mel_file}! ***")

print("\nData debugging complete.")