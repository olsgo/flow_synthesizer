#!/usr/bin/env python3
"""
Test the normalization computation to debug NaN issues
"""

import sys
sys.path.append('/Users/gjb/Projects/flow_synthesizer/code')

import numpy as np
import torch
from utils.data import SpecData, LogTransform
import os

def test_normalization():
    print("=== Testing Normalization Computation ===")
    
    # Create a SpecData instance for mel data
    datadir = '/Users/gjb/Projects/flow_synthesizer/datasets/polymax_dataset'
    data_type = 'mel'
    
    # Get list of spectral files
    import glob
    spectral_files = sorted(glob.glob(os.path.join(datadir, data_type, '*.npy')))
    spectral_files = [os.path.basename(f) for f in spectral_files]  # Get just filenames
    
    # Create SpecData instance
    spec_data = SpecData(datadir, spectral_files, data_type=data_type)
    
    print(f"Found {len(spec_data.spectral_files)} mel files")
    
    # Test the normalization computation
    print("\nComputing normalization...")
    spec_data.compute_normalization()
    
    print(f"\nNormalization results:")
    print(f"Mean: {spec_data.mean}")
    print(f"Std: {spec_data.var}")
    
    # Test loading and normalizing a sample
    print("\nTesting sample normalization...")
    if len(spec_data.spectral_files) > 0:
        sample_file = spec_data.spectral_files[0]
        sample_path = os.path.join(datadir, data_type, sample_file)
        
        # Load raw data
        raw_data = np.load(sample_path)
        print(f"Raw data shape: {raw_data.shape}")
        print(f"Raw data range: [{raw_data.min():.6f}, {raw_data.max():.6f}]")
        
        # Apply log transform
        tr = LogTransform(clip=1e-3)
        log_data = tr(torch.from_numpy(raw_data).float())
        print(f"Log data range: [{log_data.min():.6f}, {log_data.max():.6f}]")
        print(f"Log data NaN count: {torch.isnan(log_data).sum()}")
        print(f"Log data Inf count: {torch.isinf(log_data).sum()}")
        
        # Apply normalization
        normalized_data = (log_data - spec_data.mean) / spec_data.var
        print(f"Normalized data range: [{normalized_data.min():.6f}, {normalized_data.max():.6f}]")
        print(f"Normalized data NaN count: {torch.isnan(normalized_data).sum()}")
        print(f"Normalized data Inf count: {torch.isinf(normalized_data).sum()}")
        
        # Check if normalization produces reasonable values
        if torch.isnan(normalized_data).any() or torch.isinf(normalized_data).any():
            print("\n❌ PROBLEM: Normalization produces NaN/Inf values!")
            print(f"Mean: {spec_data.mean}, Std: {spec_data.var}")
            if spec_data.var == 0 or spec_data.var < 1e-8:
                print("Issue: Standard deviation is too small or zero")
        else:
            print("\n✅ Normalization looks good!")

if __name__ == '__main__':
    test_normalization()