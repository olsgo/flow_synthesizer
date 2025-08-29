#!/usr/bin/env python3

import torch
import numpy as np
from utils.data import CompSynthesizerDataset
from utils.transforms import LogTransform, NormalizeTensor
import os

def debug_normalization():
    print("=== Debugging Normalization Statistics ===")
    
    # Create dataset
    dataset = CompSynthesizerDataset(
        datadir='datasets/polymax_dataset',
        use_params=['osc1_level', 'osc2_level', 'filter_cutoff', 'filter_resonance', 'env_attack', 'env_decay'],
        data='mel'
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Check if spectral files exist
    print(f"Spectral files object: {dataset.spectral_files}")
    print(f"Trans datasets object: {dataset.trans_datasets}")
    
    # Access the SpecData object directly
    if 'mel' in dataset.trans_datasets:
        spec_dataset = dataset.trans_datasets['mel']
        print(f"\nSpecData object found for 'mel'")
        print(f"SpecData spectral files count: {len(spec_dataset.spectral_files)}")
        print(f"SpecData first few files: {spec_dataset.spectral_files[:5]}")
        
        # Check the computed normalization statistics
        print(f"\nComputed normalization statistics:")
        print(f"Mean: {spec_dataset.mean}")
        print(f"Std: {spec_dataset.var}")
        
        # Check if mean or std are NaN or problematic
        if np.isnan(spec_dataset.mean) or np.isnan(spec_dataset.var):
            print("ERROR: NaN values in normalization statistics!")
        
        if spec_dataset.var == 0 or abs(spec_dataset.var) < 1e-10:
            print("ERROR: Zero or very small standard deviation!")
    else:
        print("ERROR: No mel trans_dataset found!")
        return
    
    # Check if mean or std are NaN or problematic
    if np.isnan(spec_dataset.mean) or np.isnan(spec_dataset.var):
        print("ERROR: NaN values in normalization statistics!")
    
    if spec_dataset.var == 0 or abs(spec_dataset.var) < 1e-10:
        print("ERROR: Zero or very small standard deviation!")
    
    # Manually compute correct statistics
    print("\n=== Manual Statistics Computation ===")
    
    log_transform = LogTransform(clip=1e-3)
    all_data = []
    
    # Load all mel spectrograms
    for i in range(min(10, len(spec_dataset.spectral_files))):
        file_path = f"datasets/polymax_dataset/mel/{spec_dataset.spectral_files[i]}"
        data = np.load(file_path, allow_pickle=True)
        data = torch.from_numpy(data).float()
        data = log_transform(data)  # Apply log transform
        all_data.append(data.flatten())
        
        print(f"File {i}: {spec_dataset.spectral_files[i]}")
        print(f"  Shape: {data.shape}")
        print(f"  Min/Max: {data.min():.4f}/{data.max():.4f}")
        print(f"  Mean/Std: {data.mean():.4f}/{data.std():.4f}")
        print(f"  NaN count: {torch.isnan(data).sum()}")
        print(f"  Inf count: {torch.isinf(data).sum()}")
    
    # Compute overall statistics
    if all_data:
        all_data_tensor = torch.cat(all_data)
        correct_mean = all_data_tensor.mean()
        correct_std = all_data_tensor.std()
        
        print(f"\nCorrect statistics (from {len(all_data)} files):")
        print(f"Mean: {correct_mean:.6f}")
        print(f"Std: {correct_std:.6f}")
        
        # Test normalization with correct stats
        print("\n=== Testing Normalization ===")
        test_data = all_data[0][:100]  # First 100 elements
        normalized = (test_data - correct_mean) / correct_std
        print(f"Original data range: {test_data.min():.4f} to {test_data.max():.4f}")
        print(f"Normalized data range: {normalized.min():.4f} to {normalized.max():.4f}")
        print(f"Normalized mean/std: {normalized.mean():.6f}/{normalized.std():.6f}")
        print(f"NaN count in normalized: {torch.isnan(normalized).sum()}")
    
    # Test a single sample from the dataset
    print("\n=== Testing Dataset Sample ===")
    try:
        sample = dataset[0]
        data, params, meta, audio = sample
        print(f"Sample data shape: {data.shape}")
        print(f"Sample data range: {data.min():.4f} to {data.max():.4f}")
        print(f"Sample NaN count: {torch.isnan(data).sum()}")
        print(f"Sample Inf count: {torch.isinf(data).sum()}")
    except Exception as e:
        print(f"Error loading sample: {e}")

if __name__ == "__main__":
    debug_normalization()