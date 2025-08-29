#!/usr/bin/env python3

import sys
sys.path.append('code')

import torch
import numpy as np
from utils.data import load_dataset
import argparse

# Create args similar to training
args = argparse.Namespace(
    path='datasets',
    dataset='polymax_dataset',
    data='mel',
    batch_size=32,
    nbworkers=0,
    train_type='sequential'
)

print("Loading dataset...")
train_loader, valid_loader, test_loader, args = load_dataset(args)

print(f"Dataset loaded successfully")
print(f"Input size: {args.input_size}")
print(f"Output size: {args.output_size}")

# Get a batch from train loader
print("\nGetting first batch...")
data_batch, params_batch, meta_batch, audio_batch = next(iter(train_loader))

print(f"Data batch shape: {data_batch.shape}")
print(f"Data batch dtype: {data_batch.dtype}")
print(f"Data batch min: {data_batch.min().item():.6f}")
print(f"Data batch max: {data_batch.max().item():.6f}")
print(f"Data batch mean: {data_batch.mean().item():.6f}")
print(f"Data batch std: {data_batch.std().item():.6f}")

# Check for NaN/Inf values
nan_count = torch.isnan(data_batch).sum().item()
inf_count = torch.isinf(data_batch).sum().item()
print(f"NaN values in data: {nan_count}")
print(f"Inf values in data: {inf_count}")

print(f"\nParams batch shape: {params_batch.shape}")
print(f"Params batch dtype: {params_batch.dtype}")
print(f"Params batch min: {params_batch.min().item():.6f}")
print(f"Params batch max: {params_batch.max().item():.6f}")
print(f"Params batch mean: {params_batch.mean().item():.6f}")
print(f"Params batch std: {params_batch.std().item():.6f}")

# Check for NaN/Inf values in params
nan_count_params = torch.isnan(params_batch).sum().item()
inf_count_params = torch.isinf(params_batch).sum().item()
print(f"NaN values in params: {nan_count_params}")
print(f"Inf values in params: {inf_count_params}")

# Check individual samples
print("\nChecking individual samples:")
for i in range(min(3, data_batch.shape[0])):
    sample_data = data_batch[i]
    sample_params = params_batch[i]
    
    sample_nan = torch.isnan(sample_data).sum().item()
    sample_inf = torch.isinf(sample_data).sum().item()
    params_nan = torch.isnan(sample_params).sum().item()
    params_inf = torch.isinf(sample_params).sum().item()
    
    print(f"Sample {i}: data NaN={sample_nan}, data Inf={sample_inf}, params NaN={params_nan}, params Inf={params_inf}")
    print(f"  Data range: [{sample_data.min().item():.6f}, {sample_data.max().item():.6f}]")
    print(f"  Params range: [{sample_params.min().item():.6f}, {sample_params.max().item():.6f}]")

# Check the dataset's normalization stats
print("\nDataset normalization stats:")
for dtype in train_loader.dataset.data:
    mean = train_loader.dataset.means[dtype]
    var = train_loader.dataset.vars[dtype]
    print(f"{dtype}: mean={mean:.6f}, var={var:.6f}, std={np.sqrt(var):.6f}")

print("\nDebug complete.")