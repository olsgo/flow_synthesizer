#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
from models.basic import GatedCNN
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
    batch_size=4,  # smaller batch for easier debugging
    kernel=5,
    dilation=3
)

print("Loading dataset...")
train_loader, valid_loader, test_loader, args = load_dataset(args)

# Get a batch of data
print("Getting test batch...")
data_iter = iter(train_loader)
x, y, _, _ = next(data_iter)

print(f"Input batch shape: {x.shape}")
print(f"Input min/max: {x.min().item():.6f}/{x.max().item():.6f}")
print(f"Input contains NaN: {torch.isnan(x).any()}")
print(f"Input contains Inf: {torch.isinf(x).any()}")

# Create a simple model to test layer by layer
print("\nCreating simple test model...")
model = GatedCNN(args.input_size, args.output_size, channels=32, n_layers=2, hidden_size=256, n_mlp=2, type_mod='gated', args=args)

# Test forward pass step by step
print("\nTesting forward pass step by step...")
model.eval()
with torch.no_grad():
    # Initial input
    out = x
    print(f"Step 0 - Input: shape={out.shape}, min/max={out.min().item():.6f}/{out.max().item():.6f}, NaN={torch.isnan(out).any()}, Inf={torch.isinf(out).any()}")
    
    # Add channel dimension
    out = out.unsqueeze(1) if len(out.shape) < 4 else out
    print(f"Step 1 - After unsqueeze: shape={out.shape}, min/max={out.min().item():.6f}/{out.max().item():.6f}, NaN={torch.isnan(out).any()}, Inf={torch.isinf(out).any()}")
    
    # Test each CNN layer
    for i, layer in enumerate(model.net):
        out = layer(out)
        print(f"Step {i+2} - After CNN layer {i}: shape={out.shape}, min/max={out.min().item():.6f}/{out.max().item():.6f}, NaN={torch.isnan(out).any()}, Inf={torch.isinf(out).any()}")
        if torch.isnan(out).any() or torch.isinf(out).any():
            print(f"  *** NaN/Inf detected at CNN layer {i}! ***")
            break
    
    # Flatten
    if not torch.isnan(out).any() and not torch.isinf(out).any():
        out = out.view(x.shape[0], -1)
        print(f"Step flatten - After flatten: shape={out.shape}, min/max={out.min().item():.6f}/{out.max().item():.6f}, NaN={torch.isnan(out).any()}, Inf={torch.isinf(out).any()}")
        
        # Test each MLP layer
        for i, layer in enumerate(model.mlp):
            out = layer(out)
            print(f"Step MLP{i} - After MLP layer {i}: shape={out.shape}, min/max={out.min().item():.6f}/{out.max().item():.6f}, NaN={torch.isnan(out).any()}, Inf={torch.isinf(out).any()}")
            if torch.isnan(out).any() or torch.isinf(out).any():
                print(f"  *** NaN/Inf detected at MLP layer {i}! ***")
                break

print("\nLayer-by-layer debugging complete.")