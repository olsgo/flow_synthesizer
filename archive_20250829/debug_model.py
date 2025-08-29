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
    batch_size=32,
    kernel=5,
    dilation=3
)

print("Loading dataset...")
train_loader, valid_loader, test_loader, args = load_dataset(args)

print(f"Input size: {args.input_size}")
print(f"Output size: {args.output_size}")

# Create model
print("Creating GatedCNN model...")
model = GatedCNN(args.input_size, args.output_size, channels=64, n_layers=4, hidden_size=1024, n_mlp=3, type_mod='gated', args=args)

# Check initial weights
print("\nChecking initial model weights...")
for name, param in model.named_parameters():
    if torch.isnan(param).any():
        print(f"NaN found in initial weights: {name}")
    if torch.isinf(param).any():
        print(f"Inf found in initial weights: {name}")
    print(f"{name}: min={param.min().item():.6f}, max={param.max().item():.6f}, mean={param.mean().item():.6f}")

# Skip custom initialization - use PyTorch defaults
print("\nSkipping custom initialization - using PyTorch defaults...")

# Check weights after initialization
print("\nChecking weights after initialization...")
for name, param in model.named_parameters():
    if torch.isnan(param).any():
        print(f"NaN found in weights after init: {name}")
    if torch.isinf(param).any():
        print(f"Inf found in weights after init: {name}")
    print(f"{name}: min={param.min().item():.6f}, max={param.max().item():.6f}, mean={param.mean().item():.6f}")

# Get a batch of data
print("\nGetting test batch...")
data_iter = iter(train_loader)
x, y, _, _ = next(data_iter)

print(f"Input batch shape: {x.shape}")
print(f"Target batch shape: {y.shape}")
print(f"Input min/max: {x.min().item():.6f}/{x.max().item():.6f}")
print(f"Target min/max: {y.min().item():.6f}/{y.max().item():.6f}")

# Check for NaN/Inf in input data
if torch.isnan(x).any():
    print("NaN found in input data!")
if torch.isinf(x).any():
    print("Inf found in input data!")
if torch.isnan(y).any():
    print("NaN found in target data!")
if torch.isinf(y).any():
    print("Inf found in target data!")

# Forward pass
print("\nPerforming forward pass...")
model.train()
out = model(x)

print(f"Output shape: {out.shape}")
print(f"Output min/max: {out.min().item():.6f}/{out.max().item():.6f}")

if torch.isnan(out).any():
    print("NaN found in model output!")
if torch.isinf(out).any():
    print("Inf found in model output!")

# Compute loss
print("\nComputing loss...")
loss_fn = nn.MSELoss()
loss = loss_fn(out, y)
print(f"Loss: {loss.item()}")

if torch.isnan(loss):
    print("NaN found in loss!")
if torch.isinf(loss):
    print("Inf found in loss!")

# Backward pass
print("\nPerforming backward pass...")
loss.backward()

# Check gradients
print("\nChecking gradients...")
for name, param in model.named_parameters():
    if param.grad is not None:
        if torch.isnan(param.grad).any():
            print(f"NaN found in gradients: {name}")
        if torch.isinf(param.grad).any():
            print(f"Inf found in gradients: {name}")
        grad_norm = param.grad.norm().item()
        print(f"{name} grad norm: {grad_norm:.6f}")
        if grad_norm > 1000:
            print(f"  WARNING: Large gradient norm in {name}")
    else:
        print(f"{name}: No gradient")

print("\nDebugging complete.")