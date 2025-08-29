#!/usr/bin/env python3

import sys
sys.path.append('code')

import torch
import torch.nn as nn
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
    train_type='sequential',
    device='cpu'  # Use CPU for debugging
)

print("Loading dataset...")
train_loader, valid_loader, test_loader, args = load_dataset(args)

# Get a batch
data_batch, params_batch, meta_batch, audio_batch = next(iter(train_loader))

print(f"Input data shape: {data_batch.shape}")
print(f"Input params shape: {params_batch.shape}")

# Test reconstruction loss computation
print("\nTesting reconstruction loss computation:")

# Create a simple reconstruction loss
rec_loss_fn = nn.MSELoss(reduction='sum')

# Simulate some reconstructed data (same as input for testing)
x_tilde = data_batch.clone()
x = data_batch

print(f"x shape: {x.shape}")
print(f"x_tilde shape: {x_tilde.shape}")

# Test the division that might be causing issues
divisor = x.shape[1] * x.shape[2]
print(f"Divisor (x.shape[1] * x.shape[2]): {divisor}")

# Compute reconstruction loss
rec_loss_raw = rec_loss_fn(x_tilde, x)
print(f"Raw reconstruction loss: {rec_loss_raw.item():.6f}")

rec_loss_normalized = rec_loss_raw / divisor
print(f"Normalized reconstruction loss: {rec_loss_normalized.item():.6f}")

# Check for NaN/Inf
print(f"Is rec_loss_raw NaN? {torch.isnan(rec_loss_raw).item()}")
print(f"Is rec_loss_raw Inf? {torch.isinf(rec_loss_raw).item()}")
print(f"Is rec_loss_normalized NaN? {torch.isnan(rec_loss_normalized).item()}")
print(f"Is rec_loss_normalized Inf? {torch.isinf(rec_loss_normalized).item()}")

# Test parameter loss computation
print("\nTesting parameter loss computation:")

param_loss_fn = nn.MSELoss(reduction='mean')

# Simulate some predicted parameters (same as target for testing)
p_tilde = params_batch.clone()
y = params_batch

print(f"p_tilde shape: {p_tilde.shape}")
print(f"y shape: {y.shape}")

reg_loss = param_loss_fn(p_tilde, y)
print(f"Parameter regression loss: {reg_loss.item():.6f}")

# Check for NaN/Inf
print(f"Is reg_loss NaN? {torch.isnan(reg_loss).item()}")
print(f"Is reg_loss Inf? {torch.isinf(reg_loss).item()}")

# Test combined loss computation
print("\nTesting combined loss computation:")

# Simulate some latent loss
z_loss = torch.tensor(0.1)  # Typical VAE latent loss

# Test different beta and gamma values
beta = 0.84  # From training logs
gamma = 1100.0  # From training logs

print(f"rec_loss: {rec_loss_normalized.item():.6f}")
print(f"z_loss: {z_loss.item():.6f}")
print(f"reg_loss: {reg_loss.item():.6f}")
print(f"beta: {beta}")
print(f"gamma: {gamma}")

# Combined loss computation
b_loss = (rec_loss_normalized + (beta * z_loss) + (gamma * reg_loss)).mean(dim=0)
print(f"Combined loss: {b_loss.item():.6f}")

# Check for NaN/Inf
print(f"Is b_loss NaN? {torch.isnan(b_loss).item()}")
print(f"Is b_loss Inf? {torch.isinf(b_loss).item()}")

# Test with extreme gamma values
print("\nTesting with extreme gamma values:")
for test_gamma in [1.0, 100.0, 1000.0, 10000.0]:
    test_b_loss = (rec_loss_normalized + (beta * z_loss) + (test_gamma * reg_loss)).mean(dim=0)
    print(f"Gamma {test_gamma}: loss = {test_b_loss.item():.6f}, NaN = {torch.isnan(test_b_loss).item()}, Inf = {torch.isinf(test_b_loss).item()}")

print("\nDebug complete.")