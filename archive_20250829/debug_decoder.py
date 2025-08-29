#!/usr/bin/env python3
"""
Debug the decoder output step by step
"""

import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append('code')

from models.basic import construct_encoder_decoder
from utils.data import load_dataset
import argparse

# Create args for model construction
args_dict = {
    'device': 'cpu',
    'dataset': 'polymax_dataset',
    'data': 'mel',
    'path': 'datasets',
    'nbworkers': 0,
    'batch_size': 16,
    'train_type': 'random',
    'kernel': 5,
    'dilation': 3,
    'n_layers': 4,
    'channels': 32,
    'n_hidden': 512
}

args = argparse.Namespace(**args_dict)

print("Loading dataset...")
dataset, train_loader, val_loader, test_loader = load_dataset(args)

# Get a batch of data
batch_data = next(iter(train_loader))
if len(batch_data) == 2:
    data_batch, param_batch = batch_data
else:
    data_batch = batch_data[0]
    param_batch = batch_data[1] if len(batch_data) > 1 else None
print(f"Input data shape: {data_batch.shape}")
print(f"Input data range: [{data_batch.min().item():.6f}, {data_batch.max().item():.6f}]")

# Model parameters
input_size = [1, 128, 173]  # [C, H, W]
encoder_dims = 128
latent_dims = 35

print(f"\nConstructing encoder/decoder with input_size={input_size}")
encoder, decoder = construct_encoder_decoder(
    input_size, encoder_dims, latent_dims, 
    channels=args.channels, n_layers=args.n_layers, 
    hidden_size=args.n_hidden, n_mlp=args.n_layers // 2, 
    type_mod='gated_cnn', args=args
)

print(f"\nDecoder CNN size: {decoder.cnn_size}")
print(f"Decoder out_size: {decoder.out_size}")

# Test encoder
print("\nTesting encoder...")
with torch.no_grad():
    encoded = encoder(data_batch)
    print(f"Encoded shape: {encoded.shape}")
    print(f"Encoded range: [{encoded.min().item():.6f}, {encoded.max().item():.6f}]")

# Create latent input for decoder
latent_input = torch.randn(data_batch.shape[0], latent_dims)
print(f"\nLatent input shape: {latent_input.shape}")
print(f"Latent input range: [{latent_input.min().item():.6f}, {latent_input.max().item():.6f}]")

# Set target width on decoder
decoder.target_width = data_batch.shape[-1]
print(f"Set decoder target_width to: {decoder.target_width}")

# Test decoder step by step
print("\nTesting decoder step by step...")
with torch.no_grad():
    # MLP part
    out = latent_input
    print(f"Initial input to decoder: {out.shape}")
    
    for i, layer in enumerate(decoder.mlp):
        out = layer(out)
        print(f"After MLP layer {i}: {out.shape}, range: [{out.min().item():.6f}, {out.max().item():.6f}]")
    
    # Reshape for CNN
    out = out.unsqueeze(1).view(-1, 1, decoder.cnn_size[0], decoder.cnn_size[1])
    print(f"After reshape for CNN: {out.shape}, range: [{out.min().item():.6f}, {out.max().item():.6f}]")
    
    # CNN part
    for i, layer in enumerate(decoder.net):
        out = layer(out)
        print(f"After CNN layer {i}: {out.shape}, range: [{out.min().item():.6f}, {out.max().item():.6f}]")
    
    print(f"\nFinal CNN output shape: {out.shape}")
    print(f"Final CNN output range: [{out.min().item():.6f}, {out.max().item():.6f}]")
    
    # Test full decoder
    print("\nTesting full decoder...")
    decoded = decoder(latent_input)
    print(f"Full decoder output shape: {decoded.shape}")
    print(f"Full decoder output range: [{decoded.min().item():.6f}, {decoded.max().item():.6f}]")
    
    print(f"\nExpected output shape: {data_batch.shape[1:]}")
    print(f"Actual output shape: {decoded.shape[1:]}")
    print(f"Shape match: {decoded.shape[1:] == data_batch.shape[1:]}")

print("\nDecoder debugging completed.")