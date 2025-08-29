#!/usr/bin/env python3

import sys
sys.path.append('/Users/gjb/Projects/flow_synthesizer/code')

import torch
import torch.nn as nn
import numpy as np
from utils.data import load_dataset
from models.basic import construct_encoder_decoder
from models.vae.vae import VAE

print("Testing VAE latent computation for NaN/Inf issues...")

# Load dataset
args_dict = {
    'dataset': 'polymax_dataset',
    'data': 'mel',
    'batch_size': 32,
    'device': 'cpu',
    'path': 'datasets',
    'nbworkers': 0
}

class Args:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

args = Args(**args_dict)

train_loader, valid_loader, test_loader, args = load_dataset(args)

print(f"Input size: {args.input_size}")
print(f"Output size: {args.output_size}")

# Get a batch of data
data_batch, params_batch, _, _ = next(iter(train_loader))
print(f"Data batch shape: {data_batch.shape}")
print(f"Params batch shape: {params_batch.shape}")

# Create a simple VAE model
encoder_dims = 512
latent_dims = 16

encoder, decoder = construct_encoder_decoder(
    args.input_size, encoder_dims, latent_dims, 
    channels=32, n_layers=6, hidden_size=512, 
    n_mlp=3, type_mod='gated_cnn', args=args
)

vae_model = VAE(encoder, decoder, args.input_size, encoder_dims, latent_dims)
vae_model.eval()

print("\nTesting VAE encoding and latent computation...")

with torch.no_grad():
    # Encode the data
    z_params = vae_model.encode(data_batch)
    mu, log_var = z_params
    
    print(f"\nEncoder outputs:")
    print(f"mu shape: {mu.shape}")
    print(f"log_var shape: {log_var.shape}")
    
    print(f"\nmu statistics:")
    print(f"  min: {mu.min().item():.6f}")
    print(f"  max: {mu.max().item():.6f}")
    print(f"  mean: {mu.mean().item():.6f}")
    print(f"  std: {mu.std().item():.6f}")
    print(f"  NaN count: {torch.isnan(mu).sum().item()}")
    print(f"  Inf count: {torch.isinf(mu).sum().item()}")
    
    print(f"\nlog_var statistics:")
    print(f"  min: {log_var.min().item():.6f}")
    print(f"  max: {log_var.max().item():.6f}")
    print(f"  mean: {log_var.mean().item():.6f}")
    print(f"  std: {log_var.std().item():.6f}")
    print(f"  NaN count: {torch.isnan(log_var).sum().item()}")
    print(f"  Inf count: {torch.isinf(log_var).sum().item()}")
    
    # Check if log_var is too large (could cause exp() to explode)
    large_log_var = log_var > 10
    if large_log_var.any():
        print(f"\nWARNING: Found {large_log_var.sum().item()} log_var values > 10")
        print(f"  Max log_var: {log_var.max().item():.6f}")
    
    # Check if log_var is too small (could cause numerical issues)
    small_log_var = log_var < -10
    if small_log_var.any():
        print(f"\nWARNING: Found {small_log_var.sum().item()} log_var values < -10")
        print(f"  Min log_var: {log_var.min().item():.6f}")
    
    # Test the latent computation
    print(f"\nTesting latent computation...")
    z, kl_div = vae_model.latent(data_batch, z_params)
    
    print(f"\nLatent z statistics:")
    print(f"  shape: {z.shape}")
    print(f"  min: {z.min().item():.6f}")
    print(f"  max: {z.max().item():.6f}")
    print(f"  mean: {z.mean().item():.6f}")
    print(f"  std: {z.std().item():.6f}")
    print(f"  NaN count: {torch.isnan(z).sum().item()}")
    print(f"  Inf count: {torch.isinf(z).sum().item()}")
    
    print(f"\nKL divergence:")
    print(f"  value: {kl_div.item():.6f}")
    print(f"  is NaN: {torch.isnan(kl_div).item()}")
    print(f"  is Inf: {torch.isinf(kl_div).item()}")
    
    # Test the full forward pass
    print(f"\nTesting full VAE forward pass...")
    x_tilde, z_tilde, z_loss = vae_model(data_batch)
    
    print(f"\nReconstructed x_tilde statistics:")
    print(f"  shape: {x_tilde.shape}")
    print(f"  min: {x_tilde.min().item():.6f}")
    print(f"  max: {x_tilde.max().item():.6f}")
    print(f"  mean: {x_tilde.mean().item():.6f}")
    print(f"  std: {x_tilde.std().item():.6f}")
    print(f"  NaN count: {torch.isnan(x_tilde).sum().item()}")
    print(f"  Inf count: {torch.isinf(x_tilde).sum().item()}")
    
    print(f"\nz_loss (should be same as kl_div):")
    print(f"  value: {z_loss.item():.6f}")
    print(f"  is NaN: {torch.isnan(z_loss).item()}")
    print(f"  is Inf: {torch.isinf(z_loss).item()}")
    
    # Test reconstruction loss
    print(f"\nTesting reconstruction loss...")
    rec_loss_fn = nn.MSELoss(reduction='sum')
    rec_loss = rec_loss_fn(x_tilde, data_batch) / (data_batch.shape[1] * data_batch.shape[2])
    
    print(f"Reconstruction loss:")
    print(f"  value: {rec_loss.item():.6f}")
    print(f"  is NaN: {torch.isnan(rec_loss).item()}")
    print(f"  is Inf: {torch.isinf(rec_loss).item()}")

print("\nVAE latent computation test completed.")