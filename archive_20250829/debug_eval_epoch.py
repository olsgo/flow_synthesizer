#!/usr/bin/env python3

import sys
sys.path.append('code')

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.data import load_dataset
from models.vae.ae import RegressionAE
from models.vae.vae import VAE
from models.basic import construct_encoder_decoder, construct_regressor
from models.loss import multinomial_loss, multinomial_mse_loss
import argparse

# Exact same configuration as train_optimized.py
class Args:
    def __init__(self):
        self.device = 'cpu'
        self.dataset = 'polymax_dataset'
        self.data = 'mel'
        self.path = 'datasets'
        self.nbworkers = 0
        self.batch_size = 16
        self.train_type = 'fixed'
        self.kernel = 5
        self.dilation = 3
        self.n_layers = 6
        self.channels = 128
        self.n_hidden = 2048
        self.encoder_dims = 128
        self.latent_dims = 0
        self.output_size = 66
        self.layers = 'gated_cnn'
        self.rec_loss = 'l1'
        self.loss = 'l1'
        self.regressor = 'mlp'
        self.reg_hiddens = 512
        self.reg_layers = 4
        self.reg_flow = 'maf'
        self.beta_factor = 1
        self.gamma = 0
        self.beta = 0
        self.lr = 0.0001
        self.model = 'vae'
        self.warm_latent = 25
        self.start_regress = 10
        self.warm_regress = 50
        self.reg_factor = 5e3

args = Args()

print("Loading dataset...")
ref_split = args.path + '/reference_split_' + args.dataset+ "_" + args.data + '.th'
if args.train_type == 'random' or (not os.path.exists(ref_split)):
    train_loader, valid_loader, test_loader, args = load_dataset(args)
else:
    data = torch.load(ref_split)
    train_loader, valid_loader, test_loader = data[0], data[1], data[2]
    args.output_size = train_loader.dataset.output_size
    args.input_size = train_loader.dataset.input_size

if args.latent_dims == 0:
    args.latent_dims = args.output_size

print(f"Input size: {args.input_size}")
print(f"Output size: {args.output_size}")
print(f"Latent dims: {args.latent_dims}")

# Construct loss functions exactly like training script
if args.rec_loss == 'mse':
    rec_loss = nn.MSELoss(reduction='sum')
elif args.rec_loss == 'l1':
    rec_loss = nn.SmoothL1Loss(reduction='sum')
elif args.rec_loss == 'multinomial':
    rec_loss = multinomial_loss
elif args.rec_loss == 'multi_mse':
    rec_loss = multinomial_mse_loss
else:
    raise Exception('Unknown reconstruction loss ' + args.rec_loss)

if args.loss == 'mse':
    loss = nn.MSELoss(reduction='mean')
elif args.loss == 'l1':
    loss = nn.SmoothL1Loss(reduction='mean')
elif args.loss == 'bce':
    loss = nn.BCELoss(reduction='mean')
elif args.loss == 'multinomial':
    loss = multinomial_loss
elif args.loss == 'multi_mse':
    loss = multinomial_mse_loss
else:
    raise Exception('Unknown loss ' + args.loss)

# Construct model exactly like training script
print("Constructing model...")
encoder, decoder = construct_encoder_decoder(
    args.input_size, args.encoder_dims, args.latent_dims, 
    channels=args.channels, n_layers=args.n_layers, 
    hidden_size=args.n_hidden, n_mlp=args.n_layers // 2, 
    type_mod=args.layers, args=args
)

vae_model = VAE(encoder, decoder, args.input_size, args.encoder_dims, args.latent_dims)
regression_model = construct_regressor(
    args.latent_dims, args.output_size, 
    model=args.regressor, hidden_dims=args.reg_hiddens, 
    n_layers=args.reg_layers, flow_type=args.reg_flow
)

model = RegressionAE(
    vae_model, args.latent_dims, args.output_size, 
    rec_loss, regressor=regression_model, regressor_name=args.regressor
)

print("\nTesting eval_epoch method exactly...")

# Test the exact eval_epoch method step by step
print("\n=== DEBUGGING EVAL_EPOCH ===")

# Manually implement eval_epoch to debug step by step
model.eval()
full_loss = 0
batch_count = 0

with torch.no_grad():
    for batch_idx, batch_data in enumerate(valid_loader):
        if batch_idx >= 3:  # Only test first 3 batches
            break
            
        x, y, _, _ = batch_data
        x, y = x.to(args.device, non_blocking=True), y.to(args.device, non_blocking=True)
        
        print(f"\nBatch {batch_idx}:")
        print(f"  Input shapes: x={x.shape}, y={y.shape}")
        print(f"  Input ranges: x=[{x.min().item():.6f}, {x.max().item():.6f}], y=[{y.min().item():.6f}, {y.max().item():.6f}]")
        
        # Check for NaN/Inf in inputs
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"  ERROR: Input x has NaN/Inf values!")
            break
        if torch.isnan(y).any() or torch.isinf(y).any():
            print(f"  ERROR: Input y has NaN/Inf values!")
            break
        
        # Auto-encode
        x_tilde, z_tilde, z_loss = model.ae_model(x)
        print(f"  After ae_model: x_tilde={x_tilde.shape}, z_tilde={z_tilde.shape}")
        print(f"  z_loss: {z_loss.item():.6f}")
        print(f"  x_tilde range: [{x_tilde.min().item():.6f}, {x_tilde.max().item():.6f}]")
        print(f"  z_tilde range: [{z_tilde.min().item():.6f}, {z_tilde.max().item():.6f}]")
        
        # Check for NaN/Inf in ae_model outputs
        if torch.isnan(x_tilde).any() or torch.isinf(x_tilde).any():
            print(f"  ERROR: x_tilde has NaN/Inf values!")
            break
        if torch.isnan(z_tilde).any() or torch.isinf(z_tilde).any():
            print(f"  ERROR: z_tilde has NaN/Inf values!")
            break
        if torch.isnan(z_loss) or torch.isinf(z_loss):
            print(f"  ERROR: z_loss has NaN/Inf values!")
            break
        
        # Perform regression on params
        p_tilde = model.regression_model(z_tilde)
        print(f"  p_tilde shape: {p_tilde.shape}")
        print(f"  p_tilde range: [{p_tilde.min().item():.6f}, {p_tilde.max().item():.6f}]")
        
        if torch.isnan(p_tilde).any() or torch.isinf(p_tilde).any():
            print(f"  ERROR: p_tilde has NaN/Inf values!")
            break
        
        # Regression loss - THIS IS THE ONLY LOSS CALCULATED IN eval_epoch
        reg_loss = loss(p_tilde, y)
        print(f"  reg_loss: {reg_loss.item():.6f}")
        
        if torch.isnan(reg_loss) or torch.isinf(reg_loss):
            print(f"  ERROR: reg_loss has NaN/Inf values!")
            print(f"  p_tilde stats: mean={p_tilde.mean().item():.6f}, std={p_tilde.std().item():.6f}")
            print(f"  y stats: mean={y.mean().item():.6f}, std={y.std().item():.6f}")
            print(f"  p_tilde has NaN: {torch.isnan(p_tilde).any()}")
            print(f"  y has NaN: {torch.isnan(y).any()}")
            break
        
        full_loss += reg_loss
        batch_count += 1
        print(f"  Batch completed successfully, running total: {full_loss.item():.6f}")
    
    if batch_count > 0:
        full_loss /= len(valid_loader)  # This is what eval_epoch does
        print(f"\neval_epoch result: {full_loss.item():.6f}")
    else:
        print(f"\neval_epoch FAILED")

print("\n=== TESTING ACTUAL eval_epoch METHOD ===")
try:
    actual_result = model.eval_epoch(valid_loader, loss, args)
    print(f"Actual eval_epoch result: {actual_result.item():.6f}")
except Exception as e:
    print(f"Actual eval_epoch failed with error: {e}")

print("\nEval epoch debugging completed.")