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

optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

print("\nTesting train_epoch method exactly...")

# Test the exact train_epoch method
for epoch in range(2):
    print(f"\n=== EPOCH {epoch} ===")
    
    # Set beta and gamma exactly like training script
    args.beta = args.beta_factor * (float(epoch) / float(max(args.warm_latent, epoch)))
    if epoch >= args.start_regress:
        args.gamma = ((float(epoch - args.start_regress) * args.reg_factor) / float(max(args.warm_regress, epoch - args.start_regress)))
        if args.regressor != 'mlp':
            args.gamma *= 1e-1
    else:
        args.gamma = 0
    
    print(f"Beta: {args.beta:.6f}, Gamma: {args.gamma:.6f}")
    
    # Manually implement train_epoch to debug step by step
    model.train()
    full_loss = 0
    batch_count = 0
    
    for batch_idx, batch_data in enumerate(train_loader):
        if batch_idx >= 3:  # Only test first 3 batches
            break
            
        x, y, _, _ = batch_data
        x, y = x.to(args.device, non_blocking=True), y.to(args.device, non_blocking=True)
        
        print(f"\nBatch {batch_idx}:")
        print(f"  Input shapes: x={x.shape}, y={y.shape}")
        
        # Auto-encode
        x_tilde, z_tilde, z_loss = model.ae_model(x)
        print(f"  After ae_model: x_tilde={x_tilde.shape}, z_tilde={z_tilde.shape}")
        print(f"  z_loss: {z_loss.item():.6f}")
        
        # Check for NaN/Inf in ae_model outputs
        if torch.isnan(x_tilde).any():
            print(f"  ERROR: x_tilde has NaN values!")
            break
        if torch.isnan(z_tilde).any():
            print(f"  ERROR: z_tilde has NaN values!")
            break
        if torch.isnan(z_loss):
            print(f"  ERROR: z_loss has NaN values!")
            break
        
        # Reconstruction loss
        rec_loss_val = model.recons_loss(x_tilde, x) / (x.shape[1] * x.shape[2])
        print(f"  rec_loss: {rec_loss_val.item():.6f}")
        
        if torch.isnan(rec_loss_val):
            print(f"  ERROR: rec_loss has NaN values!")
            break
        
        # Regression part
        if model.regressor == 'mlp':
            # Perform regression on params
            p_tilde = model.regression_model(z_tilde)
            print(f"  p_tilde shape: {p_tilde.shape}")
            print(f"  p_tilde range: [{p_tilde.min().item():.6f}, {p_tilde.max().item():.6f}]")
            
            if torch.isnan(p_tilde).any():
                print(f"  ERROR: p_tilde has NaN values!")
                break
            
            # Regression loss
            reg_loss_val = loss(p_tilde, y)
            print(f"  reg_loss: {reg_loss_val.item():.6f}")
            
            if torch.isnan(reg_loss_val):
                print(f"  ERROR: reg_loss has NaN values!")
                break
        else:
            # Use log probability model
            p_tilde, reg_loss_val = model.regression_model.log_prob(z_tilde, y)
            print(f"  reg_loss (log_prob): {reg_loss_val.item():.6f}")
        
        # Final loss - THIS IS THE CRITICAL LINE FROM train_epoch
        b_loss_components = rec_loss_val + (args.beta * z_loss) + (args.gamma * reg_loss_val)
        print(f"  b_loss_components shape: {b_loss_components.shape}")
        print(f"  b_loss_components: {b_loss_components.item():.6f}")
        
        # The problematic line from train_epoch:
        b_loss = b_loss_components.mean(dim=0)
        print(f"  b_loss after mean(dim=0): {b_loss.item():.6f}")
        
        if torch.isnan(b_loss):
            print(f"  ERROR: b_loss has NaN values after mean(dim=0)!")
            break
        
        # Perform backward
        optimizer.zero_grad()
        b_loss.backward()
        
        # Check gradients
        grad_nan_count = 0
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                grad_nan_count += 1
                print(f"  ERROR: NaN gradient in {name}")
        
        if grad_nan_count > 0:
            print(f"  ERROR: {grad_nan_count} parameters have NaN gradients!")
            break
        
        optimizer.step()
        full_loss += b_loss
        batch_count += 1
        print(f"  Batch completed successfully")
    
    if batch_count > 0:
        full_loss /= len(train_loader)  # This is what train_epoch does
        print(f"\nEpoch {epoch} train_epoch result: {full_loss.item():.6f}")
    else:
        print(f"\nEpoch {epoch} FAILED")
        break

print("\nTrain epoch debugging completed.")