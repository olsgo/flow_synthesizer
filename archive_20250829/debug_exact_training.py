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
        self.train_type = 'fixed'  # Use fixed like in training script
        self.kernel = 5
        self.dilation = 3
        self.n_layers = 6
        self.channels = 128
        self.n_hidden = 2048
        self.encoder_dims = 128
        self.latent_dims = 0  # Will be set to output_size
        self.output_size = 66
        self.layers = 'gated_cnn'
        self.rec_loss = 'l1'
        self.loss = 'l1'
        self.regressor = 'mlp'
        self.reg_hiddens = 512
        self.reg_layers = 4
        self.reg_flow = 'maf'
        self.beta_factor = 1
        self.gamma = 0  # Start with 0 like in training
        self.beta = 0   # Start with 0 like in training
        self.lr = 0.0001
        self.model = 'vae'
        self.warm_latent = 25
        self.start_regress = 10
        self.warm_regress = 50
        self.reg_factor = 5e3

args = Args()

print("Loading dataset (using fixed split like training script)...")
# Load dataset exactly like training script
ref_split = args.path + '/reference_split_' + args.dataset+ "_" + args.data + '.th'
if args.train_type == 'random' or (not os.path.exists(ref_split)):
    train_loader, valid_loader, test_loader, args = load_dataset(args)
else:
    import os
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

print(f"Reconstruction loss: {args.rec_loss}")
print(f"Parameter loss: {args.loss}")

# Construct model exactly like training script
print("Constructing model exactly like training script...")
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

# Create optimizer exactly like training script
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

print("\nTesting exact training loop...")

# Test multiple epochs like the training script
for epoch in range(3):
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
    
    # Test training epoch
    model.train()
    train_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch_data in enumerate(train_loader):
        if batch_idx >= 3:  # Only test first 3 batches
            break
            
        if len(batch_data) == 2:
            data_batch, param_batch = batch_data
        else:
            data_batch = batch_data[0]
            param_batch = batch_data[1] if len(batch_data) > 1 else None
        
        print(f"\nBatch {batch_idx}:")
        print(f"  Data shape: {data_batch.shape}")
        print(f"  Params shape: {param_batch.shape}")
        
        optimizer.zero_grad()
        
        # Forward pass
        x_tilde, z_tilde, z_loss = model.ae_model(data_batch)
        print(f"  x_tilde range: [{x_tilde.min().item():.6f}, {x_tilde.max().item():.6f}]")
        print(f"  z_tilde range: [{z_tilde.min().item():.6f}, {z_tilde.max().item():.6f}]")
        print(f"  z_loss: {z_loss.item():.6f}")
        
        # Check for NaN/Inf
        if torch.isnan(x_tilde).any() or torch.isinf(x_tilde).any():
            print(f"  ERROR: x_tilde has NaN/Inf values!")
            break
        if torch.isnan(z_tilde).any() or torch.isinf(z_tilde).any():
            print(f"  ERROR: z_tilde has NaN/Inf values!")
            break
        if torch.isnan(z_loss) or torch.isinf(z_loss):
            print(f"  ERROR: z_loss has NaN/Inf values!")
            break
        
        # Reconstruction loss
        rec_loss_val = model.recons_loss(x_tilde, data_batch) / (data_batch.shape[1] * data_batch.shape[2])
        print(f"  rec_loss: {rec_loss_val.item():.6f}")
        
        if torch.isnan(rec_loss_val) or torch.isinf(rec_loss_val):
            print(f"  ERROR: rec_loss has NaN/Inf values!")
            break
        
        # Parameter regression
        p_tilde = model.regression_model(z_tilde)
        print(f"  p_tilde range: [{p_tilde.min().item():.6f}, {p_tilde.max().item():.6f}]")
        
        if torch.isnan(p_tilde).any() or torch.isinf(p_tilde).any():
            print(f"  ERROR: p_tilde has NaN/Inf values!")
            break
        
        # Parameter loss
        reg_loss_val = loss(p_tilde, param_batch)
        print(f"  reg_loss: {reg_loss_val.item():.6f}")
        
        if torch.isnan(reg_loss_val) or torch.isinf(reg_loss_val):
            print(f"  ERROR: reg_loss has NaN/Inf values!")
            break
        
        # Combined loss
        b_loss = rec_loss_val + (args.beta * z_loss) + (args.gamma * reg_loss_val)
        print(f"  combined_loss: {b_loss.item():.6f}")
        
        if torch.isnan(b_loss) or torch.isinf(b_loss):
            print(f"  ERROR: combined_loss has NaN/Inf values!")
            break
        
        # Backward pass
        try:
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
            train_loss += b_loss.item()
            num_batches += 1
            print(f"  Batch completed successfully")
            
        except Exception as e:
            print(f"  ERROR during backward/optimizer: {e}")
            break
    
    if num_batches > 0:
        avg_train_loss = train_loss / num_batches
        print(f"\nEpoch {epoch} average train loss: {avg_train_loss:.6f}")
    else:
        print(f"\nEpoch {epoch} FAILED - no batches completed")
        break

print("\nExact training loop debugging completed.")