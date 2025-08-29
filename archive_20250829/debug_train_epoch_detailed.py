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

# Setup optimizer exactly like training script
optimizer = optim.AdamW(model.parameters(), lr=args.lr)

print("\n=== DEBUGGING TRAIN_EPOCH STEP BY STEP ===")

# Manually implement train_epoch to debug step by step
model.train()
full_loss = 0
batch_count = 0

# Simulate epoch parameters like in training script
epoch = 15  # After warm_latent and start_regress
beta = min(1, (epoch - args.warm_latent) / args.beta_factor) if epoch > args.warm_latent else 0
gamma = args.gamma
delta = min(1, (epoch - args.start_regress) / args.warm_regress) if epoch > args.start_regress else 0

print(f"Epoch {epoch}: beta={beta:.3f}, gamma={gamma:.3f}, delta={delta:.3f}")

for batch_idx, batch_data in enumerate(train_loader):
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
    
    # Zero gradients
    optimizer.zero_grad()
    
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
    
    # Calculate individual losses
    rec_loss_val = model.recons_loss(x_tilde, x)
    reg_loss_val = loss(p_tilde, y)
    
    print(f"  rec_loss: {rec_loss_val.item():.6f}")
    print(f"  reg_loss: {reg_loss_val.item():.6f}")
    
    if torch.isnan(rec_loss_val) or torch.isinf(rec_loss_val):
        print(f"  ERROR: rec_loss has NaN/Inf values!")
        print(f"  x_tilde stats: mean={x_tilde.mean().item():.6f}, std={x_tilde.std().item():.6f}")
        print(f"  x stats: mean={x.mean().item():.6f}, std={x.std().item():.6f}")
        break
    
    if torch.isnan(reg_loss_val) or torch.isinf(reg_loss_val):
        print(f"  ERROR: reg_loss has NaN/Inf values!")
        print(f"  p_tilde stats: mean={p_tilde.mean().item():.6f}, std={p_tilde.std().item():.6f}")
        print(f"  y stats: mean={y.mean().item():.6f}, std={y.std().item():.6f}")
        break
    
    # Combine losses exactly like train_epoch
    b_loss_components = torch.stack([rec_loss_val, beta * z_loss, delta * args.reg_factor * reg_loss_val])
    b_loss = b_loss_components.mean(dim=0)
    
    print(f"  b_loss_components: {b_loss_components.detach().cpu().numpy()}")
    print(f"  b_loss: {b_loss.item():.6f}")
    
    if torch.isnan(b_loss_components).any() or torch.isinf(b_loss_components).any():
        print(f"  ERROR: b_loss_components has NaN/Inf values!")
        print(f"  rec_loss_val: {rec_loss_val.item():.6f}")
        print(f"  beta * z_loss: {(beta * z_loss).item():.6f}")
        print(f"  delta * args.reg_factor * reg_loss_val: {(delta * args.reg_factor * reg_loss_val).item():.6f}")
        break
    
    if torch.isnan(b_loss) or torch.isinf(b_loss):
        print(f"  ERROR: b_loss has NaN/Inf values!")
        break
    
    # Backward pass
    b_loss.backward()
    
    # Check gradients
    grad_norm = 0
    param_count = 0
    nan_grad_count = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_count += 1
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                nan_grad_count += 1
                print(f"  ERROR: Parameter {name} has NaN/Inf gradients!")
            grad_norm += param.grad.data.norm(2).item() ** 2
    
    grad_norm = grad_norm ** 0.5
    print(f"  Gradient norm: {grad_norm:.6f}, params with grads: {param_count}, NaN grads: {nan_grad_count}")
    
    if nan_grad_count > 0:
        print(f"  ERROR: Found {nan_grad_count} parameters with NaN/Inf gradients!")
        break
    
    # Optimizer step
    optimizer.step()
    
    full_loss += b_loss.detach()
    batch_count += 1
    print(f"  Batch completed successfully, running total: {full_loss.item():.6f}")

if batch_count > 0:
    full_loss /= len(train_loader)
    print(f"\nManual train_epoch result: {full_loss.item():.6f}")
else:
    print(f"\nManual train_epoch FAILED")

print("\n=== TESTING ACTUAL train_epoch METHOD ===")
# Set args.beta and args.gamma for the actual method call
args.beta = beta
args.gamma = delta * args.reg_factor  # This is the key - gamma includes reg_factor
try:
    actual_result = model.train_epoch(train_loader, loss, optimizer, args)
    print(f"Actual train_epoch result: {actual_result.item():.6f}")
except Exception as e:
    print(f"Actual train_epoch failed with error: {e}")

print("\nTrain epoch debugging completed.")