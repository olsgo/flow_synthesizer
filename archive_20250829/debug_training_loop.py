#!/usr/bin/env python3

import sys
sys.path.append('code')

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.data import load_dataset
from models.vae.ae import RegressionAE
from models.vae.vae import VAE
from models.basic import construct_encoder_decoder, construct_regressor
import argparse

class Args:
    def __init__(self):
        self.device = 'cpu'
        self.dataset = 'polymax_dataset'
        self.data = 'mel'
        self.path = 'datasets'
        self.nbworkers = 0
        self.batch_size = 16
        self.train_type = 'random'
        self.kernel = 5
        self.dilation = 3
        self.n_layers = 4
        self.channels = 32
        self.n_hidden = 512
        self.encoder_dims = 128
        self.latent_dims = 35
        self.output_size = 66
        self.layers = 'gated'
        self.rec_loss = 'l1'
        self.loss = 'l1'
        self.regressor = 'mlp'
        self.reg_hiddens = 512
        self.reg_layers = 3
        self.reg_flow = 'none'
        self.beta = 1.0
        self.gamma = 1900.0
        self.lr = 0.0001

args = Args()

print("Loading dataset...")
train_loader, valid_loader, test_loader, args = load_dataset(args)

if args.latent_dims == 0:
    args.latent_dims = args.output_size

print(f"Input size: {args.input_size}")
print(f"Output size: {args.output_size}")
print(f"Latent dims: {args.latent_dims}")

# Construct loss functions
if args.rec_loss == 'mse':
    rec_loss_fn = nn.MSELoss(reduction='sum')
elif args.rec_loss == 'l1':
    rec_loss_fn = nn.SmoothL1Loss(reduction='sum')
else:
    rec_loss_fn = nn.MSELoss(reduction='sum')

if args.loss == 'mse':
    loss_params_fn = nn.MSELoss(reduction='mean')
elif args.loss == 'l1':
    loss_params_fn = nn.SmoothL1Loss(reduction='mean')
else:
    loss_params_fn = nn.MSELoss(reduction='mean')

print(f"Reconstruction loss: {args.rec_loss}")
print(f"Parameter loss: {args.loss}")

# Construct model
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
    rec_loss_fn, regressor=regression_model, regressor_name=args.regressor
)

# Create optimizer
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

print("\nTesting training loop...")

# Get a batch of data
batch_data = next(iter(train_loader))
if len(batch_data) == 2:
    data_batch, param_batch = batch_data
else:
    data_batch = batch_data[0]
    param_batch = batch_data[1] if len(batch_data) > 1 else None

print(f"Data batch shape: {data_batch.shape}")
print(f"Params batch shape: {param_batch.shape}")

# Test training step
model.train()
optimizer.zero_grad()

print("\nStep 1: Forward pass...")
# Auto-encode
x_tilde, z_tilde, z_loss = model.ae_model(data_batch)
print(f"x_tilde shape: {x_tilde.shape}")
print(f"z_tilde shape: {z_tilde.shape}")
print(f"z_loss: {z_loss.item():.6f}")

# Check for NaN/Inf in intermediate results
print(f"x_tilde NaN count: {torch.isnan(x_tilde).sum().item()}")
print(f"x_tilde Inf count: {torch.isinf(x_tilde).sum().item()}")
print(f"z_tilde NaN count: {torch.isnan(z_tilde).sum().item()}")
print(f"z_tilde Inf count: {torch.isinf(z_tilde).sum().item()}")
print(f"z_loss NaN: {torch.isnan(z_loss).item()}")
print(f"z_loss Inf: {torch.isinf(z_loss).item()}")

print("\nStep 2: Reconstruction loss...")
# Reconstruction loss
rec_loss = model.recons_loss(x_tilde, data_batch) / (data_batch.shape[1] * data_batch.shape[2])
print(f"rec_loss: {rec_loss.item():.6f}")
print(f"rec_loss NaN: {torch.isnan(rec_loss).item()}")
print(f"rec_loss Inf: {torch.isinf(rec_loss).item()}")

print("\nStep 3: Parameter regression...")
# Perform regression on params
p_tilde = model.regression_model(z_tilde)
print(f"p_tilde shape: {p_tilde.shape}")
print(f"p_tilde NaN count: {torch.isnan(p_tilde).sum().item()}")
print(f"p_tilde Inf count: {torch.isinf(p_tilde).sum().item()}")

print("\nStep 4: Parameter loss...")
# Regression loss
reg_loss = loss_params_fn(p_tilde, param_batch)
print(f"reg_loss: {reg_loss.item():.6f}")
print(f"reg_loss NaN: {torch.isnan(reg_loss).item()}")
print(f"reg_loss Inf: {torch.isinf(reg_loss).item()}")

print("\nStep 5: Combined loss...")
# Final loss
print(f"rec_loss: {rec_loss.item():.6f}")
print(f"beta * z_loss: {(args.beta * z_loss).item():.6f}")
print(f"gamma * reg_loss: {(args.gamma * reg_loss).item():.6f}")

b_loss_components = rec_loss + (args.beta * z_loss) + (args.gamma * reg_loss)
print(f"b_loss_components: {b_loss_components.item():.6f}")
print(f"b_loss_components NaN: {torch.isnan(b_loss_components).item()}")
print(f"b_loss_components Inf: {torch.isinf(b_loss_components).item()}")

b_loss = b_loss_components.mean(dim=0)
print(f"b_loss: {b_loss.item():.6f}")
print(f"b_loss NaN: {torch.isnan(b_loss).item()}")
print(f"b_loss Inf: {torch.isinf(b_loss).item()}")

print("\nStep 6: Backward pass...")
try:
    b_loss.backward()
    print("Backward pass successful")
    
    # Check gradients
    grad_nan_count = 0
    grad_inf_count = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            total_params += 1
            if torch.isnan(param.grad).any():
                grad_nan_count += 1
                print(f"NaN gradient in {name}")
            if torch.isinf(param.grad).any():
                grad_inf_count += 1
                print(f"Inf gradient in {name}")
    
    print(f"Total parameters with gradients: {total_params}")
    print(f"Parameters with NaN gradients: {grad_nan_count}")
    print(f"Parameters with Inf gradients: {grad_inf_count}")
    
except Exception as e:
    print(f"Error during backward pass: {e}")

print("\nStep 7: Optimizer step...")
try:
    optimizer.step()
    print("Optimizer step successful")
except Exception as e:
    print(f"Error during optimizer step: {e}")

print("\nTraining loop debugging completed.")