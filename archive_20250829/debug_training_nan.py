#!/usr/bin/env python3

import sys
sys.path.append('/Users/gjb/Projects/flow_synthesizer/code')

import torch
import torch.nn as nn
import numpy as np
from utils.data import load_dataset
from models.basic import construct_encoder_decoder, construct_regressor
from models.vae.vae import VAE
from models.vae.ae import RegressionAE
from models.loss import multinomial_loss, multinomial_mse_loss

print("Debugging NaN values during training process...")

# Replicate the exact training setup from train_optimized.py
class Args:
    def __init__(self):
        self.dataset = 'polymax_dataset'
        self.data = 'mel'
        self.batch_size = 16
        self.device = 'cpu'
        self.path = 'datasets'
        self.nbworkers = 0
        self.train_type = 'fixed'
        
        # Model parameters (from train_optimized.py)
        self.model = 'vae'
        self.loss = 'l1'
        self.rec_loss = 'l1'
        self.n_classes = 61
        self.n_hidden = 2048
        self.n_layers = 6
        self.channels = 128
        self.kernel = 5
        self.dilation = 3
        self.layers = 'gated_cnn'
        self.encoder_dims = 128
        self.latent_dims = 0  # Will be set to output_size
        self.warm_latent = 25
        self.start_regress = 10
        self.warm_regress = 50
        self.beta_factor = 1
        self.regressor = 'mlp'
        self.reg_layers = 4
        self.reg_hiddens = 512
        self.reg_flow = 'maf'
        self.reg_factor = 5e3
        self.lr = 5e-4
        self.semantic_dim = -1

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
    rec_loss = nn.MSELoss(reduction='sum')
elif args.rec_loss == 'l1':
    rec_loss = nn.SmoothL1Loss(reduction='sum')
else:
    rec_loss = nn.MSELoss(reduction='sum')

if args.loss == 'mse':
    loss_params = nn.MSELoss(reduction='mean')
elif args.loss == 'l1':
    loss_params = nn.SmoothL1Loss(reduction='mean')
else:
    loss_params = nn.MSELoss(reduction='mean')

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
    rec_loss, regression_model, args.regressor
)

model.eval()

print("\nTesting forward pass with a single batch...")

# Get a batch of data
data_batch, params_batch, meta_batch, audio_batch = next(iter(train_loader))
print(f"Data batch shape: {data_batch.shape}")
print(f"Params batch shape: {params_batch.shape}")

# Check input data for NaN/Inf
print(f"\nInput data check:")
print(f"  Data NaN count: {torch.isnan(data_batch).sum().item()}")
print(f"  Data Inf count: {torch.isinf(data_batch).sum().item()}")
print(f"  Data min: {data_batch.min().item():.6f}")
print(f"  Data max: {data_batch.max().item():.6f}")
print(f"  Params NaN count: {torch.isnan(params_batch).sum().item()}")
print(f"  Params Inf count: {torch.isinf(params_batch).sum().item()}")
print(f"  Params min: {params_batch.min().item():.6f}")
print(f"  Params max: {params_batch.max().item():.6f}")

with torch.no_grad():
    print("\nStep 1: VAE Encoding...")
    z_params = vae_model.encode(data_batch)
    mu, log_var = z_params
    
    print(f"  mu NaN count: {torch.isnan(mu).sum().item()}")
    print(f"  mu Inf count: {torch.isinf(mu).sum().item()}")
    print(f"  mu range: [{mu.min().item():.6f}, {mu.max().item():.6f}]")
    
    print(f"  log_var NaN count: {torch.isnan(log_var).sum().item()}")
    print(f"  log_var Inf count: {torch.isinf(log_var).sum().item()}")
    print(f"  log_var range: [{log_var.min().item():.6f}, {log_var.max().item():.6f}]")
    
    print("\nStep 2: VAE Latent sampling...")
    z_tilde, z_loss = vae_model.latent(data_batch, z_params)
    
    print(f"  z_tilde NaN count: {torch.isnan(z_tilde).sum().item()}")
    print(f"  z_tilde Inf count: {torch.isinf(z_tilde).sum().item()}")
    print(f"  z_tilde range: [{z_tilde.min().item():.6f}, {z_tilde.max().item():.6f}]")
    
    print(f"  z_loss value: {z_loss.item():.6f}")
    print(f"  z_loss NaN: {torch.isnan(z_loss).item()}")
    print(f"  z_loss Inf: {torch.isinf(z_loss).item()}")
    
    print("\nStep 3: VAE Decoding...")
    x_tilde = vae_model.decode(z_tilde)
    
    print(f"  x_tilde NaN count: {torch.isnan(x_tilde).sum().item()}")
    print(f"  x_tilde Inf count: {torch.isinf(x_tilde).sum().item()}")
    print(f"  x_tilde range: [{x_tilde.min().item():.6f}, {x_tilde.max().item():.6f}]")
    
    print("\nStep 4: Reconstruction loss...")
    rec_loss_val = rec_loss(x_tilde, data_batch) / (data_batch.shape[1] * data_batch.shape[2])
    
    print(f"  rec_loss value: {rec_loss_val.item():.6f}")
    print(f"  rec_loss NaN: {torch.isnan(rec_loss_val).item()}")
    print(f"  rec_loss Inf: {torch.isinf(rec_loss_val).item()}")
    
    print("\nStep 5: Parameter regression...")
    p_tilde = regression_model(z_tilde)
    
    print(f"  p_tilde NaN count: {torch.isnan(p_tilde).sum().item()}")
    print(f"  p_tilde Inf count: {torch.isinf(p_tilde).sum().item()}")
    print(f"  p_tilde range: [{p_tilde.min().item():.6f}, {p_tilde.max().item():.6f}]")
    
    print("\nStep 6: Parameter loss...")
    reg_loss_val = loss_params(p_tilde, params_batch)
    
    print(f"  reg_loss value: {reg_loss_val.item():.6f}")
    print(f"  reg_loss NaN: {torch.isnan(reg_loss_val).item()}")
    print(f"  reg_loss Inf: {torch.isinf(reg_loss_val).item()}")
    
    print("\nStep 7: Combined loss...")
    beta = 1.0  # From training
    gamma = 1900.0  # From training logs
    
    print(f"  beta: {beta}")
    print(f"  gamma: {gamma}")
    
    b_loss = (rec_loss_val + (beta * z_loss) + (gamma * reg_loss_val)).mean(dim=0)
    
    print(f"  Combined loss value: {b_loss.item():.6f}")
    print(f"  Combined loss NaN: {torch.isnan(b_loss).item()}")
    print(f"  Combined loss Inf: {torch.isinf(b_loss).item()}")
    
    print("\nStep 8: Full model forward pass...")
    model_output = model(data_batch)
    
    print(f"  Model output NaN count: {torch.isnan(model_output).sum().item()}")
    print(f"  Model output Inf count: {torch.isinf(model_output).sum().item()}")
    print(f"  Model output range: [{model_output.min().item():.6f}, {model_output.max().item():.6f}]")

print("\nDebugging completed.")