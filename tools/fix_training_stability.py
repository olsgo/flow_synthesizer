#!/usr/bin/env python3
"""
Fix for training stability issues in flow_synthesizer.
The main issues identified:
1. Reconstruction loss is extremely high (500k+) compared to other losses
2. This causes numerical instability and NaN values
3. Need gradient clipping and better loss scaling
"""

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

# Create a patched version of RegressionAE with better stability
class StableRegressionAE(RegressionAE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rec_loss_scale = 1e-4  # Scale down reconstruction loss
        self.max_grad_norm = 1.0    # Gradient clipping
    
    def train_epoch(self, loader, loss_params, optimizer, args):
        self.train()
        full_loss = 0
        for (x, y, _, _) in loader:
            # Send to device
            x, y = x.to(args.device, non_blocking=True), y.to(args.device, non_blocking=True)
            
            # Auto-encode
            x_tilde, z_tilde, z_loss = self.ae_model(x)
            
            # Reconstruction loss with scaling
            rec_loss = self.recons_loss(x_tilde, x) / (x.shape[1] * x.shape[2])
            rec_loss = rec_loss * self.rec_loss_scale  # Scale down
            
            if (self.regressor == 'mlp'):
                # Perform regression on params
                p_tilde = self.regression_model(z_tilde)
                # Regression loss
                reg_loss = loss_params(p_tilde, y)
            else:
                # Use log probability model
                p_tilde, reg_loss = self.regression_model.log_prob(z_tilde, y)
            
            # Final loss with better balancing
            b_loss = rec_loss + (args.beta * z_loss) + (args.gamma * reg_loss)
            
            # Check for NaN before backward pass
            if torch.isnan(b_loss) or torch.isinf(b_loss):
                print(f"Warning: NaN/Inf detected in loss, skipping batch")
                print(f"  rec_loss: {rec_loss.item():.6f}")
                print(f"  z_loss: {z_loss.item():.6f}")
                print(f"  reg_loss: {reg_loss.item():.6f}")
                continue
            
            # Perform backward
            optimizer.zero_grad()
            b_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
            
            # Check for NaN gradients
            has_nan_grad = False
            for param in self.parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                print(f"Warning: NaN/Inf gradients detected, skipping optimizer step")
                continue
            
            optimizer.step()
            full_loss += b_loss.detach()
            
        full_loss /= len(loader)
        return full_loss

def test_stable_training():
    # Same configuration as train_optimized.py
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

    # Construct loss functions
    if args.rec_loss == 'l1':
        rec_loss = nn.SmoothL1Loss(reduction='sum')
    else:
        rec_loss = nn.MSELoss(reduction='sum')

    if args.loss == 'l1':
        loss = nn.SmoothL1Loss(reduction='mean')
    else:
        loss = nn.MSELoss(reduction='mean')

    # Construct model
    print("Constructing stable model...")
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

    # Use the stable version
    model = StableRegressionAE(
        vae_model, args.latent_dims, args.output_size, 
        rec_loss, regressor=regression_model, regressor_name=args.regressor
    )

    # Setup optimizer with weight decay for stability
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print("\n=== TESTING STABLE TRAINING ===")
    
    # Test training for a few epochs
    for epoch in range(3):
        # Set beta and gamma like in training script
        beta = min(1, (epoch - args.warm_latent) / args.beta_factor) if epoch > args.warm_latent else 0
        gamma = 0
        if epoch >= args.start_regress:
            delta = min(1, (epoch - args.start_regress) / args.warm_regress)
            gamma = delta * args.reg_factor
        
        args.beta = beta
        args.gamma = gamma
        
        print(f"\nEpoch {epoch}: beta={beta:.3f}, gamma={gamma:.3f}")
        
        # Train epoch
        train_loss = model.train_epoch(train_loader, loss, optimizer, args)
        print(f"  Train loss: {train_loss.item():.6f}")
        
        # Eval epoch
        eval_loss = model.eval_epoch(valid_loader, loss, args)
        print(f"  Valid loss: {eval_loss.item():.6f}")
        
        # Check for NaN
        if torch.isnan(train_loss) or torch.isnan(eval_loss):
            print(f"  ERROR: NaN detected in epoch {epoch}!")
            break
        else:
            print(f"  Epoch {epoch} completed successfully")
    
    print("\nStable training test completed.")
    return model, args

if __name__ == "__main__":
    test_stable_training()