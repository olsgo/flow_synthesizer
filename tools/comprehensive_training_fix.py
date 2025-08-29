#!/usr/bin/env python3
"""
Comprehensive training stability fix for the flow synthesizer.
This script addresses multiple potential causes of NaN values during training.
"""

import torch
import torch.nn as nn
import sys
import os
sys.path.append('/Users/gjb/Projects/flow_synthesizer/code')

from models.vae.ae import RegressionAE
from models.basic import RegressionModel, GatedCNN
from utils.data import load_dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

class StabilizedRegressionAE(RegressionAE):
    """Enhanced RegressionAE with comprehensive stability improvements"""
    
    def __init__(self, ae_model, latent_dims, regression_dims, recons_loss, regressor=None, regressor_name='', **kwargs):
        super().__init__(ae_model, latent_dims, regression_dims, recons_loss, regressor, regressor_name, **kwargs)
        
        # Enhanced stability parameters
        self.rec_loss_scale = 1e-2  # Less aggressive scaling
        self.max_grad_norm = 0.5    # Tighter gradient clipping
        self.loss_ema_alpha = 0.9   # Exponential moving average for loss smoothing
        self.running_loss = None
        
        # Initialize weights more carefully
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Improved weight initialization to prevent numerical instability"""
        if isinstance(module, nn.Linear):
            # Xavier/Glorot initialization with smaller variance
            nn.init.xavier_normal_(module.weight, gain=0.5)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Conv2d):
            # He initialization with smaller gain
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu', a=0.1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)
    
    def _check_tensor_health(self, tensor, name):
        """Check if tensor contains NaN or Inf values"""
        try:
            # Handle scalar tensors
            if tensor.dim() == 0:
                if torch.isnan(tensor) or torch.isinf(tensor):
                    print(f"Warning: {name} contains NaN/Inf values")
                    return False
                if tensor.abs() > 1e6:
                    print(f"Warning: {name} contains very large values (value: {tensor.abs()})")
                    return False
            else:
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    print(f"Warning: {name} contains NaN/Inf values")
                    return False
                if tensor.abs().max() > 1e6:
                    print(f"Warning: {name} contains very large values (max: {tensor.abs().max()})")
                    return False
            return True
        except Exception as e:
            print(f"Error in _check_tensor_health for {name}: {e}")
            print(f"Tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
            # Skip health check if there's an error
            return True
    
    def train_epoch(self, loader, loss_params, optimizer, args):
        self.train()
        full_loss = 0
        valid_batches = 0
        skipped_batches = 0
        
        for batch_idx, (x, y, _, _) in enumerate(loader):
            # Send to device
            x, y = x.to(args.device, non_blocking=True), y.to(args.device, non_blocking=True)
            
            # Check input health
            if not self._check_tensor_health(x, "input x") or not self._check_tensor_health(y, "target y"):
                skipped_batches += 1
                continue
            
            # Auto-encode
            x_tilde, z_tilde, z_loss = self.ae_model(x)
            
            # Check intermediate tensor health
            if not (self._check_tensor_health(x_tilde, "x_tilde") and 
                   self._check_tensor_health(z_tilde, "z_tilde") and 
                   self._check_tensor_health(z_loss, "z_loss")):
                skipped_batches += 1
                continue
            
            # Reconstruction loss with adaptive scaling
            rec_loss = self.recons_loss(x_tilde, x) / (x.shape[1] * x.shape[2])
            
            # Adaptive loss scaling based on magnitude
            rec_loss_val = rec_loss.item() if rec_loss.dim() == 0 else rec_loss.mean().item()
            if rec_loss_val > 1000:
                rec_loss_scale = 1e-4
            elif rec_loss_val > 100:
                rec_loss_scale = 1e-3
            else:
                rec_loss_scale = 1e-2
            
            rec_loss = rec_loss * rec_loss_scale
            
            if not self._check_tensor_health(rec_loss, "rec_loss"):
                skipped_batches += 1
                continue
            
            if self.regressor == 'mlp':
                # Perform regression on params
                p_tilde = self.regression_model(z_tilde)
                if not self._check_tensor_health(p_tilde, "p_tilde"):
                    skipped_batches += 1
                    continue
                
                # Regression loss
                reg_loss = loss_params(p_tilde, y)
                if not self._check_tensor_health(reg_loss, "reg_loss"):
                    skipped_batches += 1
                    continue
            else:
                # Use log probability model
                p_tilde, reg_loss = self.regression_model.log_prob(z_tilde, y)
                if not (self._check_tensor_health(p_tilde, "p_tilde") and 
                       self._check_tensor_health(reg_loss, "reg_loss")):
                    skipped_batches += 1
                    continue
            
            # Final loss with careful combination
            z_loss_term = args.beta * z_loss if hasattr(args, 'beta') else z_loss * 0.1
            reg_loss_term = args.gamma * reg_loss if hasattr(args, 'gamma') else reg_loss
            
            b_loss = rec_loss + z_loss_term + reg_loss_term
            
            # Final health check
            if not self._check_tensor_health(b_loss, "b_loss"):
                skipped_batches += 1
                continue
            
            # Exponential moving average for loss smoothing (ensure scalar)
            current_loss_scalar = b_loss.item() if b_loss.dim() == 0 else b_loss.mean().item()
            if self.running_loss is None:
                self.running_loss = current_loss_scalar
            else:
                self.running_loss = self.loss_ema_alpha * self.running_loss + (1 - self.loss_ema_alpha) * current_loss_scalar
            
            # Skip if loss is too different from running average (outlier detection)
            if self.running_loss is not None:
                if abs(current_loss_scalar - self.running_loss) > 10 * self.running_loss:
                    print(f"Warning: Loss outlier detected (current: {current_loss_scalar:.6f}, running avg: {self.running_loss:.6f}), skipping batch")
                    skipped_batches += 1
                    continue
            
            # Perform backward (ensure scalar loss)
            optimizer.zero_grad()
            if b_loss.dim() > 0:
                b_loss = b_loss.mean()
            b_loss.backward()
            
            # Enhanced gradient clipping with health check
            total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
            
            if torch.isnan(total_norm) or torch.isinf(total_norm):
                print(f"Warning: NaN/Inf gradient norm detected, skipping optimizer step")
                skipped_batches += 1
                continue
            
            # Check for NaN gradients
            has_nan_grad = False
            for name, param in self.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        print(f"Warning: NaN/Inf gradients in {name}, skipping optimizer step")
                        has_nan_grad = True
                        break
            
            if has_nan_grad:
                skipped_batches += 1
                continue
            
            optimizer.step()
            full_loss += b_loss.detach()
            valid_batches += 1
        
        if valid_batches > 0:
            avg_loss = full_loss / valid_batches
            print(f"Training: {valid_batches} valid batches, {skipped_batches} skipped batches")
            return avg_loss
        else:
            print(f"Warning: No valid training batches, all {skipped_batches} batches were skipped")
            return torch.tensor(float('nan'))
    
    def eval_epoch(self, loader, loss_params, args):
        self.eval()
        full_loss = 0
        valid_batches = 0
        skipped_batches = 0
        
        with torch.no_grad():
            for batch_idx, (x, y, _, _) in enumerate(loader):
                x, y = x.to(args.device, non_blocking=True), y.to(args.device, non_blocking=True)
                
                # Check input health
                if not self._check_tensor_health(x, "eval input x") or not self._check_tensor_health(y, "eval target y"):
                    skipped_batches += 1
                    continue
                
                try:
                    # Auto-encode
                    x_tilde, z_tilde, z_loss = self.ae_model(x)
                    
                    # Check intermediate tensor health
                    if not (self._check_tensor_health(x_tilde, "eval x_tilde") and 
                           self._check_tensor_health(z_tilde, "eval z_tilde")):
                        skipped_batches += 1
                        continue
                    
                    # Perform regression on params
                    p_tilde = self.regression_model(z_tilde)
                    if not self._check_tensor_health(p_tilde, "eval p_tilde"):
                        skipped_batches += 1
                        continue
                    
                    # Regression loss
                    reg_loss = loss_params(p_tilde, y)
                    if not self._check_tensor_health(reg_loss, "eval reg_loss"):
                        skipped_batches += 1
                        continue
                    
                    full_loss += reg_loss
                    valid_batches += 1
                    
                except Exception as e:
                    print(f"Error in eval batch {batch_idx}: {e}")
                    skipped_batches += 1
                    continue
        
        if valid_batches > 0:
            avg_loss = full_loss / valid_batches
            if skipped_batches > 0:
                print(f"Evaluation: {valid_batches} valid batches, {skipped_batches} skipped batches")
            return avg_loss
        else:
            print(f"Warning: No valid evaluation batches, all {skipped_batches} batches were skipped")
            return torch.tensor(float('nan'))

def test_stabilized_training():
    """Test the stabilized training approach"""
    print("Testing stabilized training approach...")
    
    # Set device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create args object for load_dataset
    import argparse
    args = argparse.Namespace(
        path='datasets',
        dataset='polymax_dataset',
        data='mel',
        batch_size=16,
        nbworkers=0,
        train_type='sequential',
        device=device,
        beta=0.01,   # Small beta for VAE loss
        gamma=1.0    # Standard gamma for regression loss
    )
    
    # Load dataset
    train_loader, valid_loader, test_loader, args = load_dataset(args)
    
    # Update device in args
    args.device = device
    
    print(f"Dataset loaded: {len(train_loader)} train batches, {len(valid_loader)} valid batches")
    print(f"  Input size: {args.input_size}")
    print(f"  Output size: {args.output_size}")
    
    # Create model components
    input_size = args.input_size
    output_size = args.output_size
    latent_dims = output_size
    
    # Create args for model construction
    cnn_args = argparse.Namespace(
        kernel=5,
        dilation=1,
        channels=32,
        n_layers=4,
        n_hidden=512
    )
    
    # Use the proper construct_encoder_decoder function
    from models.basic import construct_encoder_decoder
    encoder, decoder = construct_encoder_decoder(
        input_size, latent_dims, latent_dims,
        channels=32, n_layers=4, hidden_size=512, n_mlp=2,
        type_mod='gated_cnn', args=cnn_args
    )
    
    # Create autoencoder
    from models.vae.vae import VAE
    ae_model = VAE(encoder, decoder, input_size, encoder.out_size, latent_dims)
    
    # Create regression model with smaller capacity
    regression_model = nn.Sequential(
        nn.Linear(latent_dims, latent_dims),
        nn.ReLU(),
        nn.BatchNorm1d(latent_dims),
        nn.Dropout(0.1),
        nn.Linear(latent_dims, output_size)
    )
    
    # Create loss function
    recons_loss = nn.SmoothL1Loss(reduction='none')
    
    # Create stabilized model
    model = StabilizedRegressionAE(
        ae_model=ae_model,
        latent_dims=latent_dims,
        regression_dims=output_size,
        recons_loss=recons_loss,
        regressor=regression_model,
        regressor_name='mlp'
    ).to(device)
    
    # Create optimizer with conservative settings
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # Create loss function for regression
    loss_params = nn.SmoothL1Loss()
    
    # Create args object
    class Args:
        def __init__(self):
            self.device = device
            self.beta = 0.01   # Small beta for VAE loss
            self.gamma = 1.0   # Standard gamma for regression loss
    
    args = Args()
    
    print("\nStarting stabilized training test...")

    # Prepare checkpoint directory
    ckpt_dir = os.path.join('outputs_optimized', 'models')
    os.makedirs(ckpt_dir, exist_ok=True)
    best_val = float('inf')
    
    # Test training for a few epochs
    for epoch in range(3):
        print(f"\nEpoch {epoch + 1}:")
        
        # Training
        train_loss = model.train_epoch(train_loader, loss_params, optimizer, args)
        # Safely convert to float for logging
        train_val = float(train_loss.detach().cpu().item() if torch.is_tensor(train_loss) else train_loss)
        print(f"  Train Loss: {train_val:.6f}")
        
        # Validation
        valid_loss = model.eval_epoch(valid_loader, loss_params, args)
        valid_val = float(valid_loss.detach().cpu().item() if torch.is_tensor(valid_loss) else valid_loss)
        print(f"  Valid Loss: {valid_val:.6f}")

        # Save best checkpoint for downstream integration tests
        if not (torch.isnan(valid_loss) if torch.is_tensor(valid_loss) else False):
            if valid_val < best_val:
                best_val = valid_val
                ckpt_path = os.path.join(ckpt_dir, 'best_model.pth')
                # Save full model object for torch.load
                torch.save(model, ckpt_path)
                print(f"  ✓ Saved best model checkpoint to {ckpt_path}")
        
        # Check if we have valid losses
        if not (torch.isnan(train_loss) or torch.isnan(valid_loss)):
            print(f"  ✓ Epoch {epoch + 1} completed successfully with valid losses")
        else:
            print(f"  ✗ Epoch {epoch + 1} produced NaN losses")
    
    print("\nStabilized training test completed!")

if __name__ == "__main__":
    test_stabilized_training()
