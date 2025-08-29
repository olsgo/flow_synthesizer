#!/usr/bin/env python3
"""
PolyMAX Flow Synthesizer Training with Cycle Loss

This script implements cycle-consistent training where:
1. Predictor: Audio -> Parameters
2. Surrogate: Parameters -> Audio
3. Cycle Loss: Audio -> Parameters -> Audio (should match original)
"""

import argparse
import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from pathlib import Path

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class PolyMAXDataset(Dataset):
    """Dataset for PolyMAX audio and parameter pairs"""
    
    def __init__(self, manifest_path, split='train', sample_rate=44100, duration=4.0):
        self.manifest = pd.read_csv(manifest_path)
        self.data = self.manifest[self.manifest['split'] == split].reset_index(drop=True)
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_length = int(sample_rate * duration)
        
        print(f"Loaded {len(self.data)} {split} samples")
    
    def __len__(self):
        return len(self.data)
    
    def load_audio(self, audio_path):
        """Load and preprocess audio"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            
            # Normalize
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            
            # Pad or truncate to target length
            if len(audio) < self.target_length:
                audio = np.pad(audio, (0, self.target_length - len(audio)))
            else:
                audio = audio[:self.target_length]
                
            return audio
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            return np.zeros(self.target_length)
    
    def compute_log_mel(self, audio):
        """Compute log-mel spectrogram"""
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=128,
            n_fft=2048,
            hop_length=512,
            fmax=8000
        )
        
        # Convert to log scale
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 1]
        log_mel = (log_mel + 80) / 80  # Assuming -80dB to 0dB range
        log_mel = np.clip(log_mel, 0, 1)
        
        return log_mel
    
    def load_parameters(self, params_path):
        """Load parameter vector from JSON"""
        try:
            with open(params_path, 'r') as f:
                data = json.load(f)
            
            params = np.array(data['parameter_vector'], dtype=np.float32)
            
            # Validate parameter count
            if len(params) != 66:
                print(f"Warning: Expected 66 parameters, got {len(params)} in {params_path}")
                # Pad or truncate to 66
                if len(params) < 66:
                    params = np.pad(params, (0, 66 - len(params)))
                else:
                    params = params[:66]
            
            return params
        except Exception as e:
            print(f"Error loading parameters {params_path}: {e}")
            return np.zeros(66, dtype=np.float32)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load audio and compute features
        audio = self.load_audio(row['audio_path'])
        log_mel = self.compute_log_mel(audio)
        
        # Load parameters
        params = self.load_parameters(row['params_path'])
        
        return {
            'audio': torch.FloatTensor(audio),
            'log_mel': torch.FloatTensor(log_mel),
            'params': torch.FloatTensor(params),
            'preset_name': row['preset_name']
        }

class PolyMAXPredictor(nn.Module):
    """CNN model to predict parameters from log-mel spectrograms"""
    
    def __init__(self, input_channels=128, output_dim=66):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim),
            nn.Sigmoid()  # Ensure output is in [0, 1] range
        )
    
    def forward(self, x):
        # Add channel dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

class PolyMAXSurrogate(nn.Module):
    """Surrogate model to generate audio from parameters"""
    
    def __init__(self, param_dim=66, audio_length=176400):
        super().__init__()
        self.audio_length = audio_length
        
        # Parameter encoder
        self.param_encoder = nn.Sequential(
            nn.Linear(param_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU()
        )
        
        # Audio decoder using transposed convolutions
        self.audio_decoder = nn.Sequential(
            nn.Linear(1024, 256 * 64),  # 16384
            nn.ReLU(),
            nn.Unflatten(1, (256, 64)),
            
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),  # 128
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),   # 256
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),    # 512
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),    # 1024
            nn.BatchNorm1d(16),
            nn.ReLU(),
            
            nn.ConvTranspose1d(16, 8, kernel_size=4, stride=2, padding=1),     # 2048
            nn.BatchNorm1d(8),
            nn.ReLU(),
            
            nn.ConvTranspose1d(8, 4, kernel_size=4, stride=2, padding=1),      # 4096
            nn.BatchNorm1d(4),
            nn.ReLU(),
            
            nn.ConvTranspose1d(4, 2, kernel_size=4, stride=2, padding=1),      # 8192
            nn.BatchNorm1d(2),
            nn.ReLU(),
            
            nn.ConvTranspose1d(2, 1, kernel_size=4, stride=2, padding=1),      # 16384
            nn.Tanh()  # Output in [-1, 1] range
        )
        
        # Final upsampling to target length
        self.final_upsample = nn.Upsample(size=audio_length, mode='linear', align_corners=False)
    
    def forward(self, params):
        x = self.param_encoder(params)
        x = self.audio_decoder(x)
        x = self.final_upsample(x)
        return x.squeeze(1)  # Remove channel dimension

def create_dataloaders(manifest_path, batch_size=16):
    """Create train and validation dataloaders"""
    train_dataset = PolyMAXDataset(manifest_path, split='train')
    val_dataset = PolyMAXDataset(manifest_path, split='val')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader

def train_epoch_cycle(predictor, surrogate, train_loader, optimizer_pred, optimizer_surr, 
                     criterion_mse, criterion_audio, device, lambda_cycle=1.0):
    """Train one epoch with cycle loss"""
    predictor.train()
    surrogate.train()
    
    total_pred_loss = 0
    total_cycle_loss = 0
    total_loss = 0
    
    for batch in train_loader:
        audio = batch['audio'].to(device)
        log_mel = batch['log_mel'].to(device)
        params_true = batch['params'].to(device)
        
        # Zero gradients
        optimizer_pred.zero_grad()
        optimizer_surr.zero_grad()
        
        # Forward pass: Audio -> Parameters
        params_pred = predictor(log_mel)
        pred_loss = criterion_mse(params_pred, params_true)
        
        # Cycle: Audio -> Parameters -> Audio
        audio_reconstructed = surrogate(params_pred)
        cycle_loss = criterion_audio(audio_reconstructed, audio)
        
        # Total loss
        loss = pred_loss + lambda_cycle * cycle_loss
        
        # Backward pass
        loss.backward()
        optimizer_pred.step()
        optimizer_surr.step()
        
        total_pred_loss += pred_loss.item()
        total_cycle_loss += cycle_loss.item()
        total_loss += loss.item()
    
    return {
        'total_loss': total_loss / len(train_loader),
        'pred_loss': total_pred_loss / len(train_loader),
        'cycle_loss': total_cycle_loss / len(train_loader)
    }

def validate_epoch_cycle(predictor, surrogate, val_loader, criterion_mse, criterion_audio, 
                        device, lambda_cycle=1.0):
    """Validate one epoch with cycle loss"""
    predictor.eval()
    surrogate.eval()
    
    total_pred_loss = 0
    total_cycle_loss = 0
    total_loss = 0
    all_true = []
    all_pred = []
    
    with torch.no_grad():
        for batch in val_loader:
            audio = batch['audio'].to(device)
            log_mel = batch['log_mel'].to(device)
            params_true = batch['params'].to(device)
            
            # Forward pass
            params_pred = predictor(log_mel)
            pred_loss = criterion_mse(params_pred, params_true)
            
            # Cycle loss
            audio_reconstructed = surrogate(params_pred)
            cycle_loss = criterion_audio(audio_reconstructed, audio)
            
            # Total loss
            loss = pred_loss + lambda_cycle * cycle_loss
            
            total_pred_loss += pred_loss.item()
            total_cycle_loss += cycle_loss.item()
            total_loss += loss.item()
            
            # Collect for MAE calculation
            all_true.extend(params_true.cpu().numpy())
            all_pred.extend(params_pred.cpu().numpy())
    
    mae = mean_absolute_error(np.array(all_true), np.array(all_pred))
    
    return {
        'total_loss': total_loss / len(val_loader),
        'pred_loss': total_pred_loss / len(val_loader),
        'cycle_loss': total_cycle_loss / len(val_loader),
        'mae': mae
    }

def main():
    parser = argparse.ArgumentParser(description='Train PolyMAX Flow Synthesizer with Cycle Loss')
    parser.add_argument('--manifest_path', type=str, 
                       default='/Users/gjb/Projects/flow_synthesizer/data/manifest.csv',
                       help='Path to manifest CSV file')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lambda_cycle', type=float, default=10.0, help='Cycle loss weight')
    parser.add_argument('--load_predictor', type=str, 
                       default='/Users/gjb/Projects/flow_synthesizer/code/outputs/models/polymax_predictor_best.pth',
                       help='Path to pretrained predictor model')
    
    args = parser.parse_args()
    
    # Setup - Force MPS (M1 Max GPU) usage only
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS (M1 Max GPU) is not available! Training requires M1 Max GPU.")
    
    device = torch.device('mps')
    print(f"Using device: {device} (M1 Max GPU)")
    
    # Create output directories
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/plots', exist_ok=True)
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(args.manifest_path, args.batch_size)
    
    # Initialize models
    print("Initializing models...")
    predictor = PolyMAXPredictor().to(device)
    surrogate = PolyMAXSurrogate().to(device)
    
    # Load pretrained predictor if available
    if os.path.exists(args.load_predictor):
        print(f"Loading pretrained predictor from {args.load_predictor}")
        checkpoint = torch.load(args.load_predictor, map_location=device)
        if 'model_state_dict' in checkpoint:
            predictor.load_state_dict(checkpoint['model_state_dict'])
        else:
            predictor.load_state_dict(checkpoint)
    
    # Loss functions
    criterion_mse = nn.MSELoss()
    criterion_audio = nn.L1Loss()  # L1 loss for audio reconstruction
    
    # Optimizers
    optimizer_pred = optim.Adam(predictor.parameters(), lr=args.lr)
    optimizer_surr = optim.Adam(surrogate.parameters(), lr=args.lr)
    
    # Learning rate schedulers
    scheduler_pred = optim.lr_scheduler.ReduceLROnPlateau(optimizer_pred, patience=10, factor=0.5)
    scheduler_surr = optim.lr_scheduler.ReduceLROnPlateau(optimizer_surr, patience=10, factor=0.5)
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20
    
    train_losses = []
    val_losses = []
    
    for epoch in range(args.epochs):
        # Train
        train_metrics = train_epoch_cycle(
            predictor, surrogate, train_loader, 
            optimizer_pred, optimizer_surr,
            criterion_mse, criterion_audio, device, args.lambda_cycle
        )
        
        # Validate
        val_metrics = validate_epoch_cycle(
            predictor, surrogate, val_loader,
            criterion_mse, criterion_audio, device, args.lambda_cycle
        )
        
        train_losses.append(train_metrics['total_loss'])
        val_losses.append(val_metrics['total_loss'])
        
        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"Train Loss: {train_metrics['total_loss']:.6f} "
              f"(Pred: {train_metrics['pred_loss']:.6f}, Cycle: {train_metrics['cycle_loss']:.6f}), "
              f"Val Loss: {val_metrics['total_loss']:.6f} "
              f"(Pred: {val_metrics['pred_loss']:.6f}, Cycle: {val_metrics['cycle_loss']:.6f}), "
              f"Val MAE: {val_metrics['mae']:.6f}")
        
        # Learning rate scheduling
        scheduler_pred.step(val_metrics['pred_loss'])
        scheduler_surr.step(val_metrics['cycle_loss'])
        
        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            torch.save(predictor.state_dict(), 'outputs/models/polymax_predictor_cycle_best.pth')
            torch.save(surrogate.state_dict(), 'outputs/models/polymax_surrogate_best.pth')
            print(f"Saved best models with val_loss: {best_val_loss:.6f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Predictor model saved to: {os.path.abspath('outputs/models/polymax_predictor_cycle_best.pth')}")
    print(f"Surrogate model saved to: {os.path.abspath('outputs/models/polymax_surrogate_best.pth')}")

if __name__ == '__main__':
    main()