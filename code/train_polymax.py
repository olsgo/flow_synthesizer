import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import json
import librosa
from torch.utils.data import Dataset, DataLoader

class PolyMAXDataset(Dataset):
    def __init__(self, manifest_path, split='train', sr=44100, duration=4.0, n_mels=64, n_fft=2048, hop_length=512):
        self.manifest = pd.read_csv(manifest_path)
        self.manifest = self.manifest[self.manifest['split'] == split]
        self.sr = sr
        self.duration = duration
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_length = int(sr * duration)

    def __len__(self):
        return len(self.manifest)

    def load_audio(self, audio_path):
        """Load and preprocess audio file."""
        # Load audio
        audio, _ = librosa.load(audio_path, sr=self.sr, mono=True)
        
        # Normalize to target length
        if len(audio) > self.target_length:
            audio = audio[:self.target_length]
        elif len(audio) < self.target_length:
            audio = np.pad(audio, (0, self.target_length - len(audio)), mode='constant')
        
        # Peak normalize to 0.9
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.9
        
        return audio
    
    def compute_log_mel(self, audio):
        """Compute log-mel spectrogram."""
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=30,
            fmax=self.sr // 2
        )
        
        # Convert to log scale
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        
        return log_mel
    
    def load_params(self, params_path):
        """Load parameter vector from JSON file."""
        with open(params_path, 'r') as f:
            data = json.load(f)
        
        # Extract parameter vector
        if 'parameter_vector' in data:
            params = np.array(data['parameter_vector'], dtype=np.float32)
        else:
            raise ValueError(f"No 'parameter_vector' found in {params_path}")
        
        # Ensure parameters are in [0, 1] range
        params = np.clip(params, 0.0, 1.0)
        
        return params

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        
        # Load and process audio
        audio = self.load_audio(row['audio_path'])
        log_mel = self.compute_log_mel(audio)
        
        # Load parameters
        params = self.load_params(row['params_path'])
        
        return {
            'log_mel': torch.FloatTensor(log_mel),
            'params': torch.FloatTensor(params),
            'preset_name': row['preset_name']
        }

class PolyMAXPredictor(nn.Module):
    """Simple CNN-based model for predicting PolyMAX parameters from log-mel spectrograms."""
    
    def __init__(self, n_mels=64, n_params=66):
        super(PolyMAXPredictor, self).__init__()
        
        # CNN layers for feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, n_params),
            nn.Sigmoid()  # Ensure output is in [0, 1] range
        )
    
    def forward(self, x):
        # Add channel dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # CNN feature extraction
        x = self.conv_layers(x)
        
        # Flatten for FC layers
        x = x.view(x.size(0), -1)
        
        # Predict parameters
        x = self.fc_layers(x)
        
        return x

def create_dataloaders(manifest_path, batch_size):
    train_dataset = PolyMAXDataset(manifest_path, split='train')
    val_dataset = PolyMAXDataset(manifest_path, split='val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in train_loader:
        log_mel = batch['log_mel'].to(device)
        params = batch['params'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        pred_params = model(log_mel)
        loss = criterion(pred_params, params)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            log_mel = batch['log_mel'].to(device)
            params = batch['params'].to(device)
            
            # Forward pass
            pred_params = model(log_mel)
            loss = criterion(pred_params, params)
            mae = torch.mean(torch.abs(pred_params - params))
            
            total_loss += loss.item()
            total_mae += mae.item()
            num_batches += 1
    
    return total_loss / num_batches, total_mae / num_batches

def main():
    parser = argparse.ArgumentParser(description='Train PolyMAX model')
    parser.add_argument('--manifest_path', type=str, default='/Users/gjb/Projects/flow_synthesizer/data/manifest.csv', help='Path to the manifest file')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='/Users/gjb/Projects/flow_synthesizer/code/outputs/models', help='Directory to save models')
    args = parser.parse_args()

    # Setup - Force MPS (M1 Max GPU) usage only
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS (M1 Max GPU) is not available! Training requires M1 Max GPU.")
    
    device = torch.device('mps')
    print(f"Using device: {device} (M1 Max GPU)")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(args.manifest_path, args.batch_size)
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
    
    # Create model
    model = PolyMAXPredictor(n_mels=64, n_params=66).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20
    
    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_mae = validate_epoch(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"Train Loss: {train_loss:.6f}, "
              f"Val Loss: {val_loss:.6f}, "
              f"Val MAE: {val_mae:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save model
            import os
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae
            }, f"{args.save_dir}/polymax_predictor_best.pth")
            print(f"Saved best model with val_loss: {val_loss:.6f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break
    
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {args.save_dir}/polymax_predictor_best.pth")

if __name__ == '__main__':
    main()