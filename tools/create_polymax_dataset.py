#!/usr/bin/env python3
"""
PolyMAX Dataset Creation Script

This script processes the rendered PolyMAX audio files and parameter JSON files
to create the dataset structure expected by the flow synthesizer training pipeline.
"""

import os
import json
import numpy as np
import librosa
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm

def load_audio_and_extract_features(audio_path, sr=22050, n_mels=128, n_mfcc=13, hop_length=512, n_fft=2048):
    """
    Load audio file and extract mel spectrogram and MFCC features.
    """
    try:
        # Load audio
        y, _ = librosa.load(audio_path, sr=sr, duration=4.0)  # Load 4 seconds
        
        # Ensure consistent length (4 seconds)
        target_length = sr * 4
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode='constant')
        else:
            y = y[:target_length]
        
        # Extract mel spectrogram (raw power, not dB - log transform applied later in pipeline)
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft
        )
        
        # Extract MFCC
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft
        )
        
        return {
            'audio': y,
            'mel': mel_spec,
            'mfcc': mfcc,
            'sr': sr
        }
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def create_dataset_from_manifest(manifest_path, output_dir, audio_base_path=None):
    """
    Create dataset from manifest CSV file.
    """
    # Read manifest
    df = pd.read_csv(manifest_path)
    
    # Create output directories
    raw_dir = Path(output_dir) / 'raw'
    mel_dir = Path(output_dir) / 'mel'
    mfcc_dir = Path(output_dir) / 'mfcc'
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    mel_dir.mkdir(parents=True, exist_ok=True)
    mfcc_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {len(df)} files...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # Get file paths
            audio_path = row['audio_path']
            params_path = row['params_path']
            stem = row['stem']
            
            # Override audio path if base path provided
            if audio_base_path:
                audio_path = os.path.join(audio_base_path, row['audio_file'])
            
            # Check if files exist
            if not os.path.exists(audio_path):
                print(f"Warning: Audio file not found: {audio_path}")
                continue
                
            if not os.path.exists(params_path):
                print(f"Warning: Params file not found: {params_path}")
                continue
            
            # Load parameters
            with open(params_path, 'r') as f:
                params_data = json.load(f)
            
            # Extract parameter vector
            if 'parameter_vector' in params_data:
                params = np.array(params_data['parameter_vector'], dtype=np.float32)
            else:
                print(f"Warning: No parameter_vector found in {params_path}")
                continue
            
            # Extract audio features
            features = load_audio_and_extract_features(audio_path)
            if features is None:
                continue
            
            # Create metadata
            metadata = {
                'preset_name': params_data.get('preset_name', stem),
                'stem': stem,
                'split': row['split'],
                'audio_file': row['audio_file'],
                'param_file': row['param_file']
            }
            
            # Save raw data (audio + params + metadata)
            raw_data = {
                'audio': features['audio'],
                'params': params,
                'metadata': metadata,
                'sr': features['sr']
            }
            np.savez_compressed(raw_dir / f"{stem}.npz", **raw_data)
            
            # Save mel spectrogram
            np.save(mel_dir / f"{stem}.npy", features['mel'])
            
            # Save MFCC
            np.save(mfcc_dir / f"{stem}.npy", features['mfcc'])
            
        except Exception as e:
            print(f"Error processing {stem}: {e}")
            continue
    
    print(f"Dataset creation complete! Files saved to {output_dir}")
    print(f"Raw files: {len(list(raw_dir.glob('*.npz')))}")
    print(f"Mel files: {len(list(mel_dir.glob('*.npy')))}")
    print(f"MFCC files: {len(list(mfcc_dir.glob('*.npy')))}")

def main():
    parser = argparse.ArgumentParser(description='Create PolyMAX dataset for flow synthesizer training')
    parser.add_argument('--manifest', type=str, default='manifest.csv',
                       help='Path to manifest CSV file')
    parser.add_argument('--output', type=str, default='datasets/polymax_dataset',
                       help='Output directory for dataset')
    parser.add_argument('--audio_base', type=str, default=None,
                       help='Base path for audio files (if different from manifest)')
    
    args = parser.parse_args()
    
    create_dataset_from_manifest(args.manifest, args.output, args.audio_base)

if __name__ == '__main__':
    main()