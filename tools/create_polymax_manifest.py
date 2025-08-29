#!/usr/bin/env python3
"""
Create Manifest CSV for PolyMAX Dataset

This script creates a manifest.csv file from the rendered PolyMAX dataset
at /Users/gjb/Datasets/polymax/render with train/val/test splits.
"""

import os
import csv
import random
import glob
import argparse
from pathlib import Path

def create_manifest(dataset_dir, output_path, train_split=0.9, val_split=0.05):
    """
    Create manifest CSV from PolyMAX dataset.
    
    Args:
        dataset_dir: Path to dataset directory containing .wav and .json files
        output_path: Path to save manifest.csv
        train_split: Fraction for training (default 0.9)
        val_split: Fraction for validation (default 0.05)
    """
    print(f"Creating manifest from dataset: {dataset_dir}")
    
    # Find all WAV files
    wav_pattern = os.path.join(dataset_dir, '*.wav')
    wavs = sorted(glob.glob(wav_pattern))
    print(f"Found {len(wavs)} WAV files")
    
    # Match with JSON files
    rows = []
    for w in wavs:
        stem = os.path.splitext(os.path.basename(w))[0]
        j = os.path.join(dataset_dir, f'{stem}.json')
        
        if os.path.exists(j):
            rows.append((stem, w, j))
        else:
            print(f"Warning: No matching JSON for {stem}.wav")
    
    print(f"Matched {len(rows)} audio-parameter pairs")
    
    if len(rows) == 0:
        print("Error: No matching files found!")
        return False
    
    # Shuffle for random splits
    random.Random(42).shuffle(rows)
    
    # Calculate split indices
    N = len(rows)
    n_tr = int(train_split * N)
    n_v = int(val_split * N)
    
    print(f"Split: {n_tr} train, {n_v} val, {N - n_tr - n_v} test")
    
    # Create manifest CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['split', 'audio_path', 'params_path', 'preset_name'])
        
        for i, (stem, w, j) in enumerate(rows):
            if i < n_tr:
                split = 'train'
            elif i < n_tr + n_v:
                split = 'val'
            else:
                split = 'test'
            
            writer.writerow([split, w, j, stem])
    
    print(f"Manifest created: {output_path}")
    print(f"Total entries: {len(rows)}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Create PolyMAX dataset manifest')
    parser.add_argument('--dataset_dir', type=str, 
                       default='/Users/gjb/Datasets/polymax/render',
                       help='Path to dataset directory')
    parser.add_argument('--output_path', type=str,
                       default='/Users/gjb/Projects/flow_synthesizer/data/manifest.csv',
                       help='Path to save manifest.csv')
    parser.add_argument('--train_split', type=float, default=0.9,
                       help='Fraction for training split')
    parser.add_argument('--val_split', type=float, default=0.05,
                       help='Fraction for validation split')
    
    args = parser.parse_args()
    
    # Validate dataset directory exists
    if not os.path.exists(args.dataset_dir):
        print(f"Error: Dataset directory does not exist: {args.dataset_dir}")
        return 1
    
    # Create manifest
    success = create_manifest(
        args.dataset_dir, 
        args.output_path, 
        args.train_split, 
        args.val_split
    )
    
    if success:
        print("\n=== Manifest Creation Complete ===")
        print(f"Ready to train with: python code/train_polymax.py --manifest_path {args.output_path}")
        return 0
    else:
        print("\n=== Manifest Creation Failed ===")
        return 1

if __name__ == '__main__':
    exit(main())