#!/usr/bin/env python3

import sys
sys.path.append('/Users/gjb/Projects/flow_synthesizer/code')

import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

print("Fixing NaN/Inf values in mel data files...")

# Path to mel data directory
mel_dir = Path('/Users/gjb/Projects/flow_synthesizer/code/datasets/polymax_dataset/mel')

if not mel_dir.exists():
    print(f"Error: Mel directory not found: {mel_dir}")
    sys.exit(1)

# Get all .npy files
mel_files = list(mel_dir.glob('*.npy'))
print(f"Found {len(mel_files)} mel files to check")

fixed_count = 0
total_nan_count = 0
total_inf_count = 0

for mel_file in tqdm(mel_files, desc="Processing mel files"):
    try:
        # Load the mel data
        mel_data = np.load(mel_file)
        
        # Check for NaN and Inf values
        nan_mask = np.isnan(mel_data)
        inf_mask = np.isinf(mel_data)
        
        nan_count = np.sum(nan_mask)
        inf_count = np.sum(inf_mask)
        
        if nan_count > 0 or inf_count > 0:
            print(f"\nFixing {mel_file.name}: {nan_count} NaN, {inf_count} Inf values")
            
            # Create a copy for fixing
            fixed_data = mel_data.copy()
            
            # Replace NaN values with zeros
            if nan_count > 0:
                fixed_data[nan_mask] = 0.0
                total_nan_count += nan_count
            
            # Replace Inf values with the maximum finite value in the array
            if inf_count > 0:
                finite_mask = np.isfinite(mel_data)
                if np.any(finite_mask):
                    max_finite = np.max(mel_data[finite_mask])
                    min_finite = np.min(mel_data[finite_mask])
                    
                    # Replace +Inf with max finite value
                    pos_inf_mask = np.isposinf(mel_data)
                    fixed_data[pos_inf_mask] = max_finite
                    
                    # Replace -Inf with min finite value
                    neg_inf_mask = np.isneginf(mel_data)
                    fixed_data[neg_inf_mask] = min_finite
                else:
                    # If no finite values, replace with zeros
                    fixed_data[inf_mask] = 0.0
                
                total_inf_count += inf_count
            
            # Verify the fix
            if np.any(np.isnan(fixed_data)) or np.any(np.isinf(fixed_data)):
                print(f"Warning: {mel_file.name} still contains NaN/Inf after fixing")
            else:
                # Save the fixed data
                np.save(mel_file, fixed_data)
                fixed_count += 1
                
    except Exception as e:
        print(f"Error processing {mel_file.name}: {e}")

print(f"\nSummary:")
print(f"  Files processed: {len(mel_files)}")
print(f"  Files fixed: {fixed_count}")
print(f"  Total NaN values replaced: {total_nan_count}")
print(f"  Total Inf values replaced: {total_inf_count}")

if fixed_count > 0:
    print(f"\nFixed {fixed_count} mel files. You should now be able to train without NaN loss.")
else:
    print(f"\nNo files needed fixing.")

print("\nDone!")