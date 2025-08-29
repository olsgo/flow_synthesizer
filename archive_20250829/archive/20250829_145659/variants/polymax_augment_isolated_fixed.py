#!/usr/bin/env python3
"""
Fixed isolated augmentation driver: generates PolyMAX format compatible files.
Saves parameters as vector (params key) instead of dictionary (param key).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf
import librosa

# Add code directory to path
sys.path.append('code')
from synth.synthesize import create_synth, midiname2num


def validate_wav(path: Path, min_rms: float = 0.005, max_peak: float = 0.99, tail_secs: float = 0.1) -> bool:
    try:
        y, sr = sf.read(str(path))
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = y.astype(np.float32)
        if len(y) < 10:
            return False
        rms = float(np.sqrt(np.mean(y**2)))
        peak = float(np.max(np.abs(y)))
        tail_n = max(1, int(sr * tail_secs))
        tail_rms = float(np.sqrt(np.mean(y[-tail_n:]**2)))
        # Require adequate RMS, not clipping, and tail RMS not absurdly larger than overall
        if rms < min_rms:
            return False
        if peak > max_peak:
            return False
        if tail_rms > 3.0 * rms:
            return False
        return True
    except Exception:
        return False


def load_param_schema() -> List[str]:
    """Load parameter names in correct order"""
    with open('params_schema.json', 'r') as f:
        schema = json.load(f)
    return schema.get('parameter_order', [])


def generate_sample(outdir: Path, sample_id: str, param_names: List[str], 
                   duration: float, note: int, velocity: int, plugin_path: str,
                   strategy: str, seed: int) -> bool:
    """Generate a single augmented sample in PolyMAX format"""
    try:
        # Set random seed for this sample
        np.random.seed(seed)
        
        # Generate random parameters (0-1 range)
        if strategy == 'uniform':
            params_vector = np.random.uniform(0.0, 1.0, len(param_names))
        elif strategy == 'jitter':
            # Start from center and add jitter
            params_vector = np.random.normal(0.5, 0.2, len(param_names))
            params_vector = np.clip(params_vector, 0.0, 1.0)
        else:
            params_vector = np.random.uniform(0.0, 1.0, len(param_names))
        
        # Initialize synthesizer
        synth_engine, synth_generator, param_defaults_synth, rev_idx = create_synth('polymax_dataset', 'polymax', path=plugin_path)
        
        # Create parameter dictionary for synthesis using defaults
        params_dict = param_defaults_synth.copy()
        for name, val in zip(param_names, params_vector):
            if name in params_dict:
                params_dict[name] = float(val)
        
        # Set parameters using synthesize_audio function
        from synth.synthesize import synthesize_audio, midiname2num
        
        # Use the synthesize_audio function from the existing codebase
        audio = synthesize_audio(params_dict, synth_engine, synth_generator, rev_idx)
        
        # Resample to 22050 Hz if needed
        from synth.synthesize import resample
        audio = resample(audio, 44100, 22050)
        
        # Validate audio
        if len(audio) == 0 or np.all(audio == 0):
            return False
        
        # Ensure mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        
        # Normalize
        audio = audio.astype(np.float32)
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.8
        
        # Extract features
        # Mel spectrogram (128 x 173 to match original dataset)
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=22050, n_mels=128, n_fft=1024, hop_length=256
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Ensure correct shape (128, 173)
        if mel_spec_db.shape[1] != 173:
            mel_spec_db = librosa.util.fix_length(mel_spec_db, size=173, axis=1)
        
        # MFCC (13 x 173 to match original dataset)
        mfcc = librosa.feature.mfcc(
            y=audio, sr=22050, n_mfcc=13, n_fft=1024, hop_length=256
        )
        
        # Ensure correct shape (13, 173)
        if mfcc.shape[1] != 173:
            mfcc = librosa.util.fix_length(mfcc, size=173, axis=1)
        
        # Create output directories
        for subdir in ['raw', 'mel', 'mfcc', 'wav']:
            (outdir / subdir).mkdir(parents=True, exist_ok=True)
        
        # Save files in PolyMAX format
        # Raw data with params as vector (not dict!)
        raw_path = outdir / 'raw' / f'{sample_id}.npz'
        np.savez_compressed(
            str(raw_path),
            audio=audio,
            params=params_vector.astype(np.float32),  # Vector format!
            metadata=np.array(0.0),  # Scalar metadata
            sr=np.array(22050)
        )
        
        # Feature files
        np.save(str(outdir / 'mel' / f'{sample_id}.npy'), mel_spec_db.astype(np.float32))
        np.save(str(outdir / 'mfcc' / f'{sample_id}.npy'), mfcc.astype(np.float32))
        
        # Audio file
        sf.write(str(outdir / 'wav' / f'{sample_id}.wav'), audio, 22050)
        
        # Save parameter mapping as JSON for reference
        param_json = {
            'sample_id': sample_id,
            'parameter_vector': params_vector.tolist(),
            'parameter_mapping': {name: float(val) for name, val in zip(param_names, params_vector)},
            'note': note,
            'velocity': velocity,
            'duration': duration,
            'strategy': strategy,
            'seed': seed
        }
        with open(str(outdir / 'raw' / f'{sample_id}.json'), 'w') as f:
            json.dump(param_json, f, indent=2)
        
        return True
        
    except Exception as e:
        print(f"Error generating sample {sample_id}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser(description='Fixed isolated PolyMAX augmentation')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--count', type=int, default=100)
    ap.add_argument('--duration', type=float, default=4.0)
    ap.add_argument('--note', type=int, default=-1)
    ap.add_argument('--velocity', type=int, default=100)
    ap.add_argument('--plugin', default='/Library/Audio/Plug-Ins/VST3/uaudio_polymax.vst3')
    ap.add_argument('--strategy', choices=['uniform', 'jitter'], default='jitter')
    ap.add_argument('--mode', choices=['single', 'phrase'], default='single')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--retries', type=int, default=3)
    args = ap.parse_args()
    
    outdir = Path(args.outdir)
    param_names = load_param_schema()
    
    print(f"Generating {args.count} samples in PolyMAX format...")
    print(f"Parameters: {len(param_names)}")
    print(f"Strategy: {args.strategy}")
    print(f"Mode: {args.mode}")
    
    successful = 0
    failed = 0
    
    for i in range(args.count):
        sample_id = f"aug_{i:06d}"
        seed = args.seed + i
        
        # Try with retries
        success = False
        for retry in range(args.retries):
            if generate_sample(outdir, sample_id, param_names, args.duration, 
                             args.note, args.velocity, args.plugin, 
                             args.strategy, seed + retry * 1000):
                success = True
                break
        
        if success:
            successful += 1
        else:
            failed += 1
            print(f"Failed to generate sample {sample_id} after {args.retries} retries")
        
        if (i + 1) % 50 == 0:
            print(f"Progress: {i+1}/{args.count} ({successful} successful, {failed} failed)")
    
    print(f"\nGeneration complete: {successful}/{args.count} samples successful")
    print(f"Output directory: {outdir}")


if __name__ == '__main__':
    main()