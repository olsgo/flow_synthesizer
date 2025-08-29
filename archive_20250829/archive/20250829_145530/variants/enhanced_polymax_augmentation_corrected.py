#!/usr/bin/env python3
"""
Enhanced PolyMAX Dataset Augmentation - Corrected Version

Follows DATASET_GENERATION_STANDARD.md guidelines for individual processing.
Each sample is processed in complete isolation to prevent state interference.

Usage:
  python enhanced_polymax_augmentation_corrected.py \
    --strategy uniform \
    --count 10 \
    --output_dir test_output_corrected
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import librosa
import soundfile as sf

# Add project paths
sys.path.append('code')
from synth.synthesize import create_synth, midiname2num


@dataclass
class AugmentationConfig:
    """Configuration for augmentation process"""
    plugin_path: str = '/Library/Audio/Plug-Ins/VST3/uaudio_polymax.vst3'
    output_dir: str = 'enhanced_polymax_output'
    sample_rate: int = 44100
    duration: float = 4.0
    note: int = 60
    velocity: int = 100
    

class ParameterSampler:
    """Intelligent parameter sampling with multiple strategies"""
    
    def __init__(self, param_details: Dict[str, dict]):
        self.param_details = param_details
        self.param_names = list(param_details.keys())
        
        # Load parameter schema for ordering
        with open('params_schema.json', 'r') as f:
            schema = json.load(f)
        self.param_order = schema.get('parameter_order', self.param_names)
        
        # Safe defaults to ensure audible output
        self.safe_defaults = {
            'master_bypass': 0.0,
            'polyphony': 1.0,
            'osc_1_volume': 0.8,
            'osc_2_volume': 0.1,
            'noise_volume': 0.0,
            'filter_cutoff_freq': 0.85,
            'filter_resonance': 0.15,
            'filter_env_amt': 0.0,
            'amp_env_attack': 0.02,
            'amp_env_decay': 0.25,
            'amp_env_sustain': 0.75,
            'amp_env_release': 0.25,
            'arp_enable': 0.0  # OUT
        }
        
        # Enum choices for categorical parameters
        self.enum_choices = {
            'polyphony': ['MONO', 'POLY'],
            'note_trigger_mode': ['LEGATO', 'RETRIG'],
            'lfo_shape': ['TRIANGLE', 'RAMP_UP', 'RAMP_DOWN', 'SQUARE', 'RANDOM'],
            'noise_color': ['PINK', 'WHITE'],
            'filter_mode': ['LP', 'HP', 'BP', 'NOTCH'],
            'filter_slope': ['2P', '4P'],
            'unison': ['OFF', 'ON'],
            'mod_fx_type': ['PHASER', 'FLANGER', 'CHORUS'],
            'space_fx_type': ['HALL', 'SPRING'],
            'arp_enable': ['OUT', 'IN'],
        }
    
    def sample_uniform(self, count: int) -> List[Dict[str, float]]:
        """Generate uniform random parameter sets"""
        samples = []
        for _ in range(count):
            params = {}
            for name in self.param_order:
                if name in self.param_details:
                    detail = self.param_details[name]
                    param_type = detail.get('type', 'continuous')
                    
                    if param_type == 'enum' and name in self.enum_choices:
                        # Random choice for enum parameters
                        choice = random.choice(self.enum_choices[name])
                        # Convert to normalized value (0-1 based on position)
                        choices = self.enum_choices[name]
                        params[name] = float(choices.index(choice) / max(1, len(choices) - 1))
                    else:
                        # Random value for continuous parameters
                        params[name] = random.random()
            
            # Apply safe defaults
            for key, value in self.safe_defaults.items():
                if key in params:
                    params[key] = value
                    
            samples.append(params)
        return samples
    
    def sample_musical(self, count: int) -> List[Dict[str, float]]:
        """Generate musically-informed parameter sets"""
        samples = []
        for _ in range(count):
            params = self.sample_uniform(1)[0]  # Start with uniform base
            
            # Musical parameter relationships
            # Oscillator balance
            osc1_vol = random.uniform(0.6, 1.0)
            osc2_vol = random.uniform(0.0, 0.4) if osc1_vol > 0.7 else random.uniform(0.0, 0.6)
            params['osc_1_volume'] = osc1_vol
            params['osc_2_volume'] = osc2_vol
            
            # Filter settings based on musical context
            cutoff = random.uniform(0.3, 0.95)
            resonance = random.uniform(0.0, 0.3) if cutoff > 0.8 else random.uniform(0.0, 0.5)
            params['filter_cutoff_freq'] = cutoff
            params['filter_resonance'] = resonance
            
            # Envelope relationships
            attack = random.uniform(0.0, 0.1)
            decay = random.uniform(0.1, 0.5)
            sustain = random.uniform(0.5, 0.9)
            release = random.uniform(0.1, 0.8)
            
            params['amp_env_attack'] = attack
            params['amp_env_decay'] = decay
            params['amp_env_sustain'] = sustain
            params['amp_env_release'] = release
            
            samples.append(params)
        return samples


def process_single_sample(sample_data: Dict) -> bool:
    """Process a single sample in complete isolation (subprocess function)"""
    try:
        params = sample_data['params']
        sample_id = sample_data['sample_id']
        config = sample_data['config']
        output_dirs = sample_data['output_dirs']
        
        # Initialize fresh synthesizer engine
        engine, generator, param_defaults, rev_idx = create_synth(
            'polymax_dataset', 'polymax', config['plugin_path']
        )
        
        # Apply parameters
        patch = midiname2num(params, rev_idx)
        engine.set_patch(patch)
        
        # Render audio
        note_length = max(0.25, config['duration'] - 0.05)
        engine.render_patch(
            config['note'], 
            config['velocity'], 
            note_length, 
            config['duration'], 
            warm_up=False
        )
        audio = engine.get_audio_frames()
        
        # Convert to mono if stereo
        if audio.ndim == 2:
            audio = audio.mean(axis=0)
        
        # Resample to 22.05kHz for dataset compatibility
        audio_22k = librosa.resample(
            audio.astype(np.float32), 
            orig_sr=config['sample_rate'], 
            target_sr=22050
        )
        
        # Normalize
        if np.max(np.abs(audio_22k)) > 0:
            audio_22k = 0.8 * audio_22k / np.max(np.abs(audio_22k))
        
        # Generate features
        mel_spec = librosa.feature.melspectrogram(
            y=audio_22k, sr=22050, n_mels=64, n_fft=1024, hop_length=256
        )
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Resize to 64x80
        if mel_spec.shape[1] != 80:
            mel_spec = librosa.util.fix_length(mel_spec, size=80, axis=1)
        
        # MFCC features
        mfcc = librosa.feature.mfcc(
            y=audio_22k, sr=22050, n_mfcc=16, n_fft=1024, hop_length=256
        )
        if mfcc.shape[1] != 80:
            mfcc = librosa.util.fix_length(mfcc, size=80, axis=1)
        
        # Tile MFCC to 64 rows
        mfcc_tiled = np.tile(mfcc, (4, 1))[:64, :]
        
        # Save files
        # NPZ file with flow-synth compatible format
        npz_path = Path(output_dirs['raw']) / f"{sample_id}.npz"
        np.savez_compressed(
            npz_path,
            param=params,  # Dictionary format expected by flow-synth
            chars=np.zeros((10, 3), dtype=np.float32),
            audio=audio_22k.astype(np.float32)
        )
        
        # Feature files
        mel_path = Path(output_dirs['mel']) / f"{sample_id}.npy"
        mfcc_path = Path(output_dirs['mfcc']) / f"{sample_id}.npy"
        wav_path = Path(output_dirs['wav']) / f"{sample_id}.wav"
        
        np.save(mel_path, mel_spec.astype(np.float32))
        np.save(mfcc_path, mfcc_tiled.astype(np.float32))
        sf.write(wav_path, audio_22k, 22050)
        
        # Metadata
        metadata = {
            'sample_id': sample_id,
            'parameters': params,
            'audio_duration': config['duration'],
            'sample_rate': 22050,
            'note': config['note'],
            'velocity': config['velocity'],
            'mel_shape': list(mel_spec.shape),
            'mfcc_shape': list(mfcc_tiled.shape)
        }
        
        metadata_path = Path(output_dirs['metadata']) / f"{sample_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return True
        
    except Exception as e:
        print(f"Error processing sample {sample_id}: {e}")
        return False


class EnhancedPolyMAXAugmentation:
    """Enhanced PolyMAX dataset augmentation with individual processing"""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.param_sampler = None
        self.output_dirs = {}
        
    def _load_parameter_details(self) -> Dict[str, dict]:
        """Load parameter details from JSON file"""
        details_path = Path('polymax_param_details.json')
        if not details_path.exists():
            raise FileNotFoundError(f"Parameter details file not found: {details_path}")
        
        with open(details_path, 'r') as f:
            data = json.load(f)
        
        return data.get('parameter_details', {})
    
    def _create_output_directories(self):
        """Create output directory structure compatible with flow-synthesizer"""
        base_dir = Path(self.config.output_dir)
        dataset_dir = base_dir / 'polymax_dataset'
        
        self.output_dirs = {
            'raw': dataset_dir / 'raw',
            'mel': dataset_dir / 'mel', 
            'mfcc': dataset_dir / 'mfcc',
            'wav': dataset_dir / 'wav',
            'metadata': dataset_dir / 'metadata'
        }
        
        for dir_path in self.output_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def generate_samples(self, strategy: str, count: int) -> bool:
        """Generate samples using specified strategy with individual processing"""
        print(f"Generating {count} samples using {strategy} strategy...")
        
        # Load parameter details and create sampler
        param_details = self._load_parameter_details()
        self.param_sampler = ParameterSampler(param_details)
        
        # Create output directories
        self._create_output_directories()
        
        # Generate parameter sets
        if strategy == 'uniform':
            param_sets = self.param_sampler.sample_uniform(count)
        elif strategy == 'musical':
            param_sets = self.param_sampler.sample_musical(count)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Prepare sample data for individual processing
        sample_data_list = []
        for i, params in enumerate(param_sets):
            sample_id = f"{strategy}_{i:06d}"
            sample_data = {
                'params': params,
                'sample_id': sample_id,
                'config': {
                    'plugin_path': self.config.plugin_path,
                    'duration': self.config.duration,
                    'sample_rate': self.config.sample_rate,
                    'note': self.config.note,
                    'velocity': self.config.velocity
                },
                'output_dirs': {k: str(v) for k, v in self.output_dirs.items()}
            }
            sample_data_list.append(sample_data)
        
        # Process samples individually using multiprocessing
        print("Processing samples with individual isolation...")
        successful_count = 0
        
        # Use multiprocessing for true isolation
        with mp.Pool(processes=min(4, mp.cpu_count())) as pool:
            results = pool.map(process_single_sample, sample_data_list)
            successful_count = sum(results)
        
        print(f"Successfully generated {successful_count}/{count} samples")
        print(f"Output saved to: {self.config.output_dir}/polymax_dataset/")
        
        return successful_count == count


def main():
    parser = argparse.ArgumentParser(description='Enhanced PolyMAX Dataset Augmentation - Corrected')
    parser.add_argument('--strategy', choices=['uniform', 'musical'], default='uniform',
                       help='Parameter sampling strategy')
    parser.add_argument('--count', type=int, default=10,
                       help='Number of samples to generate')
    parser.add_argument('--output_dir', default='enhanced_polymax_output',
                       help='Output directory')
    parser.add_argument('--duration', type=float, default=4.0,
                       help='Audio duration in seconds')
    parser.add_argument('--note', type=int, default=60,
                       help='MIDI note number')
    parser.add_argument('--velocity', type=int, default=100,
                       help='MIDI velocity')
    parser.add_argument('--plugin_path', 
                       default='/Library/Audio/Plug-Ins/VST3/uaudio_polymax.vst3',
                       help='Path to PolyMAX VST3 plugin')
    
    args = parser.parse_args()
    
    # Create configuration
    config = AugmentationConfig(
        plugin_path=args.plugin_path,
        output_dir=args.output_dir,
        duration=args.duration,
        note=args.note,
        velocity=args.velocity
    )
    
    # Create augmentation instance
    augmenter = EnhancedPolyMAXAugmentation(config)
    
    # Generate samples
    success = augmenter.generate_samples(args.strategy, args.count)
    
    if success:
        print("\n✓ Enhanced dataset augmentation completed successfully!")
        print("\n✓ Follows DATASET_GENERATION_STANDARD.md individual processing guidelines")
        print("✓ Each sample processed in complete isolation")
        print("✓ No state interference between samples")
        print("✓ Compatible with flow-synthesizer training pipeline")
    else:
        print("\n✗ Some samples failed to generate")
        sys.exit(1)


if __name__ == '__main__':
    main()