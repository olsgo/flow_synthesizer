#!/usr/bin/env python3
"""
Enhanced PolyMAX Dataset Augmentation with Individual Processing

This script follows the DATASET_GENERATION_STANDARD.md methodology:
- Individual processing for each parameter set
- Fresh plugin instance for each sample
- No state interference between samples
- Clean, authentic audio generation
"""

import os
import sys
import json
import argparse
import subprocess
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import tempfile

# Add code directory to path
sys.path.append('code')

@dataclass
class AugmentationConfig:
    """Configuration for dataset augmentation."""
    strategy: str = 'uniform'
    count: int = 100
    output_dir: str = 'augmented_polymax'
    duration: float = 2.0
    sample_rate: int = 44100
    seed: Optional[int] = None
    note: int = 60  # Middle C
    velocity: int = 100
    note_duration: float = 1.5

class ParameterSampler:
    """Intelligent parameter sampling for PolyMAX."""
    
    def __init__(self, param_details: Dict[str, Any], seed: Optional[int] = None):
        self.param_details = param_details
        self.param_names = list(param_details.keys())
        self.rng = np.random.RandomState(seed)
        
        # Categorize parameters by type and musical function
        self._categorize_parameters()
    
    def _categorize_parameters(self):
        """Categorize parameters by musical function."""
        self.categories = {
            'envelope': [p for p in self.param_names if any(x in p.lower() for x in ['env', 'attack', 'decay', 'sustain', 'release'])],
            'oscillator': [p for p in self.param_names if any(x in p.lower() for x in ['osc', 'wave', 'pitch', 'tune'])],
            'filter': [p for p in self.param_names if any(x in p.lower() for x in ['filter', 'cutoff', 'resonance', 'freq'])],
            'lfo': [p for p in self.param_names if 'lfo' in p.lower()],
            'effects': [p for p in self.param_names if any(x in p.lower() for x in ['chorus', 'delay', 'reverb', 'distortion'])],
            'modulation': [p for p in self.param_names if any(x in p.lower() for x in ['mod', 'depth', 'rate'])],
            'global': [p for p in self.param_names if any(x in p.lower() for x in ['master', 'volume', 'polyphony', 'glide'])]
        }
    
    def sample_uniform(self) -> Dict[str, float]:
        """Sample parameters uniformly within their ranges."""
        params = {}
        for name in self.param_names:
            detail = self.param_details[name]
            if detail['min_value'] is not None and detail['max_value'] is not None:
                params[name] = self.rng.uniform(detail['min_value'], detail['max_value'])
            else:
                # For categorical parameters, use current value or random choice
                params[name] = float(detail['value']) if detail['value'].replace('.','').replace('-','').isdigit() else 0.0
        return params
    
    def sample_musical(self) -> Dict[str, float]:
        """Sample parameters with musical constraints."""
        params = self.sample_uniform()
        
        # Apply musical constraints
        # Ensure reasonable envelope times
        for param in self.categories['envelope']:
            if param in params:
                if 'attack' in param.lower():
                    params[param] = self.rng.uniform(0.001, 0.5)  # Quick to medium attack
                elif 'release' in param.lower():
                    params[param] = self.rng.uniform(0.1, 2.0)   # Medium to long release
        
        # Keep filter cutoff in musical range
        for param in self.categories['filter']:
            if 'cutoff' in param.lower() and param in params:
                detail = self.param_details[param]
                if detail['min_value'] is not None and detail['max_value'] is not None:
                    # Bias towards mid-range frequencies
                    mid_point = (detail['min_value'] + detail['max_value']) / 2
                    params[param] = self.rng.normal(mid_point, (detail['max_value'] - detail['min_value']) / 6)
                    params[param] = np.clip(params[param], detail['min_value'], detail['max_value'])
        
        return params
    
    def sample_boundary(self) -> Dict[str, float]:
        """Sample parameters near boundaries for edge case testing."""
        params = {}
        for name in self.param_names:
            detail = self.param_details[name]
            if detail['min_value'] is not None and detail['max_value'] is not None:
                # Choose boundary (min, max, or current)
                boundary_choice = self.rng.choice(['min', 'max', 'current'])
                if boundary_choice == 'min':
                    params[name] = detail['min_value']
                elif boundary_choice == 'max':
                    params[name] = detail['max_value']
                else:
                    params[name] = float(detail['value']) if detail['value'].replace('.','').replace('-','').isdigit() else 0.0
            else:
                    params[name] = float(detail['value']) if detail['value'].replace('.','').replace('-','').isdigit() else 0.0
        return params
    
    def sample_semantic(self) -> Dict[str, float]:
        """Sample parameters by semantic groups."""
        params = self.sample_uniform()
        
        # Randomly choose a category to emphasize
        emphasis_category = self.rng.choice(list(self.categories.keys()))
        
        # For the emphasized category, use more extreme values
        for param in self.categories[emphasis_category]:
            if param in params:
                detail = self.param_details[param]
                if detail['min_value'] is not None and detail['max_value'] is not None:
                    # Use extreme values (near min or max)
                    if self.rng.random() < 0.5:
                        params[param] = detail['min_value'] + 0.1 * (detail['max_value'] - detail['min_value'])
                    else:
                        params[param] = detail['max_value'] - 0.1 * (detail['max_value'] - detail['min_value'])
        
        return params
    
    def sample_progressive(self, progress: float) -> Dict[str, float]:
        """Sample parameters with progressive complexity."""
        params = {}
        for name in self.param_names:
            detail = self.param_details[name]
            if detail['min_value'] is not None and detail['max_value'] is not None:
                # Start conservative, become more extreme with progress
                center = (detail['min_value'] + detail['max_value']) / 2
                range_factor = progress  # 0 to 1
                max_deviation = (detail['max_value'] - detail['min_value']) / 2 * range_factor
                deviation = self.rng.uniform(-max_deviation, max_deviation)
                params[name] = np.clip(center + deviation, detail['min_value'], detail['max_value'])
            else:
                    params[name] = float(detail['value']) if detail['value'].replace('.','').replace('-','').isdigit() else 0.0
        return params
    
    def sample_perceptual(self) -> Dict[str, float]:
        """Sample parameters based on perceptual importance."""
        params = self.sample_uniform()
        
        # Define perceptually important parameters
        important_params = [
            'master_volume', 'filter_cutoff_freq', 'filter_resonance',
            'amp_env_attack', 'amp_env_decay', 'amp_env_sustain', 'amp_env_release'
        ]
        
        # For important parameters, use more varied sampling
        for param in important_params:
            if param in params:
                detail = self.param_details[param]
                if detail['min_value'] is not None and detail['max_value'] is not None:
                    # Use beta distribution for more interesting shapes
                    alpha, beta = 2, 2  # Symmetric around middle
                    normalized = self.rng.beta(alpha, beta)
                    params[param] = detail['min_value'] + normalized * (detail['max_value'] - detail['min_value'])
        
        return params

class SingleSampleProcessor:
    """Process a single sample in isolation."""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
    
    def process_sample(self, params: Dict[str, float], sample_id: str, output_dirs: Dict[str, Path]) -> bool:
        """Process a single sample with given parameters."""
        try:
            # Import here to avoid loading in main process
            sys.path.append('code')
            from synth.synthesize import get_synth
            import librosa
            import soundfile as sf
            
            # Initialize synthesizer with fresh state
            synth = get_synth('polymax')
            
            # Set parameters
            for param_name, value in params.items():
                if hasattr(synth, param_name):
                    setattr(synth, param_name, float(value))
            
            # Generate audio
            audio = synth.render_note(
                note=self.config.note,
                velocity=self.config.velocity,
                duration=self.config.duration,
                sample_rate=self.config.sample_rate
            )
            
            # Ensure audio is the right length and format
            target_length = int(self.config.duration * self.config.sample_rate)
            if len(audio) > target_length:
                audio = audio[:target_length]
            elif len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)))
            
            # Normalize audio
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * 0.8
            
            # Save WAV file
            wav_path = output_dirs['wav'] / f"{sample_id}.wav"
            sf.write(str(wav_path), audio, self.config.sample_rate)
            
            # Compute and save Mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.config.sample_rate,
                n_mels=64,
                hop_length=512,
                n_fft=2048
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            mel_path = output_dirs['mel'] / f"{sample_id}.npy"
            np.save(str(mel_path), mel_spec_db.astype(np.float32))
            
            # Compute and save MFCCs
            mfccs = librosa.feature.mfcc(
                y=audio,
                sr=self.config.sample_rate,
                n_mfcc=13,
                hop_length=512,
                n_fft=2048
            )
            mfcc_path = output_dirs['mfcc'] / f"{sample_id}.npy"
            np.save(str(mfcc_path), mfccs.astype(np.float32))
            
            # Save raw data (flow-synth format)
            raw_path = output_dirs['raw'] / f"{sample_id}.npz"
            chars = np.zeros((10, 3), dtype=np.float32)  # Placeholder
            param_dict = {name: float(params[name]) for name in params.keys()}
            np.savez_compressed(
                str(raw_path),
                param=param_dict,
                chars=chars,
                audio=audio
            )
            
            # Save metadata
            metadata = {
                'sample_id': sample_id,
                'parameters': params,
                'config': {
                    'duration': self.config.duration,
                    'sample_rate': self.config.sample_rate,
                    'note': self.config.note,
                    'velocity': self.config.velocity
                },
                'audio_stats': {
                    'rms': float(np.sqrt(np.mean(audio**2))),
                    'peak': float(np.max(np.abs(audio))),
                    'length': len(audio)
                }
            }
            metadata_path = output_dirs['metadata'] / f"{sample_id}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error processing sample {sample_id}: {e}")
            return False

def process_single_sample_subprocess(params_json: str, sample_id: str, config_json: str, output_dirs_json: str) -> bool:
    """Subprocess entry point for processing a single sample."""
    try:
        # Parse arguments
        params = json.loads(params_json)
        config_dict = json.loads(config_json)
        output_dirs_dict = json.loads(output_dirs_json)
        
        # Convert output_dirs back to Path objects
        output_dirs = {k: Path(v) for k, v in output_dirs_dict.items()}
        
        # Create config object
        config = AugmentationConfig(**config_dict)
        
        # Process sample
        processor = SingleSampleProcessor(config)
        return processor.process_sample(params, sample_id, output_dirs)
        
    except Exception as e:
        print(f"Subprocess error for sample {sample_id}: {e}")
        return False

class EnhancedPolyMAXAugmentation:
    """Enhanced PolyMAX dataset augmentation with individual processing."""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.param_details = self._load_parameter_details()
        self.param_names = self._load_parameter_names()
        self.sampler = ParameterSampler(self.param_details, config.seed)
        
        # Create output directories
        self._create_output_directories()
    
    def _load_parameter_details(self) -> Dict[str, Any]:
        """Load parameter details from JSON file."""
        with open('polymax_param_details.json', 'r') as f:
            data = json.load(f)
        return data['parameter_details']
    
    def _load_parameter_names(self) -> List[str]:
        """Load parameter names from schema."""
        with open('params_schema.json', 'r') as f:
            schema = json.load(f)
        return schema.get('parameter_order', [])
    
    def _create_output_directories(self):
        """Create output directory structure."""
        # Create flow-synthesizer compatible directory structure
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
    
    def generate_samples(self) -> Dict[str, Any]:
        """Generate augmented samples using individual processing."""
        print(f"Generating {self.config.count} samples using {self.config.strategy} strategy")
        print("Following individual processing standard for clean audio generation")
        
        successful_samples = 0
        failed_samples = 0
        
        # Generate parameter sets
        param_sets = self._generate_parameter_sets()
        
        print(f"\nProcessing {len(param_sets)} samples individually...")
        
        # Process each sample in a separate subprocess
        for i, params in enumerate(param_sets):
            sample_id = f"{self.config.strategy}_{i:06d}"
            
            # Convert config and output_dirs to JSON for subprocess
            config_dict = {
                'strategy': self.config.strategy,
                'count': self.config.count,
                'output_dir': self.config.output_dir,
                'duration': self.config.duration,
                'sample_rate': self.config.sample_rate,
                'seed': self.config.seed,
                'note': self.config.note,
                'velocity': self.config.velocity,
                'note_duration': self.config.note_duration
            }
            
            output_dirs_dict = {k: str(v) for k, v in self.output_dirs.items()}
            
            # Launch subprocess for this sample
            try:
                result = subprocess.run([
                    sys.executable, '-c',
                    f"""
import sys
sys.path.append('.')
from enhanced_polymax_augmentation_fixed import process_single_sample_subprocess
import json

params_json = '''{json.dumps(params)}'''
sample_id = '{sample_id}'
config_json = '''{json.dumps(config_dict)}'''
output_dirs_json = '''{json.dumps(output_dirs_dict)}'''

success = process_single_sample_subprocess(params_json, sample_id, config_json, output_dirs_json)
exit(0 if success else 1)
"""
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    successful_samples += 1
                    if (i + 1) % 10 == 0:
                        print(f"Processed {i + 1}/{len(param_sets)} samples")
                else:
                    failed_samples += 1
                    print(f"Failed to process sample {sample_id}: {result.stderr}")
                
                # Small delay to ensure clean state
                time.sleep(0.1)
                
            except subprocess.TimeoutExpired:
                failed_samples += 1
                print(f"Timeout processing sample {sample_id}")
            except Exception as e:
                failed_samples += 1
                print(f"Error processing sample {sample_id}: {e}")
        
        print(f"\nAugmentation complete: {successful_samples}/{self.config.count} samples generated")
        if failed_samples > 0:
            print(f"Failed samples: {failed_samples}")
        
        # Save generation summary
        summary = {
            'config': config_dict,
            'results': {
                'successful_samples': successful_samples,
                'failed_samples': failed_samples,
                'total_requested': self.config.count
            },
            'parameter_count': len(self.param_names),
            'strategy': self.config.strategy
        }
        
        summary_path = Path(self.config.output_dir) / 'generation_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Output directory: {self.config.output_dir}")
        return summary
    
    def _generate_parameter_sets(self) -> List[Dict[str, float]]:
        """Generate parameter sets based on strategy."""
        param_sets = []
        
        if self.config.strategy == 'uniform':
            for _ in range(self.config.count):
                param_sets.append(self.sampler.sample_uniform())
        
        elif self.config.strategy == 'musical':
            for _ in range(self.config.count):
                param_sets.append(self.sampler.sample_musical())
        
        elif self.config.strategy == 'boundary':
            for _ in range(self.config.count):
                param_sets.append(self.sampler.sample_boundary())
        
        elif self.config.strategy == 'semantic':
            for _ in range(self.config.count):
                param_sets.append(self.sampler.sample_semantic())
        
        elif self.config.strategy == 'progressive':
            for i in range(self.config.count):
                progress = i / (self.config.count - 1) if self.config.count > 1 else 0
                param_sets.append(self.sampler.sample_progressive(progress))
        
        elif self.config.strategy == 'perceptual':
            for _ in range(self.config.count):
                param_sets.append(self.sampler.sample_perceptual())
        
        elif self.config.strategy == 'all':
            # Mix of all strategies
            strategies = ['uniform', 'musical', 'boundary', 'semantic', 'progressive', 'perceptual']
            samples_per_strategy = self.config.count // len(strategies)
            remainder = self.config.count % len(strategies)
            
            for i, strategy in enumerate(strategies):
                count = samples_per_strategy + (1 if i < remainder else 0)
                
                for j in range(count):
                    if strategy == 'uniform':
                        param_sets.append(self.sampler.sample_uniform())
                    elif strategy == 'musical':
                        param_sets.append(self.sampler.sample_musical())
                    elif strategy == 'boundary':
                        param_sets.append(self.sampler.sample_boundary())
                    elif strategy == 'semantic':
                        param_sets.append(self.sampler.sample_semantic())
                    elif strategy == 'progressive':
                        progress = j / (count - 1) if count > 1 else 0
                        param_sets.append(self.sampler.sample_progressive(progress))
                    elif strategy == 'perceptual':
                        param_sets.append(self.sampler.sample_perceptual())
        
        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")
        
        return param_sets

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Enhanced PolyMAX Dataset Augmentation with Individual Processing')
    parser.add_argument('--strategy', type=str, default='uniform',
                       choices=['uniform', 'musical', 'boundary', 'semantic', 'progressive', 'perceptual', 'all'],
                       help='Augmentation strategy')
    parser.add_argument('--count', type=int, default=100, help='Number of samples to generate')
    parser.add_argument('--outdir', type=str, default='augmented_polymax', help='Output directory')
    parser.add_argument('--duration', type=float, default=2.0, help='Audio duration in seconds')
    parser.add_argument('--sample-rate', type=int, default=44100, help='Sample rate')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--note', type=int, default=60, help='MIDI note number')
    parser.add_argument('--velocity', type=int, default=100, help='MIDI velocity')
    
    args = parser.parse_args()
    
    # Create configuration
    config = AugmentationConfig(
        strategy=args.strategy,
        count=args.count,
        output_dir=args.outdir,
        duration=args.duration,
        sample_rate=args.sample_rate,
        seed=args.seed,
        note=args.note,
        velocity=args.velocity
    )
    
    # Run augmentation
    augmenter = EnhancedPolyMAXAugmentation(config)
    summary = augmenter.generate_samples()
    
    print("\nEnhanced dataset augmentation completed successfully!")
    print("Generated samples follow individual processing standard for clean audio.")
    print("Samples are ready for flow synthesizer training.")

if __name__ == '__main__':
    main()