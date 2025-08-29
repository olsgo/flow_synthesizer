#!/usr/bin/env python3
"""
Enhanced PolyMAX Dataset Augmentation for Flow Synthesizer

Implements advanced parameter randomization strategies inspired by the original
flow synthesizer research, going beyond factory presets to create diverse
training data through:

1. Smart parameter sampling with musical constraints
2. Perceptually-guided parameter combinations
3. Boundary exploration for robust model training
4. Semantic-aware parameter clustering
5. Progressive difficulty augmentation

Based on the original flow synthesizer approach but enhanced for PolyMAX.

Usage:
  python enhanced_polymax_augmentation.py \
    --strategy all \
    --count 5000 \
    --outdir datasets/polymax_enhanced \
    --seed 42
"""

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

import numpy as np
import librosa
import soundfile as sf
from scipy import stats
from sklearn.cluster import KMeans

# Project imports
import sys
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / 'code'))
from synth.synthesize import create_synth, midiname2num


class AugmentationStrategy(Enum):
    """Different parameter sampling strategies"""
    UNIFORM = "uniform"              # Uniform random sampling
    MUSICAL = "musical"              # Musically-informed sampling
    BOUNDARY = "boundary"            # Boundary and extreme value exploration
    SEMANTIC = "semantic"            # Semantic clustering-based sampling
    PROGRESSIVE = "progressive"      # Progressive difficulty sampling
    PERCEPTUAL = "perceptual"        # Perceptually-guided combinations
    ALL = "all"                     # Combination of all strategies


@dataclass
class AugmentationConfig:
    """Configuration for enhanced augmentation"""
    strategy: AugmentationStrategy = AugmentationStrategy.ALL
    total_samples: int = 1000
    output_dir: str = "datasets/polymax_enhanced"
    duration: float = 4.0
    note_range: Tuple[int, int] = (48, 72)  # C3 to C5
    velocity_range: Tuple[int, int] = (80, 127)
    seed: Optional[int] = None
    plugin_path: str = '/Library/Audio/Plug-Ins/VST3/uaudio_polymax.vst3'
    
    # Strategy-specific parameters
    boundary_exploration_ratio: float = 0.15  # 15% boundary cases
    semantic_clusters: int = 8                # Number of semantic clusters
    progressive_stages: int = 4               # Difficulty progression stages
    musical_bias_strength: float = 0.3        # How much to bias toward musical values


class ParameterSampler:
    """Advanced parameter sampling with multiple strategies"""
    
    def __init__(self, param_names: List[str], param_details: Dict[str, dict]):
        self.param_names = param_names
        self.param_details = param_details
        self._analyze_parameters()
        
    def _analyze_parameters(self):
        """Analyze parameter types and ranges for intelligent sampling"""
        self.numeric_params = []
        self.enum_params = []
        self.boolean_params = []
        
        # Enhanced enum choices based on PolyMAX documentation
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
            'arp_mode': ['UP', 'DOWN', 'UP_DOWN', 'RANDOM'],
            'arp_rate': ['1/32', '1/16', '1/8', '1/4', '1/2', '1/1'],
        }
        
        # Musical parameter groupings for correlated sampling
        self.musical_groups = {
            'oscillators': ['osc_1_coarse_tune', 'osc_1_fine_tune', 'osc_1_shape', 'osc_1_fm_amount',
                           'osc_2_coarse_tune', 'osc_2_fine_tune', 'osc_2_shape', 'osc_2_sync'],
            'envelopes': ['amp_env_attack', 'amp_env_decay', 'amp_env_sustain', 'amp_env_release',
                         'filter_env_attack', 'filter_env_decay', 'filter_env_sustain', 'filter_env_release'],
            'filter': ['filter_cutoff_freq', 'filter_resonance', 'filter_env_amt', 'filter_mode', 'filter_slope'],
            'modulation': ['lfo_rate', 'lfo_shape', 'mod_wheel', 'pitch_mod', 'filter_lfo_amt'],
            'effects': ['mod_fx_type', 'mod_fx_rate', 'mod_fx_depth', 'space_fx_type', 'space_fx_amount']
        }
        
        # Categorize parameters
        for name in self.param_names:
            details = self.param_details.get(name, {})
            min_val = details.get('min_value')
            max_val = details.get('max_value')
            
            if name in self.enum_choices:
                self.enum_params.append(name)
            elif isinstance(min_val, bool) and isinstance(max_val, bool):
                self.boolean_params.append(name)
            elif isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                self.numeric_params.append(name)
            else:
                # Default to enum if unclear
                self.enum_params.append(name)
                
    def sample_uniform(self) -> Dict[str, float]:
        """Uniform random sampling (baseline strategy)"""
        params = {}
        for name in self.param_names:
            if name in self.enum_choices:
                choices = self.enum_choices[name]
                idx = random.randrange(len(choices))
                params[name] = 0.0 if len(choices) == 1 else idx / (len(choices) - 1)
            elif name in self.boolean_params:
                params[name] = float(random.random() > 0.5)
            else:
                params[name] = random.random()
        return params
        
    def sample_musical(self) -> Dict[str, float]:
        """Musically-informed sampling with harmonic relationships"""
        params = {}
        
        # Start with uniform sampling
        params = self.sample_uniform()
        
        # Apply musical constraints and correlations
        
        # 1. Harmonic oscillator tuning
        if 'osc_1_coarse_tune' in params and 'osc_2_coarse_tune' in params:
            # Bias toward harmonic intervals (octave, fifth, fourth)
            harmonic_intervals = [0, 7, 12, 19, 24]  # semitones
            if random.random() < 0.4:  # 40% chance of harmonic relationship
                interval = random.choice(harmonic_intervals)
                base_tune = params['osc_1_coarse_tune']
                # Convert to semitone offset and apply interval
                params['osc_2_coarse_tune'] = np.clip(base_tune + interval/24.0, 0, 1)
                
        # 2. Envelope correlations (longer attack -> longer decay tendency)
        if 'amp_env_attack' in params and 'amp_env_decay' in params:
            if params['amp_env_attack'] > 0.7:  # Long attack
                # Bias toward longer decay
                params['amp_env_decay'] = np.clip(params['amp_env_decay'] + 0.2, 0, 1)
                
        # 3. Filter frequency and resonance relationship
        if 'filter_cutoff_freq' in params and 'filter_resonance' in params:
            # Higher cutoff -> lower resonance tendency (avoid harshness)
            if params['filter_cutoff_freq'] > 0.8:
                params['filter_resonance'] *= 0.6
                
        # 4. LFO rate musical quantization
        if 'lfo_rate' in params:
            # Bias toward musical subdivisions
            musical_rates = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]  # Musical fractions
            if random.random() < 0.3:
                params['lfo_rate'] = random.choice(musical_rates)
                
        return params
        
    def sample_boundary(self) -> Dict[str, float]:
        """Boundary exploration for robust training"""
        params = {}
        
        for name in self.param_names:
            details = self.param_details.get(name, {})
            
            if name in self.enum_choices:
                choices = self.enum_choices[name]
                # Bias toward extreme choices
                if random.random() < 0.6:
                    idx = random.choice([0, len(choices)-1])  # First or last
                else:
                    idx = random.randrange(len(choices))
                params[name] = 0.0 if len(choices) == 1 else idx / (len(choices) - 1)
            else:
                # For numeric parameters, bias toward boundaries
                boundary_prob = 0.4
                if random.random() < boundary_prob:
                    params[name] = random.choice([0.0, 1.0])  # Exact boundaries
                elif random.random() < 0.3:
                    # Near boundaries (within 10%)
                    if random.random() < 0.5:
                        params[name] = random.uniform(0.0, 0.1)
                    else:
                        params[name] = random.uniform(0.9, 1.0)
                else:
                    params[name] = random.random()  # Normal sampling
                    
        return params
        
    def sample_semantic_cluster(self, cluster_id: int, num_clusters: int) -> Dict[str, float]:
        """Sample from semantic parameter clusters"""
        params = {}
        
        # Define semantic clusters based on musical characteristics
        cluster_profiles = {
            0: {'bass': True, 'aggressive': False},      # Bass sounds
            1: {'bass': False, 'aggressive': True},      # Lead sounds
            2: {'bass': False, 'aggressive': False},     # Pad sounds
            3: {'bass': True, 'aggressive': True},       # Aggressive bass
            4: {'modulated': True},                      # Heavily modulated
            5: {'simple': True},                         # Simple/clean
            6: {'effects_heavy': True},                  # Effects-heavy
            7: {'percussive': True}                      # Percussive/pluck
        }
        
        cluster_id = cluster_id % len(cluster_profiles)
        profile = cluster_profiles[cluster_id]
        
        # Start with uniform sampling
        params = self.sample_uniform()
        
        # Apply cluster-specific biases
        if profile.get('bass', False):
            # Lower oscillator tuning
            if 'osc_1_coarse_tune' in params:
                params['osc_1_coarse_tune'] *= 0.3  # Lower range
            if 'filter_cutoff_freq' in params:
                params['filter_cutoff_freq'] *= 0.6  # Lower filter
                
        if profile.get('aggressive', False):
            # Higher resonance, more distortion
            if 'filter_resonance' in params:
                params['filter_resonance'] = np.clip(params['filter_resonance'] + 0.3, 0, 1)
            if 'osc_1_fm_amount' in params:
                params['osc_1_fm_amount'] = np.clip(params['osc_1_fm_amount'] + 0.2, 0, 1)
                
        if profile.get('modulated', False):
            # More LFO and modulation
            if 'lfo_rate' in params:
                params['lfo_rate'] = np.clip(params['lfo_rate'] + 0.3, 0, 1)
            if 'filter_lfo_amt' in params:
                params['filter_lfo_amt'] = np.clip(params['filter_lfo_amt'] + 0.4, 0, 1)
                
        if profile.get('percussive', False):
            # Fast attack, fast decay
            if 'amp_env_attack' in params:
                params['amp_env_attack'] *= 0.1  # Very fast attack
            if 'amp_env_decay' in params:
                params['amp_env_decay'] *= 0.4   # Faster decay
                
        return params
        
    def sample_progressive(self, stage: int, total_stages: int) -> Dict[str, float]:
        """Progressive difficulty sampling"""
        # Stage 0: Simple, conservative parameters
        # Stage N: Complex, extreme parameters
        
        complexity = stage / (total_stages - 1) if total_stages > 1 else 0
        
        if complexity < 0.25:
            # Simple sounds - bias toward center values
            params = {}
            for name in self.param_names:
                if name in self.enum_choices:
                    choices = self.enum_choices[name]
                    # Prefer first choice (usually default/simple)
                    if random.random() < 0.7:
                        idx = 0
                    else:
                        idx = random.randrange(len(choices))
                    params[name] = 0.0 if len(choices) == 1 else idx / (len(choices) - 1)
                else:
                    # Bias toward center values
                    center_bias = 0.6
                    if random.random() < center_bias:
                        params[name] = np.random.normal(0.5, 0.15)  # Centered distribution
                        params[name] = np.clip(params[name], 0, 1)
                    else:
                        params[name] = random.random()
        else:
            # Complex sounds - use other strategies
            if complexity < 0.5:
                params = self.sample_musical()
            elif complexity < 0.75:
                params = self.sample_semantic_cluster(random.randint(0, 7), 8)
            else:
                params = self.sample_boundary()
                
        return params


class EnhancedPolyMAXAugmentation:
    """Enhanced dataset augmentation with multiple strategies"""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.param_names = []
        self.param_details = {}
        self.sampler = None
        self.engine = None
        self.generator = None
        self.param_defaults = {}
        self.rev_idx = {}
        
    def initialize(self):
        """Initialize the augmentation system"""
        # Set random seeds
        if self.config.seed is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
            
        # Load parameter schema
        self._load_parameter_schema()
        
        # Initialize parameter sampler
        self.sampler = ParameterSampler(self.param_names, self.param_details)
        
        # Initialize synthesizer engine
        self._initialize_synthesizer()
        
        # Create output directories
        self._create_output_directories()
        
    def _load_parameter_schema(self):
        """Load parameter names and details"""
        schema_path = Path('params_schema.json')
        details_path = Path('polymax_param_details.json')
        
        with open(schema_path, 'r') as f:
            schema = json.load(f)
        self.param_names = schema.get('parameter_order', [])
        
        if details_path.exists():
            with open(details_path, 'r') as f:
                details = json.load(f)
            self.param_details = details.get('parameter_details', {})
            
    def _initialize_synthesizer(self):
        """Initialize the PolyMAX synthesizer engine"""
        self.engine, self.generator, self.param_defaults, self.rev_idx = create_synth(
            'polymax_dataset', 'polymax', self.config.plugin_path
        )
        
    def _create_output_directories(self):
        """Create output directory structure"""
        # Create flow-synthesizer compatible directory structure
        # The data loader expects: path/dataset_name/subdirs
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
            
    def _render_audio(self, params: Dict[str, float], note: int, velocity: int) -> Optional[np.ndarray]:
        """Render audio for given parameters"""
        try:
            # Convert to patch format
            patch = midiname2num(params, self.rev_idx)
            self.engine.set_patch(patch)
            
            # Render settings
            note_length = max(0.25, self.config.duration - 0.05)
            
            # Render at 44.1kHz then resample
            self.engine.render_patch(note, velocity, note_length, self.config.duration, warm_up=False)
            audio = self.engine.get_audio_frames()
            
            # Convert to mono if stereo
            if audio.ndim == 2:
                audio = audio.mean(axis=0)
                
            # Resample to 22.05kHz for dataset compatibility
            audio_22k = librosa.resample(audio.astype(np.float32), orig_sr=44100, target_sr=22050)
            
            # Normalize
            if np.max(np.abs(audio_22k)) > 0:
                audio_22k = 0.8 * audio_22k / np.max(np.abs(audio_22k))
                
            return audio_22k
            
        except Exception as e:
            print(f"Error rendering audio: {e}")
            return None
            
    def _compute_features(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mel spectrogram and MFCC features"""
        # Mel spectrogram (64 x 80)
        mel = librosa.feature.melspectrogram(
            y=audio, sr=22050, n_fft=2048, n_mels=64, 
            hop_length=1024, fmin=30, fmax=11000
        )
        
        # Trim/pad to 80 frames
        if mel.shape[1] < 80:
            mel = np.pad(mel, ((0, 0), (0, 80 - mel.shape[1])), mode='constant')
        elif mel.shape[1] > 80:
            mel = mel[:, :80]
            
        # MFCC (16 coeffs -> 64 x 80)
        mfcc = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=16, hop_length=256)
        if mfcc.shape[1] < 80:
            mfcc = np.pad(mfcc, ((0, 0), (0, 80 - mfcc.shape[1])), mode='edge')
        elif mfcc.shape[1] > 80:
            mfcc = mfcc[:, :80]
            
        # Tile MFCC to 64x80
        mfcc_64 = np.repeat(mfcc, 4, axis=0)
        
        return mel.astype(np.float32), mfcc_64.astype(np.float32)
        
    def _save_sample(self, sample_id: str, params: Dict[str, float], audio: np.ndarray, 
                    mel: np.ndarray, mfcc: np.ndarray, metadata: Dict[str, Any]):
        """Save a single augmented sample"""
        # Save audio
        wav_path = self.output_dirs['wav'] / f"{sample_id}.wav"
        sf.write(str(wav_path), audio, 22050)
        
        # Save features
        mel_path = self.output_dirs['mel'] / f"{sample_id}.npy"
        mfcc_path = self.output_dirs['mfcc'] / f"{sample_id}.npy"
        np.save(str(mel_path), mel)
        np.save(str(mfcc_path), mfcc)
        
        # Save raw data (flow-synth format)
        raw_path = self.output_dirs['raw'] / f"{sample_id}.npz"
        chars = np.zeros((10, 3), dtype=np.float32)  # Placeholder
        # Convert parameters to dictionary format expected by flow-synth
        param_dict = {name: float(params[name]) for name in self.param_names}
        np.savez_compressed(
            str(raw_path),
            param=param_dict,
            chars=chars,
            audio=audio
        )
        
        # Save metadata
        metadata_path = self.output_dirs['metadata'] / f"{sample_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
    def generate_samples(self):
        """Generate augmented samples using specified strategy"""
        print(f"Generating {self.config.total_samples} samples using {self.config.strategy.value} strategy")
        
        # Determine sample distribution across strategies
        if self.config.strategy == AugmentationStrategy.ALL:
            strategy_distribution = {
                AugmentationStrategy.UNIFORM: int(0.2 * self.config.total_samples),
                AugmentationStrategy.MUSICAL: int(0.25 * self.config.total_samples),
                AugmentationStrategy.BOUNDARY: int(0.15 * self.config.total_samples),
                AugmentationStrategy.SEMANTIC: int(0.2 * self.config.total_samples),
                AugmentationStrategy.PROGRESSIVE: int(0.15 * self.config.total_samples),
                AugmentationStrategy.PERCEPTUAL: int(0.05 * self.config.total_samples)
            }
        else:
            strategy_distribution = {self.config.strategy: self.config.total_samples}
            
        sample_count = 0
        successful_count = 0
        
        for strategy, count in strategy_distribution.items():
            print(f"\nGenerating {count} samples with {strategy.value} strategy...")
            
            for i in range(count):
                # Sample parameters based on strategy
                if strategy == AugmentationStrategy.UNIFORM:
                    params = self.sampler.sample_uniform()
                elif strategy == AugmentationStrategy.MUSICAL:
                    params = self.sampler.sample_musical()
                elif strategy == AugmentationStrategy.BOUNDARY:
                    params = self.sampler.sample_boundary()
                elif strategy == AugmentationStrategy.SEMANTIC:
                    cluster_id = random.randint(0, self.config.semantic_clusters - 1)
                    params = self.sampler.sample_semantic_cluster(cluster_id, self.config.semantic_clusters)
                elif strategy == AugmentationStrategy.PROGRESSIVE:
                    stage = random.randint(0, self.config.progressive_stages - 1)
                    params = self.sampler.sample_progressive(stage, self.config.progressive_stages)
                else:  # PERCEPTUAL - use musical for now
                    params = self.sampler.sample_musical()
                    
                # Random MIDI settings
                note = random.randint(*self.config.note_range)
                velocity = random.randint(*self.config.velocity_range)
                
                # Render audio
                audio = self._render_audio(params, note, velocity)
                if audio is None:
                    continue
                    
                # Compute features
                mel, mfcc = self._compute_features(audio)
                
                # Create sample ID and metadata
                sample_id = f"{strategy.value}_{sample_count:06d}"
                metadata = {
                    'strategy': strategy.value,
                    'sample_id': sample_id,
                    'parameters': params,
                    'midi_note': note,
                    'midi_velocity': velocity,
                    'duration': self.config.duration,
                    'timestamp': time.time(),
                    'audio_stats': {
                        'max': float(np.max(audio)),
                        'min': float(np.min(audio)),
                        'rms': float(np.sqrt(np.mean(audio**2))),
                        'length': len(audio)
                    }
                }
                
                # Save sample
                self._save_sample(sample_id, params, audio, mel, mfcc, metadata)
                
                sample_count += 1
                successful_count += 1
                
                if (sample_count) % 50 == 0:
                    print(f"Generated {sample_count}/{self.config.total_samples} samples")
                    
        print(f"\nAugmentation complete: {successful_count}/{self.config.total_samples} samples generated")
        print(f"Output directory: {self.config.output_dir}")
        
        # Save generation summary
        summary = {
            'config': {
                'strategy': self.config.strategy.value,
                'total_samples': self.config.total_samples,
                'successful_samples': successful_count,
                'duration': self.config.duration,
                'seed': self.config.seed
            },
            'strategy_distribution': {k.value: v for k, v in strategy_distribution.items()},
            'parameter_count': len(self.param_names),
            'timestamp': time.time()
        }
        
        summary_path = Path(self.config.output_dir) / 'generation_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Enhanced PolyMAX Dataset Augmentation')
    parser.add_argument('--strategy', choices=[s.value for s in AugmentationStrategy], 
                       default='all', help='Augmentation strategy')
    parser.add_argument('--count', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('--outdir', required=True, help='Output directory')
    parser.add_argument('--duration', type=float, default=4.0, help='Audio duration (seconds)')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--plugin', default='/Library/Audio/Plug-Ins/VST3/uaudio_polymax.vst3', 
                       help='PolyMAX VST3 path')
    parser.add_argument('--note-range', nargs=2, type=int, default=[48, 72], 
                       help='MIDI note range (min max)')
    parser.add_argument('--velocity-range', nargs=2, type=int, default=[80, 127],
                       help='MIDI velocity range (min max)')
    
    args = parser.parse_args()
    
    # Create configuration
    config = AugmentationConfig(
        strategy=AugmentationStrategy(args.strategy),
        total_samples=args.count,
        output_dir=args.outdir,
        duration=args.duration,
        note_range=tuple(args.note_range),
        velocity_range=tuple(args.velocity_range),
        seed=args.seed,
        plugin_path=args.plugin
    )
    
    # Run augmentation
    augmenter = EnhancedPolyMAXAugmentation(config)
    augmenter.initialize()
    augmenter.generate_samples()
    
    print("\nEnhanced dataset augmentation completed successfully!")
    print(f"Generated samples are ready for flow synthesizer training.")


if __name__ == '__main__':
    main()