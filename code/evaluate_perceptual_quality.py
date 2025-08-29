#!/usr/bin/env python3
"""
Perceptual Quality Evaluation Tool (PolyMAX)

Evaluates perceptual similarity by rendering audio from model-predicted PolyMAX
parameters using the integrated pedalboard backend, and comparing to dataset
ground-truth audio via spectral features.
"""

import torch
import numpy as np
import librosa
import json
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from scipy.spatial.distance import cosine
import argparse
import os

# Import model and data utilities
import sys
sys.path.append('.')
from utils.data import load_dataset

class PerceptualQualityEvaluator:
    def __init__(self, model_path, dataset_path='datasets', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model_path = model_path
        self.dataset_path = dataset_path
        
        # Audio processing parameters
        self.sample_rate = 22050
        self.duration = 4.0  # seconds
        self.n_fft = 2048
        self.hop_length = 512
        self.n_mels = 128
        
        # Synth engine (lazy)
        self.engine = None
        self.generator = None
        self.param_defaults = None
        self.rev_idx = None
        self.param_names = None
        
    def load_model(self):
        """Load the trained model (saved via torch.save full model)."""
        print(f"Loading model from {self.model_path}")
        model = torch.load(self.model_path, map_location=self.device)
        model.to(self.device)
        model.eval()
        return model
        
    def _ensure_synth(self, param_names):
        if self.engine is not None:
            return
        from synth.synthesize import create_synth
        # Use PolyMAX synth with pedalboard backend
        self.engine, self.generator, self.param_defaults, self.rev_idx = create_synth('polymax_dataset', 'polymax')
        self.param_names = list(param_names)
        print("Synth engine initialized (PolyMAX)")
        
    def extract_audio_features(self, audio_wave):
        """Extract comprehensive audio features for comparison"""
        try:
            y = np.asarray(audio_wave)
            # Convert to mono if stereo (2 x N or N x 2)
            if y.ndim == 2:
                if y.shape[0] == 2:
                    y = y.mean(axis=0)
                elif y.shape[1] == 2:
                    y = y.mean(axis=1)
            sr = self.sample_rate
            
            # Spectral features
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            # MFCC features (short summary)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            
            # Tempo and rhythm
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='time', tightness=100)
            try:
                tempo_val = float(tempo)
            except Exception:
                tempo_val = float(np.asarray(tempo).reshape(-1)[0])
            
            # RMS energy
            rms = librosa.feature.rms(y=y)[0]
            
            features = {
                'mel_spectrogram': mel_spec_db,
                'mfccs': mfccs,
                'spectral_centroid_mean': np.mean(spectral_centroids),
                'spectral_centroid_std': np.std(spectral_centroids),
                'spectral_rolloff_mean': np.mean(spectral_rolloff),
                'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
                'zero_crossing_rate_mean': np.mean(zero_crossing_rate),
                'chroma': chroma,
                'tempo': tempo_val,
                'rms_mean': np.mean(rms),
                'rms_std': np.std(rms)
            }
            
            return features
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
            
    def calculate_perceptual_similarity(self, features1, features2):
        """Calculate perceptual similarity between two audio feature sets"""
        similarities = {}
        
        # Mel spectrogram similarity (cosine similarity)
        mel1_flat = features1['mel_spectrogram'].flatten()
        mel2_flat = features2['mel_spectrogram'].flatten()
        mel_similarity = 1 - cosine(mel1_flat, mel2_flat)
        similarities['mel_spectrogram_similarity'] = mel_similarity
        
        # MFCC similarity
        mfcc1_flat = features1['mfccs'].flatten()
        mfcc2_flat = features2['mfccs'].flatten()
        mfcc_similarity = 1 - cosine(mfcc1_flat, mfcc2_flat)
        similarities['mfcc_similarity'] = mfcc_similarity
        
        # Chroma similarity
        chroma1_flat = features1['chroma'].flatten()
        chroma2_flat = features2['chroma'].flatten()
        chroma_similarity = 1 - cosine(chroma1_flat, chroma2_flat)
        similarities['chroma_similarity'] = chroma_similarity
        
        # Spectral feature differences
        similarities['spectral_centroid_diff'] = abs(
            features1['spectral_centroid_mean'] - features2['spectral_centroid_mean']
        )
        similarities['spectral_rolloff_diff'] = abs(
            features1['spectral_rolloff_mean'] - features2['spectral_rolloff_mean']
        )
        similarities['spectral_bandwidth_diff'] = abs(
            features1['spectral_bandwidth_mean'] - features2['spectral_bandwidth_mean']
        )
        
        # Tempo difference
        similarities['tempo_diff'] = abs(features1['tempo'] - features2['tempo'])
        
        # RMS energy difference
        similarities['rms_diff'] = abs(features1['rms_mean'] - features2['rms_mean'])
        
        # Overall perceptual similarity (weighted combination)
        overall_similarity = (
            0.5 * mel_similarity +
            0.3 * mfcc_similarity +
            0.2 * chroma_similarity
        )
        similarities['overall_similarity'] = overall_similarity
        
        return similarities
        
    def evaluate_sample_batch(self, model, dataloader, num_samples=32):
        """Evaluate perceptual quality using in-process PolyMAX rendering."""
        from synth.synthesize import synthesize_batch
        results = []
        evaluated = 0
        
        print(f"Evaluating {num_samples} samples for perceptual quality...")
        with torch.no_grad():
            for batch_idx, (data, params, meta, audio_gt) in enumerate(dataloader):
                if evaluated >= num_samples:
                    break
                data = data.to(self.device)
                # Predict params
                pred = model(data).detach().cpu()
                # De-normalize predictions to [0,1] domain using dataset stats
                ds = dataloader.dataset
                if hasattr(ds, 'params_mean') and hasattr(ds, 'params_std'):
                    mean = ds.params_mean.detach().cpu().unsqueeze(0)
                    std = ds.params_std.detach().cpu().unsqueeze(0)
                    pred = (pred * std) + mean
                pred = pred.clamp(0.0, 1.0)
                # Init synth lazily with dataset param names
                self._ensure_synth(dataloader.dataset.final_params)
                # Map dataset parameter order to plugin parameter names:
                #  - Use exact name match when possible
                #  - Fallback: plugin default key at same index
                plugin_keys = list(self.param_defaults.keys())
                ds_names = list(dataloader.dataset.final_params)
                name_map = []
                for i, n in enumerate(ds_names):
                    if n in self.param_defaults:
                        name_map.append(n)
                    else:
                        # Fallback to same index within available plugin keys
                        idx = min(i, len(plugin_keys) - 1)
                        name_map.append(plugin_keys[idx])
                # Render predicted audio
                audio_pred = synthesize_batch(pred, name_map, self.engine, self.generator, self.param_defaults, self.rev_idx, orig_wave=None, name=None)
                
                bs = min(len(audio_pred), num_samples - evaluated)
                for i in range(bs):
                    # audio_pred[i] is np array (float32) at 22050 Hz
                    pred_feats = self.extract_audio_features(audio_pred[i])
                    gt_feats = self.extract_audio_features(audio_gt[i].numpy())
                    if pred_feats and gt_feats:
                        sim = self.calculate_perceptual_similarity(pred_feats, gt_feats)
                        results.append({'sample_idx': evaluated + i, **sim})
                        print(f"Sample {evaluated + i}: Overall similarity = {sim['overall_similarity']:.4f}")
                evaluated += bs
        return results
        
    def create_perceptual_report(self, results, output_dir):
        """Create comprehensive perceptual quality report"""
        if not results:
            print("No valid results to analyze")
            return
            
        df = pd.DataFrame(results)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print("\n" + "="*80)
        print("PERCEPTUAL QUALITY EVALUATION REPORT")
        print("="*80)
        
        # Overall statistics
        print(f"\nOVERALL PERCEPTUAL SIMILARITY:")
        print(f"Mean Overall Similarity: {df['overall_similarity'].mean():.4f} ± {df['overall_similarity'].std():.4f}")
        print(f"Median Overall Similarity: {df['overall_similarity'].median():.4f}")
        print(f"Min/Max Overall Similarity: {df['overall_similarity'].min():.4f} / {df['overall_similarity'].max():.4f}")
        
        # Feature-specific similarities
        print(f"\nFEATURE-SPECIFIC SIMILARITIES:")
        print(f"Mel Spectrogram: {df['mel_spectrogram_similarity'].mean():.4f} ± {df['mel_spectrogram_similarity'].std():.4f}")
        print(f"MFCC: {df['mfcc_similarity'].mean():.4f} ± {df['mfcc_similarity'].std():.4f}")
        print(f"Chroma: {df['chroma_similarity'].mean():.4f} ± {df['chroma_similarity'].std():.4f}")
        
        # Spectral differences
        print(f"\nSPECTRAL FEATURE DIFFERENCES:")
        print(f"Spectral Centroid Diff: {df['spectral_centroid_diff'].mean():.2f} Hz")
        print(f"Spectral Rolloff Diff: {df['spectral_rolloff_diff'].mean():.2f} Hz")
        print(f"Tempo Diff: {df['tempo_diff'].mean():.2f} BPM")
        
        # Create visualizations
        self.create_perceptual_visualizations(df, output_dir)
        
        # Save detailed results
        df.to_csv(output_dir / 'perceptual_evaluation_results.csv', index=False)
        
        return df
        
    def create_perceptual_visualizations(self, df, output_dir):
        """Create visualizations for perceptual evaluation"""
        # 1. Similarity distribution
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.hist(df['overall_similarity'], bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Overall Similarity')
        plt.ylabel('Count')
        plt.title('Distribution of Overall Similarity')
        plt.axvline(df['overall_similarity'].mean(), color='red', linestyle='--', label=f'Mean: {df["overall_similarity"].mean():.3f}')
        plt.legend()
        
        plt.subplot(2, 3, 2)
        plt.hist(df['mel_spectrogram_similarity'], bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Mel Spectrogram Similarity')
        plt.ylabel('Count')
        plt.title('Mel Spectrogram Similarity')
        
        plt.subplot(2, 3, 3)
        plt.hist(df['mfcc_similarity'], bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('MFCC Similarity')
        plt.ylabel('Count')
        plt.title('MFCC Similarity')
        
        plt.subplot(2, 3, 4)
        plt.scatter(df['mel_spectrogram_similarity'], df['overall_similarity'], alpha=0.6)
        plt.xlabel('Mel Spectrogram Similarity')
        plt.ylabel('Overall Similarity')
        plt.title('Mel vs Overall Similarity')
        
        plt.subplot(2, 3, 5)
        plt.hist(df['tempo_diff'], bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Tempo Difference (BPM)')
        plt.ylabel('Count')
        plt.title('Tempo Difference Distribution')
        
        plt.subplot(2, 3, 6)
        plt.hist(df['spectral_centroid_diff'], bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Spectral Centroid Difference (Hz)')
        plt.ylabel('Count')
        plt.title('Spectral Centroid Difference')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'perceptual_similarity_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Perceptual visualizations saved to {output_dir}")
        
    def run_evaluation(self, num_samples=32, output_dir='outputs_optimized/perceptual_analysis'):
        """Run the complete perceptual quality evaluation"""
        print("Starting Perceptual Quality Evaluation...")
        
        # Load model
        model = self.load_model()
        
        # Get test dataloader
        args = argparse.Namespace(
            path=self.dataset_path,
            dataset='polymax_dataset',
            data='mel',
            batch_size=8,
            nbworkers=0,
            train_type='fixed'
        )
        _, _, test_loader, _ = load_dataset(args)
        
        # Evaluate samples
        results = self.evaluate_sample_batch(model, test_loader, num_samples)
        
        # Create report
        report_df = self.create_perceptual_report(results, output_dir)
        
        print(f"\nPerceptual evaluation complete! Results saved to {output_dir}")
        return results, report_df

def main():
    parser = argparse.ArgumentParser(description='Evaluate perceptual quality of model predictions')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model (.model)')
    parser.add_argument('--dataset_path', type=str, default='datasets', help='Path to dataset')
    parser.add_argument('--num_samples', type=int, default=32, help='Number of samples to evaluate')
    parser.add_argument('--output_dir', type=str, default='outputs_optimized/perceptual_analysis', help='Output directory')
    
    args = parser.parse_args()
    
    evaluator = PerceptualQualityEvaluator(
        model_path=args.model_path,
        dataset_path=args.dataset_path
    )
    
    results, report_df = evaluator.run_evaluation(args.num_samples, args.output_dir)
    
if __name__ == '__main__':
    main()
