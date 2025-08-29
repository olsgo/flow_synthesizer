#!/usr/bin/env python3
"""
PolyMAX Parameter Prediction Script

Predict PolyMAX synthesizer parameters from an input audio file using a trained
RegressionAE/CNN model saved by this project.
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List

import librosa
import numpy as np
import torch

# Add code directory to path for imports
sys.path.append(str(Path(__file__).parent / 'code'))
from utils.data import load_dataset
from utils.param_types import get_binary_param_indices

warnings.filterwarnings('ignore')

def compute_mel(audio, sr=22050, n_mels=64, target_frames=None):
    """Compute mel spectrogram and adapt width to target_frames if provided."""
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=2048,
        n_mels=n_mels,
        hop_length=1024,
        fmin=30,
        fmax=11000,
    )
    # Align time frames to target if requested
    if target_frames is not None:
        if mel.shape[1] < target_frames:
            pad = target_frames - mel.shape[1]
            mel = np.pad(mel, ((0, 0), (0, pad)), mode='constant')
        elif mel.shape[1] > target_frames:
            mel = mel[:, :target_frames]
    return mel

def load_polymax_param_names():
    """Load parameter names, prefer params_schema.json, fallback to legacy mapping."""
    schema_path = Path('params_schema.json')
    if schema_path.exists():
        try:
            with open(schema_path, 'r') as f:
                schema = json.load(f)
            if isinstance(schema, dict) and 'parameter_order' in schema:
                return list(schema['parameter_order'])
        except Exception:
            pass
    # Fallback to legacy mapping
    try:
        import ast
        with open('code/synth/polymax_params.txt', 'r') as f:
            idx2name = ast.literal_eval(f.read())
        max_idx = max(int(i) for i in idx2name.keys())
        return [idx2name[i] for i in range(max_idx + 1)]
    except Exception:
        return []


class PolyMAXPredictor:
    def __init__(self, model_path: str, device: str = 'auto'):
        self.device = self._get_device(device)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model_path = model_path
        self.model = torch.load(model_path, map_location=self.device).to(self.device).eval()
        # Build a reference dataset to retrieve normalization statistics
        args = argparse.Namespace(path='datasets', dataset='polymax_dataset', data='mel', batch_size=8, nbworkers=0, train_type='fixed')
        train_loader, _, _, _ = load_dataset(args)
        self.mel_mean = float(getattr(train_loader.dataset, 'means', {}).get('mel', getattr(train_loader.dataset, 'mean', 0.0)))
        self.mel_std = float(getattr(train_loader.dataset, 'vars', {}).get('mel', getattr(train_loader.dataset, 'var', 1.0)))
        # Derive expected input size (H, W)
        in_size = getattr(train_loader.dataset, 'input_size', (64, 80))
        # input_size can be (H, W) or (C, H, W)
        if len(in_size) == 2:
            self.n_mels, self.target_frames = int(in_size[0]), int(in_size[1])
        elif len(in_size) == 3:
            self.n_mels, self.target_frames = int(in_size[1]), int(in_size[2])
        else:
            self.n_mels, self.target_frames = 64, None
        self.param_names = load_polymax_param_names()
        self.sample_rate = 22050
        self.duration = 4.0
        # Determine binary param indices from schema names
        try:
            self.binary_indices = get_binary_param_indices(self.param_names)
        except Exception:
            self.binary_indices = []

    def _get_device(self, device: str) -> torch.device:
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            return torch.device('cpu')
        return torch.device(device)

    def preprocess_audio(self, audio_path: str) -> torch.Tensor:
        audio, _ = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)
        target_len = int(self.sample_rate * self.duration)
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]
        mel = compute_mel(audio, sr=self.sample_rate, n_mels=self.n_mels, target_frames=self.target_frames)
        # Log + normalize like dataset
        mel = np.log(mel + 1e-3)
        mel = (mel - self.mel_mean) / (self.mel_std + 1e-9)
        # Shape [1,1,H,W]
        return torch.from_numpy(mel).float().unsqueeze(0).unsqueeze(0).to(self.device)

    def predict(self, audio_path: str) -> Dict:
        x = self.preprocess_audio(audio_path)
        start = time.time()
        with torch.no_grad():
            # Optional reconstruction-based confidence if ae_model available
            recon_loss_val = None
            if hasattr(self.model, 'ae_model'):
                try:
                    x_tilde, _, _ = self.model.ae_model(x)
                    rec_loss = torch.nn.functional.smooth_l1_loss(x_tilde, x, reduction='mean')
                    recon_loss_val = float(rec_loss.detach().cpu().item())
                except Exception:
                    recon_loss_val = None
            y = self.model(x)
        elapsed = time.time() - start
        params = y.squeeze(0).detach().cpu().numpy().tolist()
        # Optional binary snapping (0/1) for toggle params
        if self.binary_indices:
            for i in self.binary_indices:
                if i < len(params):
                    params[i] = 1.0 if params[i] >= 0.5 else 0.0
        # Map to names if sizes agree
        mapped = {self.param_names[i] if i < len(self.param_names) else f'param_{i}': float(params[i]) for i in range(len(params))}
        # Heuristic confidence from reconstruction loss if available
        if recon_loss_val is None:
            confidence = 0.5
        else:
            # Map smaller loss -> higher confidence (0..1)
            confidence = float(np.exp(-recon_loss_val))
            confidence = float(np.clip(confidence, 0.0, 1.0))
        return {
            'file': audio_path,
            'parameters': mapped,
            'confidence': confidence,
            'prediction_time': elapsed,
        }

    # Compatibility wrappers used by integration and tests
    def predict_from_file(self, audio_path: str) -> Dict:
        res = self.predict(audio_path)
        res.update({'success': True})
        return res

    def predict_from_audio(self, audio_path: str) -> Dict:
        # Alias expected by test_ableton_simple
        return self.predict_from_file(audio_path)

    def predict_batch(self, files: List[str]) -> List[Dict]:
        return [self.predict(f) for f in files]

def main():
    parser = argparse.ArgumentParser(description='Predict PolyMAX parameters from audio')
    parser.add_argument('--audio', '-a', required=True, help='Input audio file path')
    parser.add_argument('--model', '-m', required=True, help='Path to trained model file (.model)')
    parser.add_argument('--output', '-o', help='Output JSON file for parameters')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'], help='Device to run inference on')
    parser.add_argument('--batch', nargs='+', help='Process multiple audio files')
    args = parser.parse_args()

    predictor = PolyMAXPredictor(args.model, args.device)
    if args.batch:
        results = predictor.predict_batch(args.batch)
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Saved batch predictions to {args.output}")
        else:
            print(json.dumps(results, indent=2))
    else:
        res = predictor.predict(args.audio)
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(res, f, indent=2)
            print(f"Saved prediction to {args.output}")
        else:
            print(json.dumps(res, indent=2))

if __name__ == '__main__':
    main()
