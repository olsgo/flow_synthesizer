#!/usr/bin/env python3
"""
Lightweight audio resynthesis using STFT magnitude + Griffin–Lim phase reconstruction.

- Reads an input audio file (any format supported by soundfile)
- Computes magnitude spectrogram and reconstructs waveform via Griffin–Lim
- Writes resynthesized audio and a metrics JSON comparing to the original

Dependencies: numpy, scipy, soundfile
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import soundfile as sf
from scipy.signal import stft, istft, get_window, correlate


@dataclass
class GLConfig:
    n_fft: int = 2048
    hop_length: int = 512
    window: str = "hann"
    iterations: int = 80  # more iterations => better reconstruction


def load_audio_mono(path: str) -> Tuple[np.ndarray, int]:
    y, sr = sf.read(path, always_2d=True)
    # Average to mono
    y_mono = y.mean(axis=1).astype(np.float32)
    return y_mono, sr


def normalize_rms(y: np.ndarray, target_rms: float) -> np.ndarray:
    rms = np.sqrt(np.mean(y ** 2) + 1e-12)
    if rms == 0:
        return y
    return (y * (target_rms / rms)).astype(np.float32)


def compute_stft(y: np.ndarray, sr: int, cfg: GLConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    win = get_window(cfg.window, cfg.n_fft, fftbins=True)
    f, t, Z = stft(y, fs=sr, window=win, nperseg=cfg.n_fft, noverlap=cfg.n_fft - cfg.hop_length, boundary=None, padded=False)
    return f, t, Z


def griffin_lim(magnitude: np.ndarray, sr: int, cfg: GLConfig) -> np.ndarray:
    """Reconstruct signal from magnitude-only STFT using Griffin–Lim."""
    win = get_window(cfg.window, cfg.n_fft, fftbins=True)

    # Random phase initialization
    angles = np.exp(1j * 2 * np.pi * np.random.rand(*magnitude.shape)).astype(np.complex64)
    S = magnitude.astype(np.complex64) * angles

    # Iterate
    for _ in range(cfg.iterations):
        _, x = istft(S, fs=sr, window=win, nperseg=cfg.n_fft, noverlap=cfg.n_fft - cfg.hop_length, boundary=None)
        # Recompute STFT and reset magnitude
        _, _, S_est = stft(x, fs=sr, window=win, nperseg=cfg.n_fft, noverlap=cfg.n_fft - cfg.hop_length, boundary=None, padded=False)
        # Match shape (may differ by 1 frame due to boundary handling)
        min_t = min(S_est.shape[1], S.shape[1])
        S = magnitude[:, :min_t] * np.exp(1j * np.angle(S_est[:, :min_t]))

    # Final ISTFT
    _, x = istft(S, fs=sr, window=win, nperseg=cfg.n_fft, noverlap=cfg.n_fft - cfg.hop_length, boundary=None)
    return x.astype(np.float32)


def time_align_max_corr(x: np.ndarray, y: np.ndarray, max_shift: int = 44100) -> Tuple[np.ndarray, np.ndarray]:
    """Align y to x by maximizing normalized cross-correlation within ±max_shift samples."""
    # Limit for efficiency
    max_shift = min(max_shift, len(x) - 1, len(y) - 1)
    # Compute centralized signals
    x0 = x - np.mean(x)
    y0 = y - np.mean(y)
    # Use scipy.signal.correlate (full) then restrict neighborhood
    corr = correlate(y0, x0, mode='full')
    lags = np.arange(-len(y0) + 1, len(x0))
    # Restrict to ±max_shift
    mask = (lags >= -max_shift) & (lags <= max_shift)
    corr = corr[mask]
    lags = lags[mask]
    best_lag = int(lags[np.argmax(corr)])
    # Positive best_lag => y lags behind x; shift y forward
    if best_lag > 0:
        y_aligned = y[best_lag:]
        x_aligned = x[:len(y_aligned)]
    elif best_lag < 0:
        x_aligned = x[-best_lag:]
        y_aligned = y[:len(x_aligned)]
    else:
        n = min(len(x), len(y))
        x_aligned, y_aligned = x[:n], y[:n]
    n = min(len(x_aligned), len(y_aligned))
    return x_aligned[:n], y_aligned[:n]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def compute_metrics(orig: np.ndarray, recon: np.ndarray, sr: int, cfg: GLConfig) -> Dict[str, float]:
    # Align signals for fair comparison
    x, y = time_align_max_corr(orig, recon, max_shift=sr // 2)
    if len(x) == 0 or len(y) == 0:
        return {
            "time_correlation": 0.0,
            "spectrogram_cosine": 0.0,
            "sdr_db": -np.inf,
            "accuracy": 0.0,
        }
    # Time-domain correlation
    x0 = x - np.mean(x)
    y0 = y - np.mean(y)
    time_corr = float(np.corrcoef(x0, y0)[0, 1]) if np.std(x0) > 0 and np.std(y0) > 0 else 0.0
    # STFT magnitude cosine similarity
    _, _, Zx = compute_stft(x, sr, cfg)
    _, _, Zy = compute_stft(y, sr, cfg)
    T = min(Zx.shape[1], Zy.shape[1])
    spec_cos = cosine_similarity(np.abs(Zx[:, :T]), np.abs(Zy[:, :T]))
    # SDR
    err = x - y
    sdr = 10.0 * np.log10((np.sum(x ** 2) + 1e-12) / (np.sum(err ** 2) + 1e-12))
    # Accuracy: use spectrogram cosine similarity as primary metric
    # This is robust to small phase/time misalignments and correlates with perceptual similarity.
    accuracy = spec_cos
    return {
        "time_correlation": float(time_corr),
        "spectrogram_cosine": float(spec_cos),
        "sdr_db": float(sdr),
        "accuracy": float(accuracy),
    }


def main():
    ap = argparse.ArgumentParser(description="Resynthesize audio via Griffin–Lim and report accuracy")
    ap.add_argument("input", help="Path to input audio file")
    ap.add_argument("--outdir", default="resynth_outputs", help="Directory to store outputs")
    ap.add_argument("--n_fft", type=int, default=2048)
    ap.add_argument("--hop", type=int, default=512)
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--match_rms", action="store_true", help="Match output RMS to original")
    args = ap.parse_args()

    in_path = args.input
    if not os.path.exists(in_path):
        raise FileNotFoundError(in_path)

    os.makedirs(args.outdir, exist_ok=True)

    cfg = GLConfig(n_fft=args.n_fft, hop_length=args.hop, iterations=args.iters)

    # Load
    y, sr = load_audio_mono(in_path)
    target_rms = float(np.sqrt(np.mean(y ** 2) + 1e-12))

    # STFT magnitude
    _, _, Z = compute_stft(y, sr, cfg)
    mag = np.abs(Z).astype(np.float32)

    # Griffin–Lim reconstruction
    y_hat = griffin_lim(mag, sr, cfg)
    if args.match_rms:
        y_hat = normalize_rms(y_hat, target_rms)

    # Save mono reconstruction
    base = os.path.splitext(os.path.basename(in_path))[0]
    out_wav = os.path.join(args.outdir, f"{base}_resynth_gl.wav")
    sf.write(out_wav, y_hat, sr)

    # Also produce an exact iSTFT reconstruction for reference (upper bound)
    win = get_window(cfg.window, cfg.n_fft, fftbins=True)
    _, y_exact = istft(Z, fs=sr, window=win, nperseg=cfg.n_fft, noverlap=cfg.n_fft - cfg.hop_length, boundary=None)
    y_exact = y_exact.astype(np.float32)
    if args.match_rms:
        y_exact = normalize_rms(y_exact, target_rms)
    out_wav_exact = os.path.join(args.outdir, f"{base}_resynth_exact.wav")
    sf.write(out_wav_exact, y_exact, sr)

    # Metrics
    metrics_gl = compute_metrics(y, y_hat, sr, cfg)
    metrics_exact = compute_metrics(y, y_exact, sr, cfg)
    metrics = {
        "input": in_path,
        "output_gl": out_wav,
        "output_exact": out_wav_exact,
        "config": {
            "n_fft": cfg.n_fft,
            "hop_length": cfg.hop_length,
            "iterations": cfg.iterations,
            "window": cfg.window,
        },
        "metrics_gl": metrics_gl,
        "metrics_exact": metrics_exact,
    }

    out_json = os.path.join(args.outdir, f"{base}_metrics.json")
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps({
        "output_gl": out_wav,
        "output_exact": out_wav_exact,
        "metrics_gl_accuracy": metrics_gl.get("accuracy", 0.0),
        "metrics_exact_accuracy": metrics_exact.get("accuracy", 0.0),
        "metrics_file": out_json,
    }, indent=2))


if __name__ == "__main__":
    main()
