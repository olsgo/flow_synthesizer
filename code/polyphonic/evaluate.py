#!/usr/bin/env python3
"""
Evaluation script for polyphonic Flow Synth reconstruction.

Compares original vs. reconstructed audio using objective metrics and diagnostics.
"""

import argparse
import os
import sys
import json
import tempfile
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import librosa
import soundfile as sf

# Internal imports
from code.polyphonic.transcribe import NoteEvent, load_events_from_json

class PolyphonicEvaluator:
    """Evaluator for polyphonic reconstruction quality."""
    
    def __init__(self, sr: int = 22050, n_mels: int = 128):
        self.sr = sr
        self.n_mels = n_mels
    
    def load_audio(self, audio_path: str) -> np.ndarray:
        """Load audio file."""
        y, _ = librosa.load(audio_path, sr=self.sr)
        return y
    
    def compute_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Compute mel spectrogram."""
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.sr, n_mels=self.n_mels,
            n_fft=2048, hop_length=1024
        )
        return librosa.power_to_db(mel_spec, ref=np.max)
    
    def spectral_convergence(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """
        Compute spectral convergence metric.
        
        SC = ||X - X_hat||_F / ||X||_F
        where ||.||_F is the Frobenius norm
        """
        # Ensure same size
        min_frames = min(original.shape[1], reconstructed.shape[1])
        orig_crop = original[:, :min_frames]
        recon_crop = reconstructed[:, :min_frames]
        
        numerator = np.linalg.norm(orig_crop - recon_crop, ord='fro')
        denominator = np.linalg.norm(orig_crop, ord='fro')
        
        return float(numerator / (denominator + 1e-8))
    
    def log_magnitude_mse(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """
        Compute log-magnitude MSE.
        
        MSE = mean((log|X| - log|X_hat|)^2)
        """
        # Ensure same size
        min_frames = min(original.shape[1], reconstructed.shape[1])
        orig_crop = original[:, :min_frames]
        recon_crop = reconstructed[:, :min_frames]
        
        # Convert to linear magnitude (spectrograms are in dB)
        orig_mag = np.power(10, orig_crop / 20.0)
        recon_mag = np.power(10, recon_crop / 20.0)
        
        # Compute log magnitude MSE
        log_orig = np.log(orig_mag + 1e-8)
        log_recon = np.log(recon_mag + 1e-8)
        
        mse = np.mean((log_orig - log_recon) ** 2)
        return float(mse)
    
    def compute_note_diagnostics(self, events: List[NoteEvent]) -> Dict[str, Any]:
        """Compute diagnostics about note events."""
        if not events:
            return {
                'note_count': 0,
                'polyphony_histogram': {},
                'pitch_range': (0, 0),
                'duration_stats': {},
                'velocity_stats': {}
            }
        
        # Basic counts
        note_count = len(events)
        
        # Pitch analysis
        pitches = [event.pitch for event in events]
        pitch_range = (min(pitches), max(pitches))
        
        # Duration analysis
        durations = [event.duration_beats for event in events]
        duration_stats = {
            'mean': np.mean(durations),
            'std': np.std(durations),
            'min': np.min(durations),
            'max': np.max(durations)
        }
        
        # Velocity analysis
        velocities = [event.velocity for event in events]
        velocity_stats = {
            'mean': np.mean(velocities),
            'std': np.std(velocities),
            'min': np.min(velocities),
            'max': np.max(velocities)
        }
        
        # Polyphony analysis
        polyphony_histogram = self.compute_polyphony_histogram(events)
        
        return {
            'note_count': note_count,
            'polyphony_histogram': polyphony_histogram,
            'pitch_range': pitch_range,
            'duration_stats': duration_stats,
            'velocity_stats': velocity_stats
        }
    
    def compute_polyphony_histogram(self, events: List[NoteEvent], 
                                  time_resolution: float = 0.1) -> Dict[int, int]:
        """
        Compute histogram of polyphony levels over time.
        
        Args:
            events: List of note events
            time_resolution: Time resolution in beats
            
        Returns:
            Dictionary mapping polyphony level to count
        """
        if not events:
            return {}
        
        # Find total duration
        max_time = max(event.onset_beats + event.duration_beats for event in events)
        time_steps = int(max_time / time_resolution) + 1
        
        # Count active notes at each time step
        polyphony_levels = []
        
        for step in range(time_steps):
            current_time = step * time_resolution
            active_notes = 0
            
            for event in events:
                if (event.onset_beats <= current_time < 
                    event.onset_beats + event.duration_beats):
                    active_notes += 1
            
            polyphony_levels.append(active_notes)
        
        # Create histogram
        histogram = {}
        for level in polyphony_levels:
            histogram[level] = histogram.get(level, 0) + 1
        
        return histogram
    
    def compute_onset_errors(self, original_events: List[NoteEvent], 
                           transcribed_events: List[NoteEvent],
                           tolerance: float = 0.1) -> Dict[str, float]:
        """
        Compute onset timing errors between original and transcribed events.
        
        This is a simplified version - a full implementation would do proper
        note matching based on pitch and timing.
        """
        if not original_events or not transcribed_events:
            return {'avg_onset_error': float('inf'), 'onset_precision': 0.0, 'onset_recall': 0.0}
        
        # Simple onset error: compare onset times within tolerance
        orig_onsets = sorted([event.onset_beats for event in original_events])
        trans_onsets = sorted([event.onset_beats for event in transcribed_events])
        
        # Find matched onsets
        matched = 0
        errors = []
        
        for orig_onset in orig_onsets:
            # Find closest transcribed onset
            closest_dist = min(abs(orig_onset - trans_onset) for trans_onset in trans_onsets)
            if closest_dist <= tolerance:
                matched += 1
                errors.append(closest_dist)
        
        # Compute metrics
        avg_onset_error = np.mean(errors) if errors else float('inf')
        onset_precision = matched / len(trans_onsets) if trans_onsets else 0.0
        onset_recall = matched / len(orig_onsets) if orig_onsets else 0.0
        
        return {
            'avg_onset_error': float(avg_onset_error),
            'onset_precision': float(onset_precision),
            'onset_recall': float(onset_recall)
        }
    
    def evaluate_reconstruction(self, original_audio_path: str, 
                              reconstructed_audio_path: str,
                              events_path: Optional[str] = None,
                              original_events_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate polyphonic reconstruction quality.
        
        Args:
            original_audio_path: Path to original audio
            reconstructed_audio_path: Path to reconstructed audio
            events_path: Path to transcribed events JSON file
            original_events_path: Path to ground truth events (if available)
            
        Returns:
            Dictionary with evaluation metrics
        """
        results = {}
        
        # Load audio
        try:
            original_audio = self.load_audio(original_audio_path)
            reconstructed_audio = self.load_audio(reconstructed_audio_path)
        except Exception as e:
            return {'error': f'Failed to load audio: {e}'}
        
        # Compute spectrograms
        try:
            original_mel = self.compute_mel_spectrogram(original_audio)
            reconstructed_mel = self.compute_mel_spectrogram(reconstructed_audio)
        except Exception as e:
            return {'error': f'Failed to compute spectrograms: {e}'}
        
        # Spectral metrics
        try:
            results['spectral_convergence'] = self.spectral_convergence(original_mel, reconstructed_mel)
            results['log_magnitude_mse'] = self.log_magnitude_mse(original_mel, reconstructed_mel)
        except Exception as e:
            results['spectral_error'] = str(e)
        
        # Audio duration comparison
        results['original_duration'] = len(original_audio) / self.sr
        results['reconstructed_duration'] = len(reconstructed_audio) / self.sr
        results['duration_ratio'] = results['reconstructed_duration'] / results['original_duration']
        
        # Event diagnostics
        if events_path and os.path.exists(events_path):
            try:
                events = load_events_from_json(events_path)
                results['note_diagnostics'] = self.compute_note_diagnostics(events)
            except Exception as e:
                results['events_error'] = str(e)
        
        # Onset error analysis (if ground truth available)
        if (original_events_path and os.path.exists(original_events_path) and 
            events_path and os.path.exists(events_path)):
            try:
                original_events = load_events_from_json(original_events_path)
                transcribed_events = load_events_from_json(events_path)
                results['onset_errors'] = self.compute_onset_errors(original_events, transcribed_events)
            except Exception as e:
                results['onset_error_analysis_error'] = str(e)
        
        return results

def generate_html_report(results: Dict[str, Any], output_path: str):
    """Generate HTML report from evaluation results."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Polyphonic Flow Synth Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .metric {{ margin: 10px 0; }}
            .section {{ margin: 20px 0; border-left: 3px solid #007acc; padding-left: 15px; }}
            .error {{ color: red; }}
            .good {{ color: green; }}
            .warning {{ color: orange; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>Polyphonic Flow Synth Evaluation Report</h1>
    """
    
    # Spectral metrics section
    if 'spectral_convergence' in results:
        html_content += """
        <div class="section">
            <h2>Spectral Quality Metrics</h2>
        """
        
        sc = results['spectral_convergence']
        sc_class = 'good' if sc < 0.5 else 'warning' if sc < 0.8 else 'error'
        
        lm_mse = results.get('log_magnitude_mse', 'N/A')
        lm_class = 'good' if isinstance(lm_mse, (int, float)) and lm_mse < 1.0 else 'warning'
        
        html_content += f"""
            <div class="metric">
                <strong>Spectral Convergence:</strong> 
                <span class="{sc_class}">{sc:.4f}</span>
                (lower is better, < 0.5 is good)
            </div>
            <div class="metric">
                <strong>Log Magnitude MSE:</strong> 
                <span class="{lm_class}">{lm_mse if isinstance(lm_mse, str) else f'{lm_mse:.4f}'}</span>
                (lower is better, < 1.0 is good)
            </div>
        </div>
        """
    
    # Duration analysis
    if 'original_duration' in results:
        html_content += f"""
        <div class="section">
            <h2>Duration Analysis</h2>
            <div class="metric">
                <strong>Original Duration:</strong> {results['original_duration']:.2f}s
            </div>
            <div class="metric">
                <strong>Reconstructed Duration:</strong> {results['reconstructed_duration']:.2f}s
            </div>
            <div class="metric">
                <strong>Duration Ratio:</strong> {results['duration_ratio']:.3f}
            </div>
        </div>
        """
    
    # Note diagnostics
    if 'note_diagnostics' in results:
        diag = results['note_diagnostics']
        html_content += f"""
        <div class="section">
            <h2>Note Event Diagnostics</h2>
            <div class="metric">
                <strong>Note Count:</strong> {diag['note_count']}
            </div>
            <div class="metric">
                <strong>Pitch Range:</strong> {diag['pitch_range'][0]} - {diag['pitch_range'][1]}
            </div>
        """
        
        if 'duration_stats' in diag:
            stats = diag['duration_stats']
            html_content += f"""
            <div class="metric">
                <strong>Duration Stats:</strong> 
                Mean: {stats['mean']:.2f}, 
                Std: {stats['std']:.2f}, 
                Range: {stats['min']:.2f} - {stats['max']:.2f} beats
            </div>
            """
        
        # Polyphony histogram
        if 'polyphony_histogram' in diag:
            html_content += """
            <h3>Polyphony Histogram</h3>
            <table>
                <tr><th>Polyphony Level</th><th>Time Steps</th><th>Percentage</th></tr>
            """
            
            hist = diag['polyphony_histogram']
            total_steps = sum(hist.values())
            
            for level in sorted(hist.keys()):
                count = hist[level]
                percentage = (count / total_steps) * 100 if total_steps > 0 else 0
                html_content += f"""
                <tr>
                    <td>{level}</td>
                    <td>{count}</td>
                    <td>{percentage:.1f}%</td>
                </tr>
                """
            
            html_content += "</table>"
        
        html_content += "</div>"
    
    # Onset error analysis
    if 'onset_errors' in results:
        errors = results['onset_errors']
        html_content += f"""
        <div class="section">
            <h2>Onset Error Analysis</h2>
            <div class="metric">
                <strong>Average Onset Error:</strong> {errors['avg_onset_error']:.4f} beats
            </div>
            <div class="metric">
                <strong>Onset Precision:</strong> {errors['onset_precision']:.3f}
            </div>
            <div class="metric">
                <strong>Onset Recall:</strong> {errors['onset_recall']:.3f}
            </div>
        </div>
        """
    
    # Errors section
    error_keys = [k for k in results.keys() if 'error' in k]
    if error_keys:
        html_content += """
        <div class="section">
            <h2>Errors and Warnings</h2>
        """
        for key in error_keys:
            html_content += f"""
            <div class="metric error">
                <strong>{key}:</strong> {results[key]}
            </div>
            """
        html_content += "</div>"
    
    html_content += """
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_content)

def main():
    """CLI interface for polyphonic evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate polyphonic reconstruction quality')
    parser.add_argument('original_audio', help='Original audio file')
    parser.add_argument('reconstructed_audio', help='Reconstructed audio file')
    parser.add_argument('--events', help='Transcribed events JSON file')
    parser.add_argument('--original_events', help='Ground truth events JSON file')
    parser.add_argument('--output', help='Output report path (HTML or JSON)')
    parser.add_argument('--format', choices=['html', 'json'], default='html', help='Output format')
    
    args = parser.parse_args()
    
    # Validate inputs
    for path in [args.original_audio, args.reconstructed_audio]:
        if not os.path.exists(path):
            print(f"Error: File not found: {path}")
            sys.exit(1)
    
    # Set output path
    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(os.path.basename(args.original_audio))[0]
        ext = '.html' if args.format == 'html' else '.json'
        output_path = f"{base_name}_evaluation{ext}"
    
    # Run evaluation
    evaluator = PolyphonicEvaluator()
    
    print("Running polyphonic reconstruction evaluation...")
    results = evaluator.evaluate_reconstruction(
        args.original_audio,
        args.reconstructed_audio,
        args.events,
        args.original_events
    )
    
    # Generate output
    if args.format == 'html':
        generate_html_report(results, output_path)
        print(f"HTML report saved to: {output_path}")
    else:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"JSON results saved to: {output_path}")
    
    # Print summary to console
    print("\nEvaluation Summary:")
    if 'spectral_convergence' in results:
        print(f"  Spectral Convergence: {results['spectral_convergence']:.4f}")
    if 'log_magnitude_mse' in results:
        print(f"  Log Magnitude MSE: {results['log_magnitude_mse']:.4f}")
    if 'note_diagnostics' in results:
        diag = results['note_diagnostics']
        print(f"  Note Count: {diag['note_count']}")
        hist = diag.get('polyphony_histogram', {})
        max_poly = max(hist.keys()) if hist else 0
        print(f"  Max Polyphony: {max_poly}")

if __name__ == '__main__':
    main()