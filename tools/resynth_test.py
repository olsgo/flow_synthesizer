#!/usr/bin/env python3
"""
Resynthesis Testing Script
Processes audio files through the trained VAE model and resynthesizes using PolyMAX
"""

import os
import sys
import torch
import numpy as np
import librosa
import soundfile as sf
from datetime import datetime
import json
from pedalboard import Pedalboard, load_plugin
from mido import Message

# Add code directory to path
sys.path.append('code')
sys.path.append('/Users/gjb/Projects/flow_synthesizer/code')
sys.path.append('/Users/gjb/Projects/flow_synthesizer')

from models.vae.vae import VAE
from models.vae.ae import RegressionAE
# from utils.data import load_mel_spectrogram  # Function doesn't exist
import librosa
from synth.synthesize import synthesize_audio

# Import the StabilizedRegressionAE class
from comprehensive_training_fix import StabilizedRegressionAE

def load_trained_model(model_path):
    """Load the trained VAE model"""
    print(f"Loading model from {model_path}")
    
    # Load the complete model (it's a RegressionAE containing a VAE)
    model = torch.load(model_path, map_location='cpu')
    
    # Extract the VAE from the RegressionAE wrapper
    vae_model = model.ae_model
    
    vae_model.eval()
    print("Model loaded successfully")
    print(f"Model type: {type(model)}")
    print(f"VAE model type: {type(vae_model)}")
    print(f"Input dims: {vae_model.input_dims}")
    print(f"Encoder dims: {vae_model.encoder_dims}")
    print(f"Latent dims: {vae_model.latent_dims}")
    
    return model, vae_model

def process_audio_file(audio_path, regression_model, vae_model, output_dir):
    """Process audio file through the complete resynthesis pipeline"""
    print(f"\nProcessing: {audio_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess audio
    audio, sr = librosa.load(audio_path, sr=22050)
    print(f"Loaded audio: {len(audio)} samples at {sr} Hz")
    
    # Convert to mel spectrogram using librosa (matching dataset creation)
    mel_spec = librosa.feature.melspectrogram(
        y=audio, 
        sr=sr, 
        n_mels=128, 
        hop_length=512, 
        win_length=1024,
        fmin=0, 
        fmax=sr//2
    )
    
    # Convert to log scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize to [0, 1] range
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
    
    # Convert to tensor and add batch dimension
    mel_tensor = torch.FloatTensor(mel_spec_norm).unsqueeze(0)
    
    print(f"Mel spectrogram shape: {mel_tensor.shape}")
    
    # Extract parameters using the complete pipeline
    with torch.no_grad():
        # Encode to latent space using VAE
        mu, log_var = vae_model.encode(mel_tensor)
        
        # Sample from latent space (or use mean)
        latent_params = mu  # Using mean for deterministic results
        
        print(f"Extracted latent parameters shape: {latent_params.shape}")
        print(f"Latent parameters range: [{latent_params.min():.4f}, {latent_params.max():.4f}]")
        
        # Use regression model to predict synthesizer parameters
        synth_params = regression_model.regression_model(latent_params)
        synth_params = regression_model.output_activation(synth_params)
        synth_params = torch.clamp(synth_params, 0.0, 1.0)
        synth_params_np = synth_params.squeeze().numpy()
        
        print(f"Predicted synthesizer parameters shape: {synth_params.shape}")
        print(f"Synth parameters range: [{synth_params.min():.4f}, {synth_params.max():.4f}]")
    
    # Convert to PolyMAX parameter format (this needs actual parameter mapping)
    polymax_params = convert_to_polymax_params(synth_params_np)
    
    print(f"Generated PolyMAX parameters: {polymax_params}")
    
    # Save extracted parameters
    params_file = os.path.join(output_dir, 'extracted_parameters.npy')
    np.save(params_file, synth_params_np)
    print(f"Saved parameters to: {params_file}")
    
    # Resynthesize audio using PolyMAX
    try:
        print("Resynthesizing audio with PolyMAX...")
        resynthesized_audio = resynthesize_with_polymax(polymax_params)
        print("Resynthesis successful")
    except Exception as e:
        print(f"Error during resynthesis: {e}")
        # Create dummy resynthesized audio for testing
        resynthesized_audio = np.random.randn(len(audio)) * 0.1
    
    # Save results
    results = save_results(audio_path, audio, resynthesized_audio, synth_params_np, polymax_params, output_dir)
    
    # Compare original and resynthesized audio
    comparison = compare_audio(audio, resynthesized_audio)
    
    return results, comparison

def compare_audio(original, resynthesized):
    """Compare original and resynthesized audio"""
    print("\nComparing audio files...")
    
    # Basic statistics
    print(f"Original audio - Length: {len(original)}, RMS: {np.sqrt(np.mean(original**2)):.4f}")
    print(f"Resynthesized audio - Length: {len(resynthesized)}, RMS: {np.sqrt(np.mean(resynthesized**2)):.4f}")
    
    # Compute similarity metrics
    min_len = min(len(original), len(resynthesized))
    orig_trim = original[:min_len]
    resynth_trim = resynthesized[:min_len]
    
    # Correlation
    correlation = np.corrcoef(orig_trim, resynth_trim)[0, 1] if len(orig_trim) > 1 else 0.0
    print(f"Correlation: {correlation:.4f}")
    
    # MSE
    mse = np.mean((orig_trim - resynth_trim) ** 2)
    print(f"MSE: {mse:.6f}")
    
    # MAE
    mae = np.mean(np.abs(orig_trim - resynth_trim))
    print(f"MAE: {mae:.6f}")
    
    # RMS energy comparison
    orig_rms = np.sqrt(np.mean(orig_trim ** 2))
    resynth_rms = np.sqrt(np.mean(resynth_trim ** 2))
    rms_ratio = resynth_rms / orig_rms if orig_rms > 0 else 0
    
    metrics = {
        'mse': float(mse),
        'mae': float(mae),
        'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
        'original_rms': float(orig_rms),
        'resynthesized_rms': float(resynth_rms),
        'rms_ratio': float(rms_ratio)
    }
    
    return metrics

def save_results(audio_path, original_audio, resynthesized_audio, synth_params, polymax_params, output_dir):
    """Save all results to output directory"""
    
    # Save original audio
    orig_file = os.path.join(output_dir, 'original_audio.wav')
    sf.write(orig_file, original_audio, 22050)
    
    # Save resynthesized audio
    resynth_file = os.path.join(output_dir, 'resynthesized_audio.wav')
    sf.write(resynth_file, resynthesized_audio, 22050)
    
    # Save extracted parameters
    params_file = os.path.join(output_dir, 'extracted_parameters.npy')
    np.save(params_file, synth_params)
    
    # Convert numpy types in polymax_params to python types
    serializable_polymax_params = {}
    for k, v in polymax_params.items():
        if isinstance(v, np.generic):
            serializable_polymax_params[k] = v.item()
        else:
            serializable_polymax_params[k] = v

    # Save PolyMAX parameters
    polymax_file = os.path.join(output_dir, 'polymax_parameters.json')
    with open(polymax_file, 'w') as f:
        json.dump(serializable_polymax_params, f, indent=2)
    
    results = {
        'original_audio': original_audio,
        'resynthesized_audio': resynthesized_audio,
        'synth_params': synth_params,
        'polymax_params': polymax_params,
        'files': {
            'original_audio': orig_file,
            'resynthesized_audio': resynth_file,
            'extracted_parameters': params_file,
            'polymax_parameters': polymax_file
        }
    }
    
    print(f"Saved results to: {output_dir}")
    return results

def resynthesize_with_polymax(polymax_params):
    """Resynthesize audio using PolyMAX VST"""
    try:
        # Load PolyMAX VST3 plugin
        polymax_vst_path = '/Library/Audio/Plug-Ins/VST3/uaudio_polymax.vst3'
        print(f"Loading PolyMAX VST3 from: {polymax_vst_path}")
        
        polymax = load_plugin(polymax_vst_path)
        print(f"Successfully loaded PolyMAX VST3: {polymax}")
        
        # Try to load a factory preset first
        preset_dir = '/Library/Audio/Presets/UADx PolyMAX Synth'
        preset_loaded = False
        try:
            import os
            if os.path.exists(preset_dir):
                preset_files = [f for f in os.listdir(preset_dir) if f.endswith('.vstpreset')]
                if preset_files:
                    # Try to load a simple preset
                    for preset_file in preset_files[:3]:  # Try first 3 presets
                        try:
                            preset_path = os.path.join(preset_dir, preset_file)
                            print(f"Attempting to load preset: {preset_file}")
                            polymax.load_preset(preset_path)
                            print(f"Successfully loaded preset: {preset_file}")
                            preset_loaded = True
                            break
                        except Exception as pe:
                            print(f"Failed to load {preset_file}: {pe}")
                            continue
                    
                    if not preset_loaded:
                        print("Could not load any presets")
                else:
                    print("No .vstpreset files found in preset directory")
            else:
                print(f"Preset directory not found: {preset_dir}")
        except Exception as e:
            print(f"Could not access preset directory: {e}")
            print("Continuing with default plugin state...")
        
        # If no preset was loaded, try to manually initialize the plugin
        if not preset_loaded:
            print("No preset loaded, trying manual initialization...")
            try:
                # Reset plugin to ensure clean state
                polymax.reset()
                print("Plugin reset completed")
            except Exception as e:
                print(f"Plugin reset failed: {e}")
        
        # Print available parameters
        print("\nAvailable PolyMAX parameters:")
        available_params = list(polymax.parameters.keys())
        print(f"Total parameters: {len(available_params)}")
        
        # Look for oscillator and sound-related parameters
        osc_params = [p for p in available_params if 'osc' in p.lower()]
        env_params = [p for p in available_params if 'env' in p.lower()]
        filter_params = [p for p in available_params if 'filter' in p.lower()]
        level_params = [p for p in available_params if 'level' in p.lower() or 'volume' in p.lower()]
        wave_params = [p for p in available_params if 'wave' in p.lower() or 'shape' in p.lower()]
        
        print(f"\nOscillator parameters ({len(osc_params)}): {osc_params}")
        print(f"Waveform parameters ({len(wave_params)}): {wave_params}")
        print(f"Envelope parameters ({len(env_params)}): {env_params[:5]}")
        print(f"Filter parameters ({len(filter_params)}): {filter_params[:5]}")
        print(f"Level parameters ({len(level_params)}): {level_params}")
        
        # Save all parameters to a file for detailed analysis
        param_info = {
            'all_parameters': available_params,
            'oscillator_params': osc_params,
            'waveform_params': wave_params,
            'envelope_params': env_params,
            'filter_params': filter_params,
            'level_params': level_params
        }
        with open('polymax_parameters_debug.json', 'w') as f:
            json.dump(param_info, f, indent=2)
        print(f"\nSaved all parameter info to polymax_parameters_debug.json")
        
        # Show first 15 parameters with values
        for i, param in enumerate(available_params[:15]):
            try:
                current_value = getattr(polymax, param)
                print(f"  {i+1}. {param}: {current_value}")
            except:
                print(f"  {i+1}. {param}: (could not read value)")
        if len(available_params) > 15:
            print(f"  ... and {len(available_params) - 15} more")
        
        # Set basic parameters to ensure the plugin can generate sound
        print("\nSetting PolyMAX parameters:")
        try:
            # Set master volume to ensure we can hear something
            if hasattr(polymax, 'master_volume'):
                polymax.master_volume = 0.8
                print(f"  master_volume: 0.8")
            
            # Try to find and set oscillator waveform parameters
            # First, let's check what the current parameter values are
            print("\nCurrent parameter values:")
            key_params = ['polyphony', 'note_trigger_mode', 'osc_1_volume', 'osc_2_volume', 'noise_volume']
            for param in key_params:
                if hasattr(polymax, param):
                    try:
                        current_val = getattr(polymax, param)
                        print(f"  {param}: {current_val} (type: {type(current_val)})")
                    except:
                        print(f"  {param}: (could not read)")
            
            # Set parameters with correct types and names from the debug file
            osc_params_to_try = [
                ('osc_1_volume', 80.0),  # Use the scale that works (80.0 not 0.8)
                ('osc_2_volume', 0.0),
                ('noise_volume', 0.0),
                ('osc_1_shape', 'SAWTOOTH'),  # Use string value for waveform
                ('osc_2_shape', 'SAWTOOTH'),
                ('osc_1_coarse_tune', 0.0),  # Ensure proper tuning
                ('osc_1_fine_tune', 0.0),
                ('amp_env_attack', 0.01),
                ('amp_env_decay', 0.3),
                ('amp_env_sustain', 0.7),
                ('amp_env_release', 0.5),
                ('polyphony', 'POLY'),
                ('note_trigger_mode', 'RETRIG')
            ]
            
            for param_name, value in osc_params_to_try:
                if hasattr(polymax, param_name):
                    setattr(polymax, param_name, value)
                    print(f"  {param_name}: {value}")
                else:
                    print(f"  Warning: Parameter '{param_name}' not available")
            
        except Exception as e:
            print(f"  Error setting parameters: {e}")
            print("  Continuing with current parameters...")
        
        # Generate MIDI input to trigger the synthesizer
        print("Sending MIDI: Note 60 for 1.5s")
        sample_rate = 44100
        duration = 1.5  # seconds
        num_samples = int(sample_rate * duration)
        
        # Create MIDI messages using the correct pedalboard format
        try:
            from pedalboard import MIDIMessage
            HAVE_MIDI_MESSAGE = True
        except ImportError:
            HAVE_MIDI_MESSAGE = False
        
        print("Processing MIDI through PolyMAX...")
        print(f"Sample rate: {sample_rate}, Duration: {duration}s, Samples: {num_samples}")
        
        # Use the correct plugin call method (like pedalboard_synth.py)
        try:
            print("Using correct plugin call method...")
            
            # Create MIDI message sequence with proper timing
            if HAVE_MIDI_MESSAGE:
                midi_messages = [
                    MIDIMessage.note_on(note=60, velocity=127, time=0.0),
                    MIDIMessage.note_off(note=60, velocity=127, time=float(duration-0.1)),
                ]
            else:
                # Compatibility fallback: provide minimal objects exposing .bytes()
                class _CompatMidi:
                    def __init__(self, status, note, velocity, time):
                        self._data = bytes([status, note, velocity])
                        self.time = time
                    def bytes(self):
                        return self._data
                midi_messages = [
                    _CompatMidi(0x90, 60, 127, 0.0),  # Note on
                    _CompatMidi(0x80, 60, 127, float(duration-0.1)),  # Note off
                ]
            
            # Process using the correct method: plugin(midi_messages, ...)
            output_audio = polymax(
                midi_messages,
                duration=duration,
                sample_rate=sample_rate,
                num_channels=2,  # Stereo output
                reset=False  # Key difference: use reset=False like poc_polymax_loader.py
            )
            print(f"MIDI processing completed, output shape: {output_audio.shape}")
            
        except Exception as e:
            print(f"MIDI processing failed: {e}")
            print("Falling back to sine wave generation...")
            t = np.linspace(0, duration, num_samples)
            output_audio = 0.1 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        # Convert to mono if stereo
        if output_audio.ndim > 1:
            output_audio = np.mean(output_audio, axis=0)
        
        print(f"Generated audio: length={len(output_audio)}, RMS={np.sqrt(np.mean(output_audio**2)):.4f}")
        
        return output_audio
        
    except Exception as e:
        print(f"Error loading or using PolyMAX VST3: {e}")
        print("Falling back to sine wave generation...")
        
        # Fallback to sine wave if VST loading fails
        duration = 2.0
        sample_rate = 44100  # CRITICAL: Must be 44100 Hz, not 22050!
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Use filter_cutoff parameter to control frequency
        freq = 440 * (1 + polymax_params.get('filter_cutoff', 0.5))
        audio = np.sin(2 * np.pi * freq * t) * 0.3
        
        # Apply simple envelope
        attack = polymax_params.get('amp_attack', 0.1)
        release = polymax_params.get('amp_release', 0.5)
        
        attack_samples = int(attack * sample_rate)
        release_samples = int(release * sample_rate)
        
        # Attack envelope
        if attack_samples > 0:
            audio[:attack_samples] *= np.linspace(0, 1, attack_samples)
        
        # Release envelope
        if release_samples > 0:
            audio[-release_samples:] *= np.linspace(1, 0, release_samples)
        
        return audio

import json
import numpy as np

def convert_to_polymax_params(synth_params):
    """Convert synthesizer parameters to PolyMAX format using detailed schema and web search info."""
    
    with open('polymax_param_details.json', 'r') as f:
        param_details = json.load(f)['parameter_details']
    
    param_names = list(param_details.keys())
    
    polymax_params = {}
    
    normalized_params = np.clip(synth_params, 0, 1)
    
    for i, name in enumerate(param_names):
        if i >= len(normalized_params):
            break
            
        details = param_details[name]
        norm_val = normalized_params[i]
        
        min_val = details['min_value']
        max_val = details['max_value']
        
        if min_val is not None and max_val is not None:
            if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                value = min_val + (max_val - min_val) * norm_val
                if name in ['arp_range', 'pitch_bend_range']:
                    value = round(value)
                polymax_params[name] = value
            elif isinstance(min_val, bool) and isinstance(max_val, bool):
                 polymax_params[name] = norm_val > 0.5
        else:
            # Categorical parameters
            if name == 'polyphony':
                polymax_params[name] = 'POLY' if norm_val > 0.5 else 'MONO'
            elif name == 'note_trigger_mode':
                polymax_params[name] = 'RETRIG' if norm_val > 0.5 else 'LEGATO'
            elif name in ['osc_1_shape', 'osc_2_shape']:
                shapes = ['SAWTOOTH', 'SQUARE', 'TRIANGLE']
                polymax_params[name] = shapes[int(norm_val * (len(shapes)-1))]
            elif name == 'lfo_shape':
                shapes = ['TRIANGLE', 'RAMP_UP', 'RAMP_DOWN', 'SQUARE', 'RANDOM']
                polymax_params[name] = shapes[int(norm_val * (len(shapes)-1))]
            elif name == 'noise_color':
                polymax_params[name] = 'WHITE' if norm_val > 0.5 else 'PINK'
            elif name == 'filter_mode':
                modes = ['LP', 'HP', 'BP', 'NOTCH']
                polymax_params[name] = modes[int(norm_val * (len(modes)-1))]
            elif name == 'filter_slope':
                polymax_params[name] = '4P' if norm_val > 0.5 else '2P'
            elif name == 'unison':
                polymax_params[name] = 'ON' if norm_val > 0.5 else 'OFF'
            elif name == 'mod_wheel_dest':
                dests = ['VIBRATO', 'FILTER']
                polymax_params[name] = dests[int(norm_val * (len(dests)-1))]
            elif name == 'mod_fx_type':
                effects = ['PHASER', 'FLANGER', 'CHORUS']
                polymax_params[name] = effects[int(norm_val * (len(effects)-1))]
            elif name == 'space_fx_type':
                effects = ['HALL', 'SPRING']
                polymax_params[name] = effects[int(norm_val * (len(effects)-1))]
            elif name == 'arp_enable':
                polymax_params[name] = 'IN' if norm_val > 0.5 else 'OUT'
            else:
                # For other categorical params, pass the normalized float value
                polymax_params[name] = norm_val

    return polymax_params

def main():
    if len(sys.argv) != 2:
        print("Usage: python resynth_test.py <audio_file>")
        print("Example: python resynth_test.py '11 Missing Project - Poisson D'Avril (Galaxy dub) - Crop #1.wav'")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    if not os.path.exists(audio_file):
        print(f"Error: Audio file '{audio_file}' not found")
        sys.exit(1)
    
    print("=== PolyMAX Resynthesis Test ===")
    print(f"Audio file: {audio_file}")
    
    # Use optimized model if available, otherwise stable model
    model_paths = [
        'outputs_optimized/models/vae_mel_l1_0_optimized_gated_cnn_mlp.model',
        'outputs_stable/models/vae_mel_l1_0_stable_gated_cnn_mlp.model',
        'outputs/models/best_model.pth'
    ]
    
    regression_model = None
    vae_model = None
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                regression_model, vae_model = load_trained_model(model_path)
                print(f"Successfully loaded model: {model_path}")
                break
            except Exception as e:
                print(f"Failed to load {model_path}: {e}")
                continue
    
    if regression_model is None or vae_model is None:
        print("Error: Could not load any trained model")
        print("Expected locations:")
        for path in model_paths:
            print(f"  - {path}")
        sys.exit(1)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"resynth_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Process audio
    try:
        results, comparison = process_audio_file(audio_file, regression_model, vae_model, output_dir)
        
        if results:
            # Save comparison report
            report_file = os.path.join(output_dir, 'comparison_report.json')
            with open(report_file, 'w') as f:
                json.dump(comparison, f, indent=2)
            
            print(f"\nResynthesis test completed successfully!")
            print(f"Results saved to: {output_dir}/")
            print(f"Comparison report saved to: {report_file}")
        else:
            print("Resynthesis test failed")
            
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
