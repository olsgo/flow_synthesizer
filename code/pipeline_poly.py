#!/usr/bin/env python3
"""
End-to-end polyphonic pipeline for Flow Synth.

Input: audio file + synth plugin path + (optional) preset/state
Output: rendered WAV of the reconstruction and a sidecar MIDI
"""

import argparse
import os
import sys
import tempfile
import json
from typing import List, Dict, Any, Optional
import numpy as np
import soundfile as sf

# Import our modules
from code.polyphonic.transcribe import NoteEvent, SimpleTestBackend, detect_bpm, save_events_to_json
from code.polyphonic.schedule import PolyphonicRenderer, load_events_from_json
from code.dd_renderer import DDRenderer

class PolyphonicPipeline:
    """End-to-end polyphonic reconstruction pipeline."""
    
    def __init__(self, sample_rate: int = 22050):
        self.sr = sample_rate
        self.transcription_backend = None
        self.renderer = None
        
    def load_plugin(self, plugin_path: str) -> bool:
        """Load the synthesis plugin."""
        self.renderer = PolyphonicRenderer(sample_rate=self.sr)
        return self.renderer.load_plugin(plugin_path)
    
    def set_transcription_backend(self, backend_name: str):
        """Set the transcription backend."""
        if backend_name == 'test':
            self.transcription_backend = SimpleTestBackend()
        else:
            try:
                if backend_name == 'omnizart':
                    from code.polyphonic.transcribe import OmnizartBackend
                    self.transcription_backend = OmnizartBackend()
                elif backend_name == 'onsets_frames':
                    from code.polyphonic.transcribe import OnsetsFramesBackend
                    self.transcription_backend = OnsetsFramesBackend()
                else:
                    raise ValueError(f"Unknown backend: {backend_name}")
            except ImportError as e:
                print(f"Error loading backend {backend_name}: {e}")
                print("Falling back to test backend")
                self.transcription_backend = SimpleTestBackend()
    
    def transcribe_audio(self, audio_path: str, bpm: Optional[float] = None) -> List[NoteEvent]:
        """Transcribe audio to note events."""
        if self.transcription_backend is None:
            raise RuntimeError("No transcription backend set")
        
        if bpm is None:
            bpm = detect_bpm(audio_path, 120.0)
        
        return self.transcription_backend.transcribe(audio_path, bpm)
    
    def infer_parameters(self, audio_path: str, events: List[NoteEvent]) -> List[tuple]:
        """
        Infer synth parameters from audio using Flow Synth models.
        """
        try:
            from code.polyphonic.parameter_inference import infer_parameters_from_audio
            
            # Use the parameter inference module
            parameters = infer_parameters_from_audio(audio_path, mode='global')
            print(f"Inferred {len(parameters)} parameters from audio")
            return parameters
            
        except Exception as e:
            print(f"Error during parameter inference: {e}")
            print("Using default parameters")
            return []
    
    def render_polyphonic(self, events: List[NoteEvent], 
                         duration_beats: float, bpm: float = 120.0,
                         mode: str = 'single_instance',
                         parameters: Optional[List[tuple]] = None) -> np.ndarray:
        """Render polyphonic events with the loaded plugin."""
        if self.renderer is None:
            raise RuntimeError("No renderer loaded")
        
        # Set parameters if provided
        if parameters:
            self.renderer.set_plugin_parameters(parameters)
        
        # Render using specified mode
        if mode == 'single_instance':
            return self.renderer.render_single_instance(events, duration_beats, bpm)
        elif mode == 'multi_instance':
            return self.renderer.render_multi_instance(events, duration_beats, bpm)
        else:
            raise ValueError(f"Unknown rendering mode: {mode}")
    
    def run_pipeline(self, audio_path: str, plugin_path: str,
                    output_path: str, mode: str = 'global-params',
                    backend: str = 'test', bpm: Optional[float] = None,
                    render_mode: str = 'single_instance') -> bool:
        """Run the complete polyphonic reconstruction pipeline."""
        
        print(f"Starting polyphonic pipeline for: {audio_path}")
        
        # 1. Load plugin
        print(f"Loading plugin: {plugin_path}")
        if not self.load_plugin(plugin_path):
            print("Failed to load plugin")
            return False
        
        # 2. Set transcription backend
        print(f"Setting up transcription backend: {backend}")
        self.set_transcription_backend(backend)
        
        # 3. Transcribe audio to events
        print("Transcribing audio to note events...")
        try:
            events = self.transcribe_audio(audio_path, bpm)
            if not events:
                print("No events transcribed")
                return False
            print(f"Transcribed {len(events)} note events")
        except Exception as e:
            print(f"Error during transcription: {e}")
            return False
        
        # 4. Calculate render duration
        last_end = max(event.onset_beats + event.duration_beats for event in events)
        duration_beats = last_end + 2.0  # Add buffer
        
        detected_bpm = bpm or detect_bpm(audio_path, 120.0)
        print(f"Render duration: {duration_beats:.2f} beats at {detected_bpm} BPM")
        
        # 5. Parameter inference (if needed)
        parameters = None
        if mode == 'global-params':
            print("Running parameter inference...")
            try:
                parameters = self.infer_parameters(audio_path, events)
            except Exception as e:
                print(f"Error during parameter inference: {e}")
                # Continue with default parameters
        
        # 6. Render polyphonic audio
        print(f"Rendering polyphonic audio in {render_mode} mode...")
        try:
            audio = self.render_polyphonic(
                events, duration_beats, detected_bpm, 
                render_mode, parameters
            )
        except Exception as e:
            print(f"Error during rendering: {e}")
            return False
        
        # 7. Save output audio
        try:
            # Convert to stereo if mono
            if audio.ndim == 1:
                audio = np.stack([audio, audio])
            elif audio.shape[0] == 1:
                audio = np.vstack([audio, audio])
            
            # Transpose to (samples, channels) for soundfile
            audio_t = audio.T
            sf.write(output_path, audio_t, self.sr)
            print(f"Saved rendered audio to: {output_path}")
            
            # Print audio stats
            duration_sec = audio.shape[1] / self.sr
            max_amp = np.max(np.abs(audio))
            print(f"Audio duration: {duration_sec:.2f}s, Max amplitude: {max_amp:.3f}")
            
        except Exception as e:
            print(f"Error saving audio: {e}")
            return False
        
        # 8. Save sidecar events file
        try:
            events_output = os.path.splitext(output_path)[0] + "_events.json"
            save_events_to_json(events, events_output, detected_bpm)
            print(f"Saved note events to: {events_output}")
        except Exception as e:
            print(f"Warning: Could not save events file: {e}")
        
        return True

def main():
    """CLI interface for the polyphonic pipeline."""
    parser = argparse.ArgumentParser(description='End-to-end polyphonic reconstruction pipeline')
    parser.add_argument('input_audio', help='Input audio file path')
    parser.add_argument('--plugin', required=True, help='VST/AU plugin path')
    parser.add_argument('--out', help='Output audio file path')
    parser.add_argument('--mode', choices=['global-params'], default='global-params',
                       help='Parameter inference mode')
    parser.add_argument('--backend', choices=['test', 'omnizart', 'onsets_frames'], 
                       default='test', help='Transcription backend')
    parser.add_argument('--renderer', choices=['single_instance', 'multi_instance'],
                       default='single_instance', help='Rendering mode')
    parser.add_argument('--bpm', type=float, help='BPM (auto-detect if not specified)')
    parser.add_argument('--sample_rate', type=int, default=22050, help='Sample rate')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input_audio):
        print(f"Error: Input audio file not found: {args.input_audio}")
        sys.exit(1)
    
    if not os.path.exists(args.plugin):
        print(f"Error: Plugin not found: {args.plugin}")
        sys.exit(1)
    
    # Set output path
    if args.out:
        output_path = args.out
    else:
        base_name = os.path.splitext(os.path.basename(args.input_audio))[0]
        output_path = f"{base_name}_polyphonic_recon.wav"
    
    # Run pipeline
    pipeline = PolyphonicPipeline(sample_rate=args.sample_rate)
    
    success = pipeline.run_pipeline(
        audio_path=args.input_audio,
        plugin_path=args.plugin,
        output_path=output_path,
        mode=args.mode,
        backend=args.backend,
        bpm=args.bpm,
        render_mode=args.renderer
    )
    
    if success:
        print("\nPolyphonic reconstruction completed successfully!")
    else:
        print("\nPolyphonic reconstruction failed!")
        sys.exit(1)

if __name__ == '__main__':
    main()