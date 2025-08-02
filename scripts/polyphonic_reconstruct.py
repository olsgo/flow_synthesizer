#!/usr/bin/env python3
"""
Polyphonic reconstruction script for Flow-Synth.

This script provides the end-to-end polyphonic pipeline:
1. Transcribe audio to MIDI using Basic Pitch
2. Post-process MIDI (quantize, channelize, handle pitch bends)
3. Render using DawDreamer-hosted synthesizers
4. Optionally optimize synthesis parameters with Flow-Synth

Usage:
    python polyphonic_reconstruct.py \
        --audio input.wav \
        --synth diva \
        --bend_mode per_voice \
        --bpm 120 \
        --output output.wav
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flowsynth.transcription.basicpitch_backend import transcribe
from flowsynth.transcription.midi_tools import process_midi_notes
from flowsynth.render.dawdreamer_scheduler import DawDreamerScheduler, PluginRegistry, create_default_registry
import soundfile as sf
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Polyphonic audio reconstruction with Flow-Synth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic polyphonic reconstruction
  python polyphonic_reconstruct.py --audio chord.wav --synth diva --output result.wav
  
  # With pitch bends and custom settings
  python polyphonic_reconstruct.py \\
    --audio vibrato_chords.wav \\
    --synth massive_x \\
    --bend_mode per_voice \\
    --channels 8 \\
    --bpm 120 \\
    --renderer_beats 8
  
  # Notes-only mode (no pitch bends)
  python polyphonic_reconstruct.py \\
    --audio input.wav \\
    --synth diva \\
    --preset /path/to/preset.fxp \\
    --bend_mode none \\
    --output output.wav
        """
    )
    
    # Input/Output
    parser.add_argument('--audio', required=True, type=str,
                       help='Input audio file path')
    parser.add_argument('--output', type=str,
                       help='Output audio file path (default: input_reconstructed.wav)')
    
    # Synthesis
    parser.add_argument('--synth', required=True, type=str,
                       help='Synthesizer name (e.g., diva, massive_x) or plugin path')
    parser.add_argument('--preset', type=str,
                       help='Preset file path or name')
    parser.add_argument('--config', type=str,
                       help='Plugin configuration file path')
    
    # MIDI Processing
    parser.add_argument('--bend_mode', choices=['none', 'global', 'per_voice'], 
                       default='per_voice',
                       help='Pitch bend handling mode (default: per_voice)')
    parser.add_argument('--channels', type=int, default=8,
                       help='Maximum MIDI channels for per_voice mode (default: 8)')
    
    # Rendering
    parser.add_argument('--bpm', type=float, default=120.0,
                       help='BPM for beat-quantized rendering (default: 120)')
    parser.add_argument('--renderer_seconds', type=float,
                       help='Render duration in seconds (auto-detected if not specified)')
    parser.add_argument('--renderer_beats', type=float,
                       help='Render duration in beats (use instead of seconds for beat-sync)')
    
    # Transcription
    parser.add_argument('--min_freq', type=float, default=80.0,
                       help='Minimum transcription frequency in Hz (default: 80)')
    parser.add_argument('--max_freq', type=float, default=2000.0,
                       help='Maximum transcription frequency in Hz (default: 2000)')
    parser.add_argument('--onset_threshold', type=float, default=0.5,
                       help='Note onset detection threshold (default: 0.5)')
    
    # Processing
    parser.add_argument('--quantize_time', type=float, default=0.010,
                       help='Time quantization in seconds (default: 0.010)')
    parser.add_argument('--min_note_duration', type=float, default=0.050,
                       help='Minimum note duration in seconds (default: 0.050)')
    
    # Debug
    parser.add_argument('--save_midi', action='store_true',
                       help='Save intermediate MIDI file')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    parser.add_argument('--temp_dir', type=str,
                       help='Temporary directory for intermediate files')
    
    return parser


def main():
    """Main polyphonic reconstruction pipeline."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input
    audio_path = Path(args.audio)
    if not audio_path.exists():
        logger.error(f"Input audio file not found: {audio_path}")
        return 1
    
    # Setup output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = audio_path.with_name(f"{audio_path.stem}_reconstructed.wav")
    
    # Create temp directory if specified
    temp_dir = args.temp_dir
    if temp_dir:
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Transcribe audio to MIDI
        logger.info("Step 1: Transcribing audio to MIDI...")
        transcription_result = transcribe(
            audio_path=audio_path,
            save_dir=temp_dir,
            save_midi=args.save_midi,
            save_notes_csv=args.debug,
            min_freq=args.min_freq,
            max_freq=args.max_freq,
            onset_threshold=args.onset_threshold
        )
        
        notes = transcription_result['note_events']
        meta = transcription_result['meta']
        
        if not notes:
            logger.warning("No notes transcribed from audio")
            return 1
            
        logger.info(f"Transcribed {len(notes)} notes, duration: {meta['duration']:.2f}s")
        
        # Step 2: Post-process MIDI
        logger.info("Step 2: Post-processing MIDI...")
        processed_notes = process_midi_notes(
            notes=notes,
            bend_mode=args.bend_mode,
            max_channels=args.channels,
            quantize_time=args.quantize_time,
            min_note_duration=args.min_note_duration
        )
        
        logger.info(f"Processed to {len(processed_notes)} notes")
        
        # Log channel distribution for per_voice mode
        if args.bend_mode == 'per_voice':
            channel_counts = {}
            for note in processed_notes:
                channel_counts[note.channel] = channel_counts.get(note.channel, 0) + 1
            logger.info(f"Channel distribution: {dict(sorted(channel_counts.items()))}")
        
        # Step 3: Setup synthesizer
        logger.info("Step 3: Setting up synthesizer...")
        
        # Load plugin registry
        if args.config:
            registry = PluginRegistry(args.config)
        else:
            registry = create_default_registry()
            # Also try to load from default config location
            default_config = Path(__file__).parent.parent / "flowsynth" / "configs" / "synths.yaml"
            if default_config.exists():
                registry.load_config(str(default_config))
        
        # Create scheduler
        scheduler = DawDreamerScheduler(plugin_registry=registry)
        scheduler.set_bpm(args.bpm)
        
        # Load plugin
        if not scheduler.load_plugin(args.synth):
            logger.error(f"Failed to load synthesizer: {args.synth}")
            return 1
        
        # Load preset if specified
        if args.preset:
            if not scheduler.load_preset(args.preset, args.synth):
                logger.warning(f"Failed to load preset: {args.preset}")
        
        # Step 4: Render audio
        logger.info("Step 4: Rendering audio...")
        
        # Determine render duration
        if args.renderer_beats is not None:
            logger.info(f"Rendering {args.renderer_beats} beats at {args.bpm} BPM")
            audio = scheduler.render_beats(processed_notes, args.renderer_beats)
        else:
            if args.renderer_seconds is not None:
                duration_seconds = args.renderer_seconds
            else:
                # Auto-detect duration with some padding
                max_end_time = max(note.end_time for note in processed_notes) if processed_notes else 4.0
                duration_seconds = max_end_time + 1.0  # Add 1 second padding
            
            logger.info(f"Rendering {duration_seconds:.2f} seconds")
            
            # Choose rendering method based on pitch bend mode
            if args.bend_mode == 'none':
                # Notes-only rendering for best performance
                audio = scheduler.render_notes_only(processed_notes, duration_seconds)
            else:
                # Full MIDI rendering for pitch bends
                audio = scheduler.render_with_midi_file(processed_notes, duration_seconds, temp_dir)
        
        # Step 5: Save output
        logger.info("Step 5: Saving output...")
        
        # Ensure stereo output
        if audio.ndim == 1:
            audio = np.stack([audio, audio])
        elif audio.shape[0] == 1:
            audio = np.vstack([audio, audio])
        
        # Save audio file
        sf.write(str(output_path), audio.T, scheduler.sample_rate)
        logger.info(f"Output saved to: {output_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("POLYPHONIC RECONSTRUCTION SUMMARY")
        print("="*60)
        print(f"Input:           {audio_path}")
        print(f"Output:          {output_path}")
        print(f"Synthesizer:     {args.synth}")
        print(f"Notes:           {len(processed_notes)}")
        print(f"Bend mode:       {args.bend_mode}")
        print(f"Channels used:   {len(set(note.channel for note in processed_notes))}")
        print(f"Duration:        {audio.shape[1] / scheduler.sample_rate:.2f}s")
        print(f"Sample rate:     {scheduler.sample_rate} Hz")
        
        if args.save_midi and transcription_result.get('midi_path'):
            print(f"MIDI file:       {transcription_result['midi_path']}")
        
        print("="*60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Reconstruction failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())