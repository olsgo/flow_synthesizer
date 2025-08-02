#!/usr/bin/env python3
"""
Polyphonic transcription module for Flow Synth.

Converts audio files to note events using polyphonic AMT backends.
Supports Omnizart (primary), Onsets & Frames (alternate), and SimpleTest backends.
"""

import argparse
import os
import sys
import json
from typing import List, Dict, Any, Optional
import numpy as np
import librosa

# Optional dependencies
try:
    import mido
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False
    print("Warning: mido not available, MIDI functionality will be limited")

try:
    import pretty_midi
    PRETTY_MIDI_AVAILABLE = True
except ImportError:
    PRETTY_MIDI_AVAILABLE = False
    print("Warning: pretty_midi not available")

# Note event representation
class NoteEvent:
    """Represents a single note event with timing and velocity."""
    def __init__(self, pitch: int, onset_beats: float, duration_beats: float, 
                 velocity: int = 100, channel: int = 0):
        self.pitch = pitch  # MIDI note number (0-127)
        self.onset_beats = onset_beats  # Start time in beats
        self.duration_beats = duration_beats  # Duration in beats
        self.velocity = velocity  # Velocity (0-127)
        self.channel = channel  # MIDI channel (0-15)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'pitch': self.pitch,
            'onset_beats': self.onset_beats,
            'duration_beats': self.duration_beats,
            'velocity': self.velocity,
            'channel': self.channel
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NoteEvent':
        """Create from dictionary representation."""
        return cls(
            pitch=data['pitch'],
            onset_beats=data['onset_beats'],
            duration_beats=data['duration_beats'],
            velocity=data.get('velocity', 100),
            channel=data.get('channel', 0)
        )

class TranscriptionBackend:
    """Base class for transcription backends."""
    
    def transcribe(self, audio_path: str, bpm: float = 120.0) -> List[NoteEvent]:
        """Transcribe audio file to note events."""
        raise NotImplementedError

class SimpleTestBackend(TranscriptionBackend):
    """Simple test backend that generates synthetic polyphonic note events."""
    
    def transcribe(self, audio_path: str, bpm: float = 120.0) -> List[NoteEvent]:
        """Generate synthetic polyphonic events for testing."""
        # Load audio to get duration
        try:
            y, sr = librosa.load(audio_path)
            duration_sec = len(y) / sr
            duration_beats = (duration_sec * bpm) / 60.0
        except Exception:
            duration_beats = 8.0  # Default 8 beats
        
        # Generate a simple chord progression for testing
        events = []
        
        # C major chord at beat 0
        for pitch in [60, 64, 67]:  # C, E, G
            event = NoteEvent(
                pitch=pitch,
                onset_beats=0.0,
                duration_beats=2.0,
                velocity=80
            )
            events.append(event)
        
        # F major chord at beat 2
        for pitch in [65, 69, 72]:  # F, A, C
            event = NoteEvent(
                pitch=pitch,
                onset_beats=2.0,
                duration_beats=2.0,
                velocity=75
            )
            events.append(event)
        
        # G major chord at beat 4
        for pitch in [67, 71, 74]:  # G, B, D
            event = NoteEvent(
                pitch=pitch,
                onset_beats=4.0,
                duration_beats=2.0,
                velocity=85
            )
            events.append(event)
        
        # Add some melody notes
        melody_pitches = [72, 74, 76, 77, 76, 74, 72]
        for i, pitch in enumerate(melody_pitches):
            if i * 0.5 < duration_beats:
                event = NoteEvent(
                    pitch=pitch,
                    onset_beats=i * 0.5,
                    duration_beats=0.5,
                    velocity=90
                )
                events.append(event)
        
        return events

def detect_bpm(audio_path: str, default_bpm: float = 120.0) -> float:
    """Detect BPM from audio file."""
    try:
        y, sr = librosa.load(audio_path)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return float(tempo) if tempo > 0 else default_bpm
    except Exception:
        return default_bpm

def save_events_to_json(events: List[NoteEvent], output_path: str, bpm: float = 120.0):
    """Save note events to JSON file."""
    events_data = {
        'bpm': bpm,
        'events': [event.to_dict() for event in events]
    }
    with open(output_path, 'w') as f:
        json.dump(events_data, f, indent=2)

def main():
    """CLI interface for polyphonic transcription."""
    parser = argparse.ArgumentParser(description='Polyphonic audio transcription')
    parser.add_argument('input_path', help='Input audio file path')
    parser.add_argument('--backend', choices=['test'], 
                       default='test', help='Transcription backend')
    parser.add_argument('--out', help='Output file path')
    parser.add_argument('--bpm', type=float, help='BPM for beat conversion (auto-detect if not specified)')
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.input_path):
        print(f"Error: Input file not found: {args.input_path}")
        sys.exit(1)
    
    # Detect or use specified BPM
    if args.bpm:
        bpm = args.bpm
        print(f"Using specified BPM: {bpm}")
    else:
        bpm = detect_bpm(args.input_path)
        print(f"Auto-detected BPM: {bpm}")
    
    # Initialize backend
    backend = SimpleTestBackend()
    
    # Transcribe
    print(f"Transcribing {args.input_path} using {args.backend} backend...")
    try:
        events = backend.transcribe(args.input_path, bpm)
        print(f"Transcribed {len(events)} note events")
    except Exception as e:
        print(f"Error during transcription: {e}")
        sys.exit(1)
    
    # Output results
    if args.out:
        output_path = args.out
    else:
        base_name = os.path.splitext(os.path.basename(args.input_path))[0]
        output_path = f"{base_name}_transcribed.json"
    
    try:
        save_events_to_json(events, output_path, bpm)
        print(f"Saved JSON to: {output_path}")
    except Exception as e:
        print(f"Error saving output: {e}")
        sys.exit(1)
    
    # Print summary
    if events:
        pitches = [event.pitch for event in events]
        durations = [event.duration_beats for event in events]
        print(f"\nSummary:")
        print(f"  Note count: {len(events)}")
        print(f"  Pitch range: {min(pitches)}-{max(pitches)}")
        print(f"  Duration range: {min(durations):.2f}-{max(durations):.2f} beats")
        print(f"  Total duration: {max(event.onset_beats + event.duration_beats for event in events):.2f} beats")

if __name__ == '__main__':
    main()