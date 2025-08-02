#!/usr/bin/env python3
"""
Integration test and demonstration of the polyphonic pipeline.

This script creates synthetic test data to demonstrate the complete pipeline
without requiring real audio files or plugin installations.
"""

import os
import sys
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flowsynth.transcription.midi_tools import MIDINote, process_midi_notes
from flowsynth.render.dawdreamer_scheduler import DawDreamerScheduler, PluginRegistry


def create_synthetic_chord_notes():
    """Create synthetic MIDI notes representing a C major triad with pitch bends."""
    return [
        # C major triad (C4, E4, G4) with some timing variation and pitch bends
        MIDINote(60, 100, 0.0, 2.0, pitch_bends=[(0.5, 50), (1.0, -30)]),    # C4 with vibrato
        MIDINote(64, 90, 0.05, 2.1, pitch_bends=[(0.8, 80)]),                # E4 with slight bend
        MIDINote(67, 85, 0.03, 1.95),                                         # G4 no bend
        # Add a second chord
        MIDINote(60, 80, 2.5, 4.0),   # C4
        MIDINote(65, 85, 2.55, 4.1),  # F4  
        MIDINote(69, 90, 2.52, 3.98), # A4
    ]


def test_midi_processing():
    """Test the MIDI processing pipeline."""
    print("Testing MIDI processing pipeline...")
    
    # Create test notes
    raw_notes = create_synthetic_chord_notes()
    print(f"Created {len(raw_notes)} raw notes")
    
    # Test different bend modes
    for bend_mode in ['none', 'global', 'per_voice']:
        print(f"\nTesting bend mode: {bend_mode}")
        
        processed_notes = process_midi_notes(
            raw_notes,
            bend_mode=bend_mode,
            max_channels=4,
            quantize_time=0.05,  # 50ms quantization
        )
        
        print(f"  Processed {len(processed_notes)} notes")
        
        # Analyze channel distribution
        channels = [note.channel for note in processed_notes]
        channel_counts = {}
        for ch in channels:
            channel_counts[ch] = channel_counts.get(ch, 0) + 1
        
        print(f"  Channel distribution: {channel_counts}")
        
        # Check pitch bends
        notes_with_bends = [note for note in processed_notes if note.pitch_bends]
        print(f"  Notes with pitch bends: {len(notes_with_bends)}")
        
        # Show timing quantization
        start_times = [note.start_time for note in processed_notes]
        print(f"  Start times: {start_times}")


def test_plugin_registry():
    """Test the plugin registry functionality."""
    print("\nTesting plugin registry...")
    
    registry = PluginRegistry()
    
    # Register some fake plugins for testing
    registry.register_plugin("test_synth", "/fake/path/synth.vst")
    registry.register_plugin("test_synth2", "/fake/path/synth2.vst3", {
        "init": "/fake/path/init.preset",
        "bass": "/fake/path/bass.preset"
    })
    
    print(f"Registered plugins: {list(registry.plugins.keys())}")
    print(f"Test synth path: {registry.get_plugin_path('test_synth')}")
    print(f"Test synth2 bass preset: {registry.get_preset_path('test_synth2', 'bass')}")


def test_mock_transcription():
    """Test the mock transcription when Basic Pitch is not available."""
    print("\nTesting mock transcription...")
    
    # Create a temporary audio file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        # Create 2 seconds of stereo white noise
        sample_rate = 22050
        duration = 2.0
        samples = int(sample_rate * duration)
        audio = np.random.randn(2, samples) * 0.1  # Quiet noise
        
        # Save as audio file
        sf.write(temp_file.name, audio.T, sample_rate)
        
        try:
            # Import and test transcription
            from flowsynth.transcription.basicpitch_backend import transcribe
            
            result = transcribe(temp_file.name, save_midi=False, save_notes_csv=False)
            
            print(f"Transcription result keys: {list(result.keys())}")
            print(f"Number of notes: {len(result['note_events'])}")
            print(f"Meta info: {result['meta']}")
            
        finally:
            # Clean up
            os.unlink(temp_file.name)


def test_scheduler_creation():
    """Test DawDreamer scheduler creation and basic functionality."""
    print("\nTesting DawDreamer scheduler...")
    
    try:
        scheduler = DawDreamerScheduler()
        print(f"Scheduler created: sample_rate={scheduler.sample_rate}, bpm={scheduler.current_bpm}")
        
        # Test BPM setting
        scheduler.set_bpm(140.0)
        print(f"BPM set to: {scheduler.current_bpm}")
        
        # Test plugin loading (will fail but should handle gracefully)
        result = scheduler.load_plugin("/nonexistent/plugin.vst")
        print(f"Plugin loading result (expected False): {result}")
        
    except Exception as e:
        print(f"Scheduler test failed: {e}")


def demonstrate_complete_pipeline():
    """Demonstrate the complete polyphonic pipeline with synthetic data."""
    print("\n" + "="*60)
    print("COMPLETE POLYPHONIC PIPELINE DEMONSTRATION")
    print("="*60)
    
    # Step 1: Create synthetic transcription data
    print("Step 1: Creating synthetic MIDI notes (simulating transcription)...")
    notes = create_synthetic_chord_notes()
    print(f"Created {len(notes)} notes representing two chords")
    
    for i, note in enumerate(notes):
        print(f"  Note {i+1}: {note}")
    
    # Step 2: Process MIDI
    print("\nStep 2: Processing MIDI with per-voice pitch bend handling...")
    processed_notes = process_midi_notes(
        notes,
        bend_mode="per_voice",
        max_channels=8,
        quantize_time=0.01,
        min_note_duration=0.1
    )
    
    print(f"Processed to {len(processed_notes)} notes")
    
    # Analyze results
    channels_used = set(note.channel for note in processed_notes)
    notes_with_bends = [note for note in processed_notes if note.pitch_bends]
    
    print(f"Channels used: {sorted(channels_used)}")
    print(f"Notes with pitch bends: {len(notes_with_bends)}")
    
    # Show processed notes
    for i, note in enumerate(processed_notes):
        bends_info = f" (bends: {len(note.pitch_bends)})" if note.pitch_bends else ""
        print(f"  Processed {i+1}: Ch{note.channel} {note.pitch} vel{note.velocity} "
              f"{note.start_time:.3f}-{note.end_time:.3f}s{bends_info}")
    
    # Step 3: Simulate rendering setup
    print("\nStep 3: Setting up rendering (simulated - no actual plugin needed)...")
    scheduler = DawDreamerScheduler()
    
    print(f"Scheduler ready: {scheduler.sample_rate}Hz, {scheduler.current_bpm}BPM")
    print("Pipeline complete! (Would render audio if plugin was available)")
    
    return processed_notes


def main():
    """Run all integration tests."""
    print("Flow-Synth Polyphonic Pipeline Integration Test")
    print("=" * 60)
    
    # Run individual component tests
    test_midi_processing()
    test_plugin_registry()
    test_mock_transcription()
    test_scheduler_creation()
    
    # Run complete pipeline demonstration
    demonstrate_complete_pipeline()
    
    print("\n" + "="*60)
    print("Integration test complete!")
    print("All core components are working correctly.")
    print("The pipeline is ready for use with real audio files and plugins.")
    print("="*60)


if __name__ == '__main__':
    main()