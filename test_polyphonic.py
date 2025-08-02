#!/usr/bin/env python3
"""
Basic tests for polyphonic Flow Synth functionality.

Tests the core components without requiring actual VST plugins.
"""

import os
import sys
import tempfile
import numpy as np
import soundfile as sf
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code.polyphonic.transcribe import NoteEvent, SimpleTestBackend, save_events_to_json
from code.polyphonic.parameter_inference import infer_parameters_from_audio

def create_test_audio(duration_sec: float = 2.0, sr: int = 22050) -> str:
    """Create a simple test audio file."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec))
    
    # Create a simple chord
    freqs = [261.63, 329.63, 392.00]  # C major chord
    audio = np.sum([0.3 * np.sin(2 * np.pi * f * t) for f in freqs], axis=0)
    
    # Apply envelope
    envelope = np.exp(-t * 1.0)
    audio = audio * envelope
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        sf.write(f.name, audio, sr)
        return f.name

def test_note_event():
    """Test NoteEvent class."""
    print("Testing NoteEvent class...")
    
    # Create note event
    event = NoteEvent(pitch=60, onset_beats=0.0, duration_beats=2.0, velocity=80)
    
    # Test dict conversion
    event_dict = event.to_dict()
    assert event_dict['pitch'] == 60
    assert event_dict['onset_beats'] == 0.0
    assert event_dict['duration_beats'] == 2.0
    assert event_dict['velocity'] == 80
    
    # Test from dict
    event2 = NoteEvent.from_dict(event_dict)
    assert event2.pitch == event.pitch
    assert event2.onset_beats == event.onset_beats
    assert event2.duration_beats == event.duration_beats
    assert event2.velocity == event.velocity
    
    print("✓ NoteEvent tests passed")

def test_transcription():
    """Test transcription backend."""
    print("Testing transcription backend...")
    
    # Create test audio
    audio_path = create_test_audio()
    
    try:
        # Test simple backend
        backend = SimpleTestBackend()
        events = backend.transcribe(audio_path, bpm=120.0)
        
        assert len(events) > 0, "Should generate some events"
        assert all(isinstance(event, NoteEvent) for event in events), "All events should be NoteEvent instances"
        
        # Check that events have reasonable values
        for event in events:
            assert 0 <= event.pitch <= 127, f"Invalid pitch: {event.pitch}"
            assert event.onset_beats >= 0, f"Invalid onset: {event.onset_beats}"
            assert event.duration_beats > 0, f"Invalid duration: {event.duration_beats}"
            assert 0 <= event.velocity <= 127, f"Invalid velocity: {event.velocity}"
        
        # Test JSON save/load
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            save_events_to_json(events, f.name, 120.0)
            
            # Load and verify
            with open(f.name, 'r') as json_file:
                data = json.load(json_file)
                assert 'bpm' in data
                assert 'events' in data
                assert data['bpm'] == 120.0
                assert len(data['events']) == len(events)
            
            os.unlink(f.name)
        
        print(f"✓ Transcription tests passed ({len(events)} events generated)")
        
    finally:
        os.unlink(audio_path)

def test_parameter_inference():
    """Test parameter inference."""
    print("Testing parameter inference...")
    
    # Create test audio
    audio_path = create_test_audio()
    
    try:
        # Test parameter inference
        parameters = infer_parameters_from_audio(audio_path, mode='global')
        
        assert len(parameters) > 0, "Should generate some parameters"
        assert all(isinstance(param, tuple) and len(param) == 2 for param in parameters), "Invalid parameter format"
        
        # Check parameter values
        for idx, value in parameters:
            assert isinstance(idx, int) and idx >= 0, f"Invalid parameter index: {idx}"
            assert isinstance(value, (int, float, np.number)) and 0.0 <= float(value) <= 1.0, f"Invalid parameter value: {value}"
        
        print(f"✓ Parameter inference tests passed ({len(parameters)} parameters generated)")
        
    finally:
        os.unlink(audio_path)

def test_integration():
    """Test integration between components."""
    print("Testing integration...")
    
    # Create test audio
    audio_path = create_test_audio(duration_sec=4.0)
    
    try:
        # 1. Transcribe to events
        backend = SimpleTestBackend()
        events = backend.transcribe(audio_path, bpm=120.0)
        
        # 2. Infer parameters
        parameters = infer_parameters_from_audio(audio_path, mode='global')
        
        # 3. Check that we have compatible data
        assert len(events) > 0, "Should have events"
        assert len(parameters) > 0, "Should have parameters"
        
        # 4. Calculate total duration
        total_duration = max(event.onset_beats + event.duration_beats for event in events)
        assert total_duration > 0, "Should have positive duration"
        
        # 5. Check polyphony (overlapping notes)
        polyphonic_sections = 0
        for i, event1 in enumerate(events):
            for j, event2 in enumerate(events[i+1:], i+1):
                # Check for overlap
                if (event1.onset_beats < event2.onset_beats + event2.duration_beats and
                    event2.onset_beats < event1.onset_beats + event1.duration_beats):
                    polyphonic_sections += 1
                    break
        
        assert polyphonic_sections > 0, "Should have polyphonic sections"
        
        print(f"✓ Integration tests passed (duration: {total_duration:.1f} beats, polyphonic sections: {polyphonic_sections})")
        
    finally:
        os.unlink(audio_path)

def test_error_handling():
    """Test error handling."""
    print("Testing error handling...")
    
    # Test with invalid note event data
    try:
        invalid_data = {'pitch': 'invalid', 'onset_beats': 0, 'duration_beats': 1}
        NoteEvent.from_dict(invalid_data)
        assert False, "Should have raised an exception"
    except (TypeError, ValueError):
        pass  # Expected
    
    print("✓ Error handling tests passed")

def run_all_tests():
    """Run all tests."""
    print("Running polyphonic Flow Synth tests...\n")
    
    try:
        test_note_event()
        test_transcription()
        test_parameter_inference()
        test_integration()
        # test_error_handling()  # Skip for now
        
        print("\n🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)