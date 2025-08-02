"""
Tests for MIDI processing tools.
"""

import unittest
import numpy as np
from flowsynth.transcription.midi_tools import MIDINote, MIDIProcessor, PitchBendProcessor, process_midi_notes


class TestMIDINote(unittest.TestCase):
    """Test MIDINote class."""
    
    def test_creation(self):
        """Test MIDINote creation."""
        note = MIDINote(
            pitch=60,
            velocity=100,
            start_time=0.0,
            end_time=1.0,
            channel=0
        )
        
        self.assertEqual(note.pitch, 60)
        self.assertEqual(note.velocity, 100)
        self.assertEqual(note.start_time, 0.0)
        self.assertEqual(note.end_time, 1.0)
        self.assertEqual(note.duration, 1.0)
        self.assertEqual(note.channel, 0)
        self.assertEqual(note.pitch_bends, [])
    
    def test_with_pitch_bends(self):
        """Test MIDINote with pitch bends."""
        bends = [(0.5, 100), (0.8, -50)]
        note = MIDINote(60, 100, 0.0, 1.0, pitch_bends=bends)
        
        self.assertEqual(note.pitch_bends, bends)


class TestMIDIProcessor(unittest.TestCase):
    """Test MIDI processing functions."""
    
    def setUp(self):
        """Setup test notes."""
        self.processor = MIDIProcessor()
        
        # Create test notes
        self.notes = [
            MIDINote(60, 100, 0.0, 0.5),
            MIDINote(64, 80, 0.1, 0.6),
            MIDINote(67, 90, 0.2, 0.7),
            MIDINote(60, 100, 0.52, 1.0),  # Same pitch, small gap
        ]
    
    def test_quantize_timing(self):
        """Test timing quantization."""
        # Create note with off-grid timing
        notes = [MIDINote(60, 100, 0.127, 0.643)]
        quantized = self.processor.quantize_timing(notes)
        
        # Should quantize to 10ms grid
        note = quantized[0]
        self.assertAlmostEqual(note.start_time, 0.13, places=3)
        self.assertAlmostEqual(note.end_time, 0.64, places=3)
    
    def test_merge_gaps(self):
        """Test gap merging for same pitch."""
        notes = [
            MIDINote(60, 100, 0.0, 0.5),
            MIDINote(60, 80, 0.52, 1.0),  # 20ms gap, should merge
        ]
        
        merged = self.processor.merge_gaps(notes)
        self.assertEqual(len(merged), 1)
        
        note = merged[0]
        self.assertEqual(note.pitch, 60)
        self.assertEqual(note.start_time, 0.0)
        self.assertEqual(note.end_time, 1.0)
        self.assertEqual(note.velocity, 100)  # Takes higher velocity
    
    def test_clean_short_notes(self):
        """Test removal of short notes."""
        notes = [
            MIDINote(60, 100, 0.0, 1.0),    # Long note - keep
            MIDINote(64, 100, 1.0, 1.02),   # Very short - remove
        ]
        
        cleaned = self.processor.clean_short_notes(notes)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0].pitch, 60)
    
    def test_smooth_velocities(self):
        """Test velocity smoothing."""
        notes = [
            MIDINote(60, 100, 0.0, 0.5),
            MIDINote(64, 20, 0.5, 1.0),   # Very different velocity
        ]
        
        smoothed = self.processor.smooth_velocities(notes)
        
        # First note should be unchanged
        self.assertEqual(smoothed[0].velocity, 100)
        
        # Second note should be smoothed toward first
        self.assertGreater(smoothed[1].velocity, 20)
        self.assertLess(smoothed[1].velocity, 100)


class TestPitchBendProcessor(unittest.TestCase):
    """Test pitch bend processing."""
    
    def setUp(self):
        """Setup test notes with pitch bends."""
        self.processor = PitchBendProcessor(max_channels=4)
        
        self.notes_with_bends = [
            MIDINote(60, 100, 0.0, 1.0, pitch_bends=[(0.5, 100)]),
            MIDINote(64, 100, 0.5, 1.5, pitch_bends=[(1.0, -50)]),
            MIDINote(67, 100, 0.8, 1.8, pitch_bends=[(1.2, 200)]),
        ]
    
    def test_strip_pitch_bends(self):
        """Test pitch bend removal."""
        result = self.processor.process_pitch_bends(self.notes_with_bends, "none")
        
        for note in result:
            self.assertEqual(note.pitch_bends, [])
            self.assertEqual(note.channel, 0)
    
    def test_global_pitch_bends(self):
        """Test global pitch bend mode."""
        result = self.processor.process_pitch_bends(self.notes_with_bends, "global")
        
        # All notes should be on channel 0 but keep their bends
        for i, note in enumerate(result):
            self.assertEqual(note.channel, 0)
            self.assertEqual(note.pitch_bends, self.notes_with_bends[i].pitch_bends)
    
    def test_per_voice_pitch_bends(self):
        """Test per-voice channelization."""
        # Create overlapping notes
        overlapping_notes = [
            MIDINote(60, 100, 0.0, 1.0, pitch_bends=[(0.5, 100)]),
            MIDINote(64, 100, 0.5, 1.5, pitch_bends=[(1.0, -50)]),  # Overlaps with first
            MIDINote(67, 100, 1.0, 2.0, pitch_bends=[(1.5, 200)]),  # No overlap
        ]
        
        result = self.processor.process_pitch_bends(overlapping_notes, "per_voice")
        
        # First and second notes should be on different channels
        self.assertNotEqual(result[0].channel, result[1].channel)
        
        # Third note could be on same channel as first (no overlap)
        # All should keep their pitch bends
        for i, note in enumerate(result):
            self.assertEqual(note.pitch_bends, overlapping_notes[i].pitch_bends)
    
    def test_channel_overflow(self):
        """Test behavior when more overlapping notes than channels."""
        # Create more overlapping notes than available channels
        many_overlapping = [
            MIDINote(60 + i, 100, 0.0, 1.0, pitch_bends=[(0.5, i*10)])
            for i in range(6)  # 6 overlapping notes, only 4 channels
        ]
        
        result = self.processor.process_pitch_bends(many_overlapping, "per_voice")
        
        # Should not crash, extra notes go to channel 0
        channels_used = set(note.channel for note in result)
        self.assertLessEqual(len(channels_used), self.processor.max_channels + 1)


class TestProcessMIDINotes(unittest.TestCase):
    """Test the complete processing pipeline."""
    
    def test_complete_pipeline(self):
        """Test the complete MIDI processing pipeline."""
        # Create test chord with timing that will NOT overlap after quantization
        notes = [
            MIDINote(60, 100, 0.001, 0.201),   # Quantizes to 0.0-0.2 (no overlap)
            MIDINote(64, 80, 0.302, 0.502),    # Quantizes to 0.3-0.5 (no overlap)  
            MIDINote(67, 90, 0.603, 0.803),    # Quantizes to 0.6-0.8 (no overlap)
        ]
        
        processed = process_midi_notes(
            notes,
            bend_mode="per_voice",
            quantize_time=0.1,  # 100ms quantization for clear testing
        )
        
        # Should have 3 notes
        self.assertEqual(len(processed), 3)
        
        # Timing should be quantized
        expected_start_times = [0.0, 0.3, 0.6]
        expected_end_times = [0.2, 0.5, 0.8]
        
        # Sort by start time for consistent comparison
        processed_sorted = sorted(processed, key=lambda n: n.start_time)
        
        for i, note in enumerate(processed_sorted):
            self.assertAlmostEqual(note.start_time, expected_start_times[i], places=5)
            self.assertAlmostEqual(note.end_time, expected_end_times[i], places=5)
        
        # All notes should be on channel 0 (no overlap after quantization)
        for note in processed_sorted:
            self.assertEqual(note.channel, 0)


if __name__ == '__main__':
    unittest.main()