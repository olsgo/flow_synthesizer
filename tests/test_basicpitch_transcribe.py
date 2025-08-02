"""
Tests for Basic Pitch transcription backend.
"""

import unittest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from flowsynth.transcription.basicpitch_backend import (
    BasicPitchTranscriber, MockBasicPitchTranscriber, transcribe, BASIC_PITCH_AVAILABLE
)
from flowsynth.transcription.midi_tools import MIDINote


class TestMockBasicPitchTranscriber(unittest.TestCase):
    """Test the mock transcriber when Basic Pitch is not available."""
    
    def test_mock_transcriber_creation(self):
        """Test mock transcriber creation."""
        transcriber = MockBasicPitchTranscriber()
        self.assertIsInstance(transcriber, MockBasicPitchTranscriber)
    
    def test_mock_transcribe(self):
        """Test mock transcription."""
        transcriber = MockBasicPitchTranscriber()
        
        with tempfile.NamedTemporaryFile(suffix='.wav') as temp_file:
            result = transcriber.transcribe(temp_file.name)
            
            self.assertIn('midi_path', result)
            self.assertIn('note_events', result)
            self.assertIn('meta', result)
            
            self.assertIsNone(result['midi_path'])
            self.assertEqual(result['note_events'], [])
            self.assertTrue(result['meta']['mock'])


@unittest.skipIf(not BASIC_PITCH_AVAILABLE, "Basic Pitch not available")
class TestBasicPitchTranscriber(unittest.TestCase):
    """Test the real Basic Pitch transcriber when available."""
    
    def setUp(self):
        """Setup transcriber."""
        self.transcriber = BasicPitchTranscriber(
            min_freq=100.0,
            max_freq=1000.0,
            onset_threshold=0.6
        )
    
    def test_transcriber_creation(self):
        """Test transcriber creation."""
        self.assertEqual(self.transcriber.min_freq, 100.0)
        self.assertEqual(self.transcriber.max_freq, 1000.0)
        self.assertEqual(self.transcriber.onset_threshold, 0.6)
    
    @patch('flowsynth.transcription.basicpitch_backend.predict_and_save')
    def test_transcribe_with_save(self, mock_predict_and_save):
        """Test transcription with file saving."""
        # Mock the Basic Pitch response
        mock_predict_and_save.return_value = (
            ["/tmp/test.mid"],  # midi_path
            [  # note_events as list of dicts
                {
                    'pitch_midi': 60,
                    'velocity': 100,
                    'start_time': 0.0,
                    'end_time': 0.5,
                    'pitch_bends': []
                },
                {
                    'pitch_midi': 64,
                    'velocity': 80,
                    'start_time': 0.5,
                    'end_time': 1.0,
                    'pitch_bends': [{'time': 0.75, 'value': 100}]
                }
            ]
        )
        
        with tempfile.NamedTemporaryFile(suffix='.wav') as temp_file:
            result = self.transcriber.transcribe(temp_file.name, save_midi=True)
            
            # Check result structure
            self.assertIn('midi_path', result)
            self.assertIn('note_events', result)
            self.assertIn('meta', result)
            
            # Check MIDI notes
            notes = result['note_events']
            self.assertEqual(len(notes), 2)
            
            # First note
            note1 = notes[0]
            self.assertEqual(note1.pitch, 60)
            self.assertEqual(note1.velocity, 100)
            self.assertEqual(note1.start_time, 0.0)
            self.assertEqual(note1.end_time, 0.5)
            
            # Second note with pitch bend
            note2 = notes[1]
            self.assertEqual(note2.pitch, 64)
            self.assertEqual(len(note2.pitch_bends), 1)
            self.assertEqual(note2.pitch_bends[0], (0.75, 100))
    
    @patch('flowsynth.transcription.basicpitch_backend.predict')
    def test_transcribe_without_save(self, mock_predict):
        """Test transcription without saving files."""
        # Mock the Basic Pitch response
        mock_predict.return_value = (
            None,  # model_output
            None,  # midi_data
            [{'pitch_midi': 60, 'velocity': 100, 'start_time': 0.0, 'end_time': 0.5}]
        )
        
        with tempfile.NamedTemporaryFile(suffix='.wav') as temp_file:
            result = self.transcriber.transcribe(
                temp_file.name, 
                save_midi=False, 
                save_notes_csv=False
            )
            
            self.assertIsNone(result['midi_path'])
            self.assertEqual(len(result['note_events']), 1)
    
    def test_convert_note_events_list(self):
        """Test note event conversion from list format."""
        note_events = [
            {'pitch_midi': 60, 'velocity': 100, 'start_time': 0.0, 'end_time': 0.5},
            (0.5, 1.0, 64, 80),  # tuple format: (start, end, pitch, velocity)
        ]
        
        notes = self.transcriber._convert_note_events(note_events)
        
        self.assertEqual(len(notes), 2)
        self.assertEqual(notes[0].pitch, 60)
        self.assertEqual(notes[1].pitch, 64)
    
    def test_parse_pitch_bends(self):
        """Test pitch bend parsing."""
        bend_data = [
            {'time': 0.25, 'value': 100},
            (0.75, -50),  # tuple format
        ]
        
        bends = self.transcriber._parse_pitch_bends(bend_data)
        
        self.assertEqual(len(bends), 2)
        self.assertEqual(bends[0], (0.25, 100))
        self.assertEqual(bends[1], (0.75, -50))


class TestTranscribeFunction(unittest.TestCase):
    """Test the convenience transcribe function."""
    
    @patch('flowsynth.transcription.basicpitch_backend.BASIC_PITCH_AVAILABLE', False)
    def test_transcribe_with_mock(self):
        """Test transcribe function uses mock when Basic Pitch unavailable."""
        with tempfile.NamedTemporaryFile(suffix='.wav') as temp_file:
            result = transcribe(temp_file.name)
            
            # Should return mock result
            self.assertEqual(result['note_events'], [])
            self.assertTrue(result['meta']['mock'])
    
    @unittest.skipIf(not BASIC_PITCH_AVAILABLE, "Basic Pitch not available")
    @patch('flowsynth.transcription.basicpitch_backend.predict_and_save')
    def test_transcribe_with_real(self, mock_predict_and_save):
        """Test transcribe function with real Basic Pitch."""
        mock_predict_and_save.return_value = (["/tmp/test.mid"], [])
        
        with tempfile.NamedTemporaryFile(suffix='.wav') as temp_file:
            result = transcribe(
                temp_file.name,
                min_freq=100.0,
                max_freq=1000.0
            )
            
            self.assertIn('note_events', result)
            # Verify parameters were passed
            mock_predict_and_save.assert_called_once()
            call_kwargs = mock_predict_and_save.call_args[1]
            self.assertEqual(call_kwargs['minimum_frequency'], 100.0)
            self.assertEqual(call_kwargs['maximum_frequency'], 1000.0)


class TestNoteEventConversion(unittest.TestCase):
    """Test various note event format conversions."""
    
    def setUp(self):
        """Setup transcriber for testing."""
        if BASIC_PITCH_AVAILABLE:
            self.transcriber = BasicPitchTranscriber()
        else:
            self.transcriber = MockBasicPitchTranscriber()
    
    @unittest.skipIf(not BASIC_PITCH_AVAILABLE, "Basic Pitch not available")
    def test_convert_pandas_dataframe(self):
        """Test conversion from pandas DataFrame format."""
        # Mock pandas DataFrame
        mock_df = Mock()
        mock_df.iterrows.return_value = [
            (0, {'pitch_midi': 60, 'velocity': 100, 'start_time': 0.0, 'end_time': 0.5}),
            (1, {'pitch_midi': 64, 'velocity': 80, 'start_time': 0.5, 'end_time': 1.0}),
        ]
        
        notes = self.transcriber._convert_note_events(mock_df)
        
        self.assertEqual(len(notes), 2)
        self.assertIsInstance(notes[0], MIDINote)
        self.assertEqual(notes[0].pitch, 60)
        self.assertEqual(notes[1].pitch, 64)
    
    @unittest.skipIf(not BASIC_PITCH_AVAILABLE, "Basic Pitch not available")
    def test_convert_empty_events(self):
        """Test conversion of empty note events."""
        notes = self.transcriber._convert_note_events(None)
        self.assertEqual(notes, [])
        
        notes = self.transcriber._convert_note_events([])
        self.assertEqual(notes, [])
    
    @unittest.skipIf(not BASIC_PITCH_AVAILABLE, "Basic Pitch not available")
    def test_convert_malformed_events(self):
        """Test handling of malformed note events."""
        malformed_events = [
            "invalid_string",
            (1, 2),  # Too few elements
            {},  # Empty dict
        ]
        
        notes = self.transcriber._convert_note_events(malformed_events)
        
        # Should handle gracefully and skip malformed events
        self.assertEqual(len(notes), 1)  # Only the empty dict should create a note with defaults


if __name__ == '__main__':
    unittest.main()