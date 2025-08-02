"""
Tests for DawDreamer scheduler.
"""

import unittest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from flowsynth.render.dawdreamer_scheduler import DawDreamerScheduler, PluginRegistry, create_default_registry
from flowsynth.transcription.midi_tools import MIDINote


class TestPluginRegistry(unittest.TestCase):
    """Test plugin registry functionality."""
    
    def test_creation(self):
        """Test registry creation."""
        registry = PluginRegistry()
        self.assertEqual(registry.plugins, {})
        self.assertEqual(registry.presets, {})
    
    def test_register_plugin(self):
        """Test plugin registration."""
        registry = PluginRegistry()
        registry.register_plugin("test_synth", "/path/to/synth.vst")
        
        self.assertEqual(registry.get_plugin_path("test_synth"), "/path/to/synth.vst")
        self.assertIsNone(registry.get_plugin_path("nonexistent"))
    
    def test_register_with_presets(self):
        """Test plugin registration with presets."""
        registry = PluginRegistry()
        presets = {"init": "/path/to/init.preset", "bass": "/path/to/bass.preset"}
        registry.register_plugin("test_synth", "/path/to/synth.vst", presets)
        
        self.assertEqual(registry.get_preset_path("test_synth", "init"), "/path/to/init.preset")
        self.assertEqual(registry.get_preset_path("test_synth", "bass"), "/path/to/bass.preset")
        self.assertIsNone(registry.get_preset_path("test_synth", "nonexistent"))


class TestDawDreamerScheduler(unittest.TestCase):
    """Test DawDreamer scheduler functionality."""
    
    def setUp(self):
        """Setup mock DawDreamer environment."""
        # Mock dawdreamer module
        self.mock_daw = Mock()
        self.mock_engine = Mock()
        self.mock_plugin = Mock()
        
        # Setup mock returns
        self.mock_daw.RenderEngine.return_value = self.mock_engine
        self.mock_engine.make_plugin_processor.return_value = self.mock_plugin
        self.mock_engine.get_audio.return_value = np.random.randn(2, 1000)
        
        # Mock plugin methods
        self.mock_plugin.get_plugin_parameter_size.return_value = 10
        self.mock_plugin.get_parameter.return_value = 0.5
        self.mock_plugin.get_parameters_description.return_value = [
            {"name": "Filter Cutoff"}, {"name": "Resonance"}
        ]
        
        with patch('flowsynth.render.dawdreamer_scheduler.daw', self.mock_daw):
            self.scheduler = DawDreamerScheduler()
    
    def test_creation(self):
        """Test scheduler creation."""
        self.assertEqual(self.scheduler.sample_rate, 22050)
        self.assertEqual(self.scheduler.block_size, 512)
        self.assertIsNotNone(self.scheduler.plugin_registry)
        self.assertEqual(self.scheduler.current_bpm, 120.0)
    
    @patch('os.path.exists')
    def test_load_plugin_success(self, mock_exists):
        """Test successful plugin loading."""
        mock_exists.return_value = True
        
        result = self.scheduler.load_plugin("/fake/path/synth.vst")
        
        self.assertTrue(result)
        self.mock_engine.make_plugin_processor.assert_called_once()
    
    @patch('os.path.exists')
    def test_load_plugin_failure(self, mock_exists):
        """Test plugin loading failure."""
        mock_exists.return_value = False
        
        result = self.scheduler.load_plugin("/nonexistent/synth.vst")
        
        self.assertFalse(result)
    
    def test_set_bpm(self):
        """Test BPM setting."""
        self.scheduler.set_bpm(140.0)
        
        self.assertEqual(self.scheduler.current_bpm, 140.0)
        self.mock_engine.set_bpm.assert_called_with(140.0)
    
    def test_render_notes_only(self):
        """Test notes-only rendering."""
        # Setup plugin as loaded
        self.scheduler.current_plugin = self.mock_plugin
        
        notes = [
            MIDINote(60, 100, 0.0, 0.5),
            MIDINote(64, 80, 0.5, 1.0),
        ]
        
        audio = self.scheduler.render_notes_only(notes, 2.0)
        
        # Verify MIDI notes were added
        self.assertEqual(self.mock_plugin.add_midi_note.call_count, 2)
        
        # Verify render was called
        self.mock_engine.render.assert_called_with(2.0)
        
        # Check audio output
        self.assertIsInstance(audio, np.ndarray)
    
    def test_render_without_plugin(self):
        """Test rendering without loaded plugin."""
        notes = [MIDINote(60, 100, 0.0, 0.5)]
        
        with self.assertRaises(RuntimeError):
            self.scheduler.render_notes_only(notes, 1.0)
    
    @patch('flowsynth.render.dawdreamer_scheduler.tempfile.gettempdir')
    @patch('os.path.exists')
    @patch('os.unlink')
    def test_render_with_midi_file_mock(self, mock_unlink, mock_exists, mock_tempdir):
        """Test MIDI file rendering with mocked dependencies."""
        # Setup
        self.scheduler.current_plugin = self.mock_plugin
        mock_tempdir.return_value = "/tmp"
        mock_exists.return_value = True
        
        # Mock the MIDI file creation
        with patch.object(self.scheduler, '_create_midi_file') as mock_create_midi:
            mock_create_midi.return_value = "/tmp/test.mid"
            
            notes = [MIDINote(60, 100, 0.0, 0.5, pitch_bends=[(0.25, 100)])]
            
            audio = self.scheduler.render_with_midi_file(notes, 1.0)
            
            # Verify MIDI file was created and loaded
            mock_create_midi.assert_called_once()
            self.mock_plugin.load_midi.assert_called_with("/tmp/test.mid")
            
            # Verify cleanup
            mock_unlink.assert_called_with("/tmp/test.mid")
    
    def test_get_plugin_info(self):
        """Test plugin info retrieval."""
        self.scheduler.current_plugin = self.mock_plugin
        
        info = self.scheduler.get_plugin_info()
        
        self.assertIn('parameter_count', info)
        self.assertIn('parameters', info)
        self.assertEqual(info['parameter_count'], 10)
    
    def test_get_parameters(self):
        """Test parameter retrieval."""
        self.scheduler.current_plugin = self.mock_plugin
        
        params = self.scheduler.get_parameters()
        
        self.assertEqual(len(params), 10)
        for i, (param_idx, value) in enumerate(params):
            self.assertEqual(param_idx, i)
            self.assertEqual(value, 0.5)
    
    def test_set_parameters(self):
        """Test parameter setting."""
        self.scheduler.current_plugin = self.mock_plugin
        
        params = [(0, 0.7), (1, 0.3)]
        self.scheduler.set_parameters(params)
        
        self.assertEqual(self.mock_plugin.set_parameter.call_count, 2)


class TestMIDIFileCreation(unittest.TestCase):
    """Test MIDI file creation functionality."""
    
    def setUp(self):
        """Setup for MIDI file tests."""
        # Mock pretty_midi if available
        self.mock_pretty_midi = Mock()
        self.mock_midi_obj = Mock()
        self.mock_instrument = Mock()
        
        self.mock_pretty_midi.PrettyMIDI.return_value = self.mock_midi_obj
        self.mock_pretty_midi.Instrument.return_value = self.mock_instrument
        self.mock_pretty_midi.Note = Mock()
        self.mock_pretty_midi.PitchBend = Mock()
        
        self.mock_midi_obj.instruments = []
    
    @patch('flowsynth.render.dawdreamer_scheduler.tempfile.gettempdir')
    def test_create_midi_file_mock(self, mock_tempdir):
        """Test MIDI file creation with mocked pretty_midi."""
        mock_tempdir.return_value = "/tmp"
        
        # Mock pretty_midi at import time in the method
        with patch.dict('sys.modules', {'pretty_midi': self.mock_pretty_midi}):
            scheduler = DawDreamerScheduler()
            
            notes = [
                MIDINote(60, 100, 0.0, 0.5, channel=0),
                MIDINote(64, 80, 0.5, 1.0, channel=1, pitch_bends=[(0.75, 100)])
            ]
            
            # This will fail if pretty_midi is not available, which is expected
            try:
                midi_path = scheduler._create_midi_file(notes)
                # If we get here, the mock worked
                self.assertTrue(midi_path.startswith("/tmp"))
                self.assertTrue(midi_path.endswith(".mid"))
            except RuntimeError as e:
                # Expected if pretty_midi is not available
                self.assertIn("pretty_midi required", str(e))


class TestDefaultRegistry(unittest.TestCase):
    """Test default registry creation."""
    
    @patch('os.path.exists')
    def test_create_default_registry(self, mock_exists):
        """Test default registry creation."""
        # Mock that some plugins exist
        mock_exists.side_effect = lambda path: path.endswith('Diva.vst')
        
        registry = create_default_registry()
        
        # Should have registered the existing plugin
        self.assertIsNotNone(registry.get_plugin_path('diva'))
        self.assertIsNone(registry.get_plugin_path('massive_x'))  # Doesn't exist


if __name__ == '__main__':
    unittest.main()