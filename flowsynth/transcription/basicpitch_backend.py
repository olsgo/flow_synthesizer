"""
Basic Pitch transcription backend for polyphonic audio-to-MIDI conversion.

This module provides a wrapper around Spotify's Basic Pitch for:
- Polyphonic, instrument-agnostic transcription
- MIDI output with pitch bend information
- Configurable frequency range and post-processing
"""

import os
import logging
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path

from .midi_tools import MIDINote

logger = logging.getLogger(__name__)

# Try to import basic-pitch, but handle gracefully if not available
try:
    import basic_pitch
    from basic_pitch.inference import predict, predict_and_save
    BASIC_PITCH_AVAILABLE = True
    logger.info("Basic Pitch is available")
except ImportError:
    logger.warning("Basic Pitch not available. Install with: pip install basic-pitch")
    BASIC_PITCH_AVAILABLE = False
    basic_pitch = None
    predict = None
    predict_and_save = None


class BasicPitchTranscriber:
    """Wrapper for Basic Pitch transcription with polyphonic support."""
    
    def __init__(self,
                 min_freq: float = 80.0,     # Hz, roughly E2
                 max_freq: float = 2000.0,   # Hz, roughly B6
                 onset_threshold: float = 0.5,
                 frame_threshold: float = 0.3,
                 minimum_note_length: float = 0.127,  # seconds
                 minimum_frequency: Optional[float] = None,
                 maximum_frequency: Optional[float] = None,
                 multiple_pitch_bends: bool = False,
                 melodia_trick: bool = True,
                 debug_file: Optional[str] = None):
        """
        Initialize the Basic Pitch transcriber.
        
        Args:
            min_freq: Minimum frequency to transcribe (Hz)
            max_freq: Maximum frequency to transcribe (Hz)
            onset_threshold: Threshold for note onset detection
            frame_threshold: Threshold for note frame detection
            minimum_note_length: Minimum note duration in seconds
            minimum_frequency: Alias for min_freq (Basic Pitch parameter name)
            maximum_frequency: Alias for max_freq (Basic Pitch parameter name)
            multiple_pitch_bends: Whether to extract multiple pitch bends per note
            melodia_trick: Use melodia trick for better transcription
            debug_file: Optional file to save debug information
        """
        if not BASIC_PITCH_AVAILABLE:
            raise ImportError("Basic Pitch is not available. Install with: pip install basic-pitch")
            
        self.min_freq = minimum_frequency if minimum_frequency is not None else min_freq
        self.max_freq = maximum_frequency if maximum_frequency is not None else max_freq
        self.onset_threshold = onset_threshold
        self.frame_threshold = frame_threshold
        self.minimum_note_length = minimum_note_length
        self.multiple_pitch_bends = multiple_pitch_bends
        self.melodia_trick = melodia_trick
        self.debug_file = debug_file
        
        logger.info(f"BasicPitch transcriber initialized: freq_range=({self.min_freq:.1f}, {self.max_freq:.1f}) Hz")
    
    def transcribe(self, 
                   audio_path: Union[str, Path], 
                   save_dir: Optional[Union[str, Path]] = None,
                   save_midi: bool = True,
                   save_notes_csv: bool = True) -> Dict[str, Any]:
        """
        Transcribe audio to MIDI using Basic Pitch.
        
        Args:
            audio_path: Path to input audio file
            save_dir: Directory to save outputs (if None, saves next to audio file)
            save_midi: Whether to save MIDI file
            save_notes_csv: Whether to save notes CSV file
            
        Returns:
            Dictionary containing:
            - midi_path: Path to saved MIDI file (if save_midi=True)
            - note_events: List of note events as MIDINote objects
            - meta: Metadata about the transcription
        """
        if not BASIC_PITCH_AVAILABLE:
            raise RuntimeError("Basic Pitch is not available")
            
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        if save_dir is None:
            save_dir = audio_path.parent
        else:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Transcribing audio: {audio_path}")
        
        try:
            # Basic Pitch inference parameters
            inference_params = {
                'onset_threshold': self.onset_threshold,
                'frame_threshold': self.frame_threshold,
                'minimum_note_length': self.minimum_note_length,
                'minimum_frequency': self.min_freq,
                'maximum_frequency': self.max_freq,
                'multiple_pitch_bends': self.multiple_pitch_bends,
                'melodia_trick': self.melodia_trick
            }
            
            if save_midi or save_notes_csv:
                # Use predict_and_save for file outputs
                output_directory = save_dir
                midi_path, note_events = predict_and_save(
                    [str(audio_path)],
                    output_directory,
                    save_midi=save_midi,
                    save_model_outputs=False,
                    save_notes=save_notes_csv,
                    **inference_params
                )
                
                # Get the actual output path
                if save_midi and midi_path:
                    midi_file_path = midi_path[0] if isinstance(midi_path, list) else midi_path
                else:
                    midi_file_path = None
                    
            else:
                # Use predict for in-memory results only
                model_output, midi_data, note_events = predict(
                    str(audio_path),
                    **inference_params
                )
                midi_file_path = None
            
            # Convert note events to MIDINote objects
            midi_notes = self._convert_note_events(note_events)
            
            # Create metadata
            meta = {
                'audio_file': str(audio_path),
                'transcription_params': inference_params,
                'note_count': len(midi_notes),
                'freq_range': (self.min_freq, self.max_freq),
                'duration': max(note.end_time for note in midi_notes) if midi_notes else 0.0
            }
            
            if self.debug_file:
                self._save_debug_info(meta, self.debug_file)
            
            result = {
                'midi_path': midi_file_path,
                'note_events': midi_notes,
                'meta': meta
            }
            
            logger.info(f"Transcription complete: {len(midi_notes)} notes, duration={meta['duration']:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    def _convert_note_events(self, note_events) -> List[MIDINote]:
        """Convert Basic Pitch note events to MIDINote objects."""
        midi_notes = []
        
        if note_events is None:
            return midi_notes
            
        # Handle different note event formats
        if hasattr(note_events, 'iterrows'):  # pandas DataFrame
            for _, row in note_events.iterrows():
                pitch = int(row.get('pitch_midi', row.get('pitch', 60)))
                velocity = int(row.get('velocity', 80))
                start_time = float(row.get('start_time', 0))
                end_time = float(row.get('end_time', start_time + 0.5))
                
                # Extract pitch bends if available
                pitch_bends = []
                if 'pitch_bends' in row and row['pitch_bends']:
                    pitch_bends = self._parse_pitch_bends(row['pitch_bends'])
                
                midi_note = MIDINote(
                    pitch=pitch,
                    velocity=velocity,
                    start_time=start_time,
                    end_time=end_time,
                    channel=0,
                    pitch_bends=pitch_bends
                )
                midi_notes.append(midi_note)
                
        elif isinstance(note_events, list):  # List of dictionaries or tuples
            for event in note_events:
                if isinstance(event, dict):
                    pitch = int(event.get('pitch_midi', event.get('pitch', 60)))
                    velocity = int(event.get('velocity', 80))
                    start_time = float(event.get('start_time', 0))
                    end_time = float(event.get('end_time', start_time + 0.5))
                    pitch_bends = self._parse_pitch_bends(event.get('pitch_bends', []))
                    
                elif isinstance(event, (tuple, list)) and len(event) >= 4:
                    # Assume format: (start_time, end_time, pitch, velocity, ...)
                    start_time, end_time, pitch, velocity = event[:4]
                    pitch_bends = []
                    
                else:
                    logger.warning(f"Unknown note event format: {event}")
                    continue
                
                midi_note = MIDINote(
                    pitch=int(pitch),
                    velocity=int(velocity),
                    start_time=float(start_time),
                    end_time=float(end_time),
                    channel=0,
                    pitch_bends=pitch_bends
                )
                midi_notes.append(midi_note)
        
        return midi_notes
    
    def _parse_pitch_bends(self, pitch_bend_data) -> List[Tuple[float, int]]:
        """Parse pitch bend data into (time, value) tuples."""
        if not pitch_bend_data:
            return []
            
        pitch_bends = []
        
        # Handle different pitch bend formats
        if isinstance(pitch_bend_data, list):
            for bend in pitch_bend_data:
                if isinstance(bend, dict):
                    time = float(bend.get('time', 0))
                    value = int(bend.get('value', 0))
                    pitch_bends.append((time, value))
                elif isinstance(bend, (tuple, list)) and len(bend) >= 2:
                    time, value = bend[:2]
                    pitch_bends.append((float(time), int(value)))
        
        return pitch_bends
    
    def _save_debug_info(self, meta: Dict[str, Any], debug_file: str):
        """Save debug information to file."""
        try:
            import json
            with open(debug_file, 'w') as f:
                json.dump(meta, f, indent=2)
            logger.info(f"Debug info saved to: {debug_file}")
        except Exception as e:
            logger.warning(f"Failed to save debug info: {e}")


# Mock transcriber for when Basic Pitch is not available
class MockBasicPitchTranscriber:
    """Mock transcriber that returns empty results for testing."""
    
    def __init__(self, **kwargs):
        logger.warning("Using mock transcriber - Basic Pitch not available")
        
    def transcribe(self, audio_path, save_dir=None, save_midi=True, save_notes_csv=True):
        """Return mock transcription results."""
        logger.warning("Mock transcription - no actual processing performed")
        
        return {
            'midi_path': None,
            'note_events': [],
            'meta': {
                'audio_file': str(audio_path),
                'note_count': 0,
                'duration': 0.0,
                'mock': True
            }
        }


def transcribe(audio_path: Union[str, Path],
               save_dir: Optional[Union[str, Path]] = None,
               save_midi: bool = True,
               save_notes_csv: bool = True,
               min_freq: float = 80.0,
               max_freq: float = 2000.0,
               onset_threshold: float = 0.5,
               frame_threshold: float = 0.3,
               minimum_note_length: float = 0.127,
               **kwargs) -> Dict[str, Any]:
    """
    Convenience function for Basic Pitch transcription.
    
    Args:
        audio_path: Path to input audio file
        save_dir: Directory to save outputs
        save_midi: Whether to save MIDI file
        save_notes_csv: Whether to save notes CSV
        min_freq: Minimum frequency to transcribe (Hz)
        max_freq: Maximum frequency to transcribe (Hz)
        onset_threshold: Note onset detection threshold
        frame_threshold: Note frame detection threshold
        minimum_note_length: Minimum note duration in seconds
        **kwargs: Additional parameters for BasicPitchTranscriber
        
    Returns:
        Transcription results dictionary
    """
    if BASIC_PITCH_AVAILABLE:
        transcriber = BasicPitchTranscriber(
            min_freq=min_freq,
            max_freq=max_freq,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            minimum_note_length=minimum_note_length,
            **kwargs
        )
    else:
        transcriber = MockBasicPitchTranscriber(**kwargs)
    
    return transcriber.transcribe(
        audio_path=audio_path,
        save_dir=save_dir,
        save_midi=save_midi,
        save_notes_csv=save_notes_csv
    )