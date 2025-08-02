"""
MIDI post-processing tools for quantization, note cleaning, and pitch bend handling.

This module provides utilities for:
- Quantizing and merging micro-gaps in MIDI
- Cleaning sub-threshold notes and smoothing velocity
- Handling pitch bend policies (none/global/per_voice)
- Per-voice channelization for polyphonic pitch bends
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MIDINote:
    """Represents a MIDI note with timing and pitch bend information."""
    
    def __init__(self, pitch: int, velocity: int, start_time: float, end_time: float, 
                 channel: int = 0, pitch_bends: Optional[List[Tuple[float, int]]] = None):
        self.pitch = pitch
        self.velocity = velocity
        self.start_time = start_time
        self.end_time = end_time
        self.channel = channel
        self.pitch_bends = pitch_bends or []
        
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
        
    def __repr__(self):
        return f"MIDINote(pitch={self.pitch}, vel={self.velocity}, start={self.start_time:.3f}, dur={self.duration:.3f}, ch={self.channel})"


class MIDIProcessor:
    """Processes MIDI notes for polyphonic synthesis."""
    
    def __init__(self, 
                 quantize_time: float = 0.010,  # 10ms quantization
                 min_note_duration: float = 0.050,  # 50ms minimum
                 merge_gap_threshold: float = 0.030,  # 30ms gap merging
                 velocity_smoothing: float = 0.1):  # smoothing factor
        self.quantize_time = quantize_time
        self.min_note_duration = min_note_duration
        self.merge_gap_threshold = merge_gap_threshold
        self.velocity_smoothing = velocity_smoothing
        
    def quantize_timing(self, notes: List[MIDINote]) -> List[MIDINote]:
        """Quantize note timings to reduce jitter."""
        quantized_notes = []
        
        for note in notes:
            # Quantize start and end times
            quantized_start = round(note.start_time / self.quantize_time) * self.quantize_time
            quantized_end = round(note.end_time / self.quantize_time) * self.quantize_time
            
            # Ensure minimum duration
            if quantized_end - quantized_start < self.min_note_duration:
                quantized_end = quantized_start + self.min_note_duration
                
            quantized_note = MIDINote(
                pitch=note.pitch,
                velocity=note.velocity,
                start_time=quantized_start,
                end_time=quantized_end,
                channel=note.channel,
                pitch_bends=note.pitch_bends
            )
            quantized_notes.append(quantized_note)
            
        return quantized_notes
    
    def merge_gaps(self, notes: List[MIDINote]) -> List[MIDINote]:
        """Merge notes with small gaps between them (same pitch)."""
        if not notes:
            return notes
            
        # Sort by pitch, then by start time
        notes = sorted(notes, key=lambda n: (n.pitch, n.start_time))
        merged_notes = []
        
        i = 0
        while i < len(notes):
            current_note = notes[i]
            
            # Look for notes of the same pitch that are close in time
            j = i + 1
            while (j < len(notes) and 
                   notes[j].pitch == current_note.pitch and
                   notes[j].start_time - current_note.end_time <= self.merge_gap_threshold):
                
                # Merge the notes
                current_note = MIDINote(
                    pitch=current_note.pitch,
                    velocity=max(current_note.velocity, notes[j].velocity),  # Take higher velocity
                    start_time=current_note.start_time,
                    end_time=notes[j].end_time,
                    channel=current_note.channel,
                    pitch_bends=current_note.pitch_bends + notes[j].pitch_bends
                )
                j += 1
                
            merged_notes.append(current_note)
            i = j
            
        return merged_notes
    
    def clean_short_notes(self, notes: List[MIDINote]) -> List[MIDINote]:
        """Remove notes shorter than minimum duration."""
        return [note for note in notes if note.duration >= self.min_note_duration]
    
    def smooth_velocities(self, notes: List[MIDINote]) -> List[MIDINote]:
        """Apply velocity smoothing to reduce abrupt changes."""
        if not notes:
            return notes
            
        # Sort by start time
        notes = sorted(notes, key=lambda n: n.start_time)
        smoothed_notes = []
        
        for i, note in enumerate(notes):
            if i == 0:
                smoothed_velocity = note.velocity
            else:
                # Simple exponential smoothing
                prev_velocity = smoothed_notes[-1].velocity
                smoothed_velocity = int(
                    self.velocity_smoothing * note.velocity + 
                    (1 - self.velocity_smoothing) * prev_velocity
                )
                
            smoothed_note = MIDINote(
                pitch=note.pitch,
                velocity=max(1, min(127, smoothed_velocity)),  # Clamp to valid range
                start_time=note.start_time,
                end_time=note.end_time,
                channel=note.channel,
                pitch_bends=note.pitch_bends
            )
            smoothed_notes.append(smoothed_note)
            
        return smoothed_notes


class PitchBendProcessor:
    """Handles pitch bend processing for different polyphonic modes."""
    
    def __init__(self, max_channels: int = 8):
        self.max_channels = max_channels
    
    def process_pitch_bends(self, notes: List[MIDINote], bend_mode: str = "per_voice") -> List[MIDINote]:
        """
        Process pitch bends according to the specified mode.
        
        Args:
            notes: List of MIDI notes with pitch bend data
            bend_mode: "none", "global", or "per_voice"
            
        Returns:
            List of processed MIDI notes with appropriate channel assignments
        """
        if bend_mode == "none":
            return self._strip_pitch_bends(notes)
        elif bend_mode == "global":
            return self._global_pitch_bends(notes)
        elif bend_mode == "per_voice":
            return self._per_voice_pitch_bends(notes)
        else:
            raise ValueError(f"Unknown bend_mode: {bend_mode}")
    
    def _strip_pitch_bends(self, notes: List[MIDINote]) -> List[MIDINote]:
        """Remove all pitch bend information."""
        processed_notes = []
        for note in notes:
            processed_note = MIDINote(
                pitch=note.pitch,
                velocity=note.velocity,
                start_time=note.start_time,
                end_time=note.end_time,
                channel=0,  # All notes on channel 0
                pitch_bends=[]  # No bends
            )
            processed_notes.append(processed_note)
        return processed_notes
    
    def _global_pitch_bends(self, notes: List[MIDINote]) -> List[MIDINote]:
        """Keep bends but warn that they affect all simultaneous notes."""
        logger.warning("Global pitch bend mode: bends will affect all simultaneous notes on the channel")
        processed_notes = []
        for note in notes:
            processed_note = MIDINote(
                pitch=note.pitch,
                velocity=note.velocity,
                start_time=note.start_time,
                end_time=note.end_time,
                channel=0,  # All notes on channel 0
                pitch_bends=note.pitch_bends
            )
            processed_notes.append(processed_note)
        return processed_notes
    
    def _per_voice_pitch_bends(self, notes: List[MIDINote]) -> List[MIDINote]:
        """Assign overlapping notes to different channels to preserve independent bends."""
        if not notes:
            return notes
            
        # Sort notes by start time
        notes = sorted(notes, key=lambda n: n.start_time)
        processed_notes = []
        
        # Track which channels are in use at any given time
        channel_availability = [[] for _ in range(self.max_channels)]  # Lists of (start, end) times
        
        for note in notes:
            # Find an available channel
            assigned_channel = None
            
            for channel_idx in range(self.max_channels):
                # Check if this channel is free during the note's duration
                channel_free = True
                for start, end in channel_availability[channel_idx]:
                    # Check for overlap: note overlaps if it starts before existing note ends
                    # and ends after existing note starts
                    if not (note.end_time <= start or note.start_time >= end):
                        channel_free = False
                        break
                        
                if channel_free:
                    assigned_channel = channel_idx
                    channel_availability[channel_idx].append((note.start_time, note.end_time))
                    break
            
            if assigned_channel is None:
                # No free channel found, use channel 0 and warn
                logger.warning(f"No free channel for note at {note.start_time:.3f}s, using channel 0")
                assigned_channel = 0
                
            processed_note = MIDINote(
                pitch=note.pitch,
                velocity=note.velocity,
                start_time=note.start_time,
                end_time=note.end_time,
                channel=assigned_channel,
                pitch_bends=note.pitch_bends
            )
            processed_notes.append(processed_note)
            
        return processed_notes


def process_midi_notes(notes: List[MIDINote], 
                      bend_mode: str = "per_voice",
                      max_channels: int = 8,
                      quantize_time: float = 0.010,
                      min_note_duration: float = 0.050,
                      merge_gap_threshold: float = 0.030,
                      velocity_smoothing: float = 0.1) -> List[MIDINote]:
    """
    Complete MIDI processing pipeline.
    
    Args:
        notes: Input MIDI notes
        bend_mode: Pitch bend handling mode ("none", "global", "per_voice")
        max_channels: Maximum MIDI channels for per_voice mode
        quantize_time: Time quantization resolution in seconds
        min_note_duration: Minimum note duration in seconds
        merge_gap_threshold: Maximum gap to merge in seconds
        velocity_smoothing: Velocity smoothing factor (0-1)
        
    Returns:
        Processed MIDI notes
    """
    processor = MIDIProcessor(
        quantize_time=quantize_time,
        min_note_duration=min_note_duration,
        merge_gap_threshold=merge_gap_threshold,
        velocity_smoothing=velocity_smoothing
    )
    
    bend_processor = PitchBendProcessor(max_channels=max_channels)
    
    # Processing pipeline
    notes = processor.quantize_timing(notes)
    notes = processor.merge_gaps(notes)
    notes = processor.clean_short_notes(notes)
    notes = processor.smooth_velocities(notes)
    notes = bend_processor.process_pitch_bends(notes, bend_mode)
    
    return notes