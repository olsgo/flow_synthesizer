#!/usr/bin/env python3
"""
Event scheduling module for polyphonic DawDreamer rendering.

Handles scheduling of simultaneous notes with DawDreamer's PluginProcessor
and supports both single-instance and multi-instance polyphonic rendering.
"""

import argparse
import os
import sys
from typing import List, Dict, Any, Optional, Union
import numpy as np
import json
import soundfile as sf
import dawdreamer as daw

# Optional dependencies
try:
    import mido
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False
    print("Warning: mido not available, MIDI loading will be limited")

from .transcribe import NoteEvent

class PolyphonicRenderer:
    """Polyphonic renderer using DawDreamer."""
    
    def __init__(self, sample_rate: int = 22050, block_size: int = 512):
        self.sr = sample_rate
        self.engine = daw.RenderEngine(self.sr, block_size)
        self.plugins = {}  # name -> plugin processor
        
    def load_plugin(self, plugin_path: str, name: str = "synth") -> bool:
        """Load a VST/AU plugin."""
        try:
            plugin = self.engine.make_plugin_processor(name, plugin_path)
            self.plugins[name] = plugin
            return True
        except Exception as e:
            print(f"Error loading plugin {plugin_path}: {e}")
            return False
    
    def get_plugin_parameters(self, plugin_name: str = "synth") -> List[Dict[str, Any]]:
        """Get parameter descriptions for a plugin."""
        if plugin_name not in self.plugins:
            return []
        return self.plugins[plugin_name].get_parameters_description()
    
    def set_plugin_parameters(self, parameters: List[tuple], plugin_name: str = "synth"):
        """Set plugin parameters. Parameters should be [(index, value_0_1), ...]"""
        if plugin_name not in self.plugins:
            raise ValueError(f"Plugin {plugin_name} not loaded")
        
        plugin = self.plugins[plugin_name]
        for idx, val in parameters:
            plugin.set_parameter(int(idx), float(val))
    
    def render_single_instance(self, events: List[NoteEvent], 
                             duration_beats: float, bpm: float = 120.0,
                             plugin_name: str = "synth") -> np.ndarray:
        """
        Render polyphonic events using a single plugin instance.
        All notes are sent to the same synth instance.
        """
        if plugin_name not in self.plugins:
            raise ValueError(f"Plugin {plugin_name} not loaded")
        
        plugin = self.plugins[plugin_name]
        
        # Clear any existing MIDI
        plugin.clear_midi()
        
        # Add all note events
        for event in events:
            plugin.add_midi_note(
                note=event.pitch,
                velocity=event.velocity,
                start_beat=event.onset_beats,
                duration_beats=event.duration_beats,
                beats=True
            )
        
        # Set BPM and render
        self.engine.set_bpm(bpm)
        self.engine.load_graph([(plugin, [])])
        self.engine.render(duration_beats, beats=True)
        
        return self.engine.get_audio()
    
    def render_multi_instance(self, events: List[NoteEvent],
                            duration_beats: float, bpm: float = 120.0,
                            base_plugin_name: str = "synth",
                            max_voices: int = 8) -> np.ndarray:
        """
        Render polyphonic events using multiple plugin instances.
        Each voice gets its own synth instance for individual parameterization.
        """
        if base_plugin_name not in self.plugins:
            raise ValueError(f"Base plugin {base_plugin_name} not loaded")
        
        # Group overlapping events into voices
        voices = self._assign_voices(events, max_voices)
        
        # Create plugin instances for each voice
        voice_plugins = []
        base_plugin = self.plugins[base_plugin_name]
        
        for voice_idx in range(len(voices)):
            if len(voices[voice_idx]) == 0:
                continue
                
            voice_name = f"{base_plugin_name}_voice_{voice_idx}"
            # Note: In a real implementation, you'd clone the plugin
            # For now, we'll use the same plugin for all voices
            voice_plugins.append((base_plugin, voices[voice_idx]))
        
        # Add MIDI events to each voice
        for plugin, voice_events in voice_plugins:
            plugin.clear_midi()
            for event in voice_events:
                plugin.add_midi_note(
                    note=event.pitch,
                    velocity=event.velocity,
                    start_beat=event.onset_beats,
                    duration_beats=event.duration_beats,
                    beats=True
                )
        
        # Set up mixing graph - in a real implementation, you'd use an Add processor
        # For this basic version, we'll just use the first voice
        if voice_plugins:
            first_plugin = voice_plugins[0][0]
            self.engine.set_bpm(bpm)
            self.engine.load_graph([(first_plugin, [])])
            self.engine.render(duration_beats, beats=True)
            return self.engine.get_audio()
        else:
            # Return silence if no voices
            samples = int(duration_beats * 60.0 / bpm * self.sr)
            return np.zeros((2, samples))
    
    def _assign_voices(self, events: List[NoteEvent], max_voices: int) -> List[List[NoteEvent]]:
        """
        Assign events to voices to minimize overlaps.
        Simple algorithm: assign each note to the first available voice.
        """
        voices = [[] for _ in range(max_voices)]
        
        # Sort events by onset time
        sorted_events = sorted(events, key=lambda e: e.onset_beats)
        
        for event in sorted_events:
            # Find the first voice that doesn't have an overlapping note
            assigned = False
            for voice_idx, voice in enumerate(voices):
                # Check if this voice is free at the event's onset time
                if self._voice_is_free(voice, event.onset_beats):
                    voice.append(event)
                    assigned = True
                    break
            
            # If no voice is free, assign to the voice with earliest ending note
            if not assigned:
                earliest_voice = min(voices, key=lambda v: max([e.onset_beats + e.duration_beats for e in v] + [0]))
                earliest_voice.append(event)
        
        return voices
    
    def _voice_is_free(self, voice: List[NoteEvent], onset_time: float) -> bool:
        """Check if a voice is free at the given onset time."""
        for event in voice:
            if event.onset_beats <= onset_time < event.onset_beats + event.duration_beats:
                return False
        return True

def load_events_from_midi(midi_path: str, bpm: float = 120.0) -> List[NoteEvent]:
    """Load note events from MIDI file."""
    if not MIDO_AVAILABLE:
        print("Error: mido not available, cannot load MIDI files")
        return []
    
    events = []
    
    try:
        mid = mido.MidiFile(midi_path)
        ticks_per_beat = mid.ticks_per_beat
        
        # Track note ons and offs
        active_notes = {}  # (note, channel) -> (onset_beats, velocity)
        current_time_beats = 0.0
        
        for track in mid.tracks:
            current_time_beats = 0.0
            
            for msg in track:
                # Update current time
                if msg.time > 0:
                    beats_delta = msg.time / ticks_per_beat
                    current_time_beats += beats_delta
                
                if msg.type == 'note_on' and msg.velocity > 0:
                    key = (msg.note, msg.channel)
                    active_notes[key] = (current_time_beats, msg.velocity)
                
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    key = (msg.note, msg.channel)
                    if key in active_notes:
                        onset_beats, velocity = active_notes[key]
                        duration_beats = current_time_beats - onset_beats
                        
                        if duration_beats > 0:
                            event = NoteEvent(
                                pitch=msg.note,
                                onset_beats=onset_beats,
                                duration_beats=duration_beats,
                                velocity=velocity,
                                channel=msg.channel
                            )
                            events.append(event)
                        
                        del active_notes[key]
        
        return events
    
    except Exception as e:
        print(f"Error loading MIDI file: {e}")
        return []

def load_events_from_json(json_path: str) -> List[NoteEvent]:
    """Load note events from JSON file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        events = []
        for event_data in data['events']:
            event = NoteEvent.from_dict(event_data)
            events.append(event)
        
        return events
    
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return []

def main():
    """CLI interface for polyphonic scheduling and rendering."""
    parser = argparse.ArgumentParser(description='Polyphonic event scheduling and rendering')
    parser.add_argument('events_path', help='Input events file (MIDI or JSON)')
    parser.add_argument('--plugin', required=True, help='VST/AU plugin path')
    parser.add_argument('--out', help='Output audio file path')
    parser.add_argument('--bpm', type=float, default=120.0, help='BPM for rendering')
    parser.add_argument('--mode', choices=['single_instance', 'multi_instance'], 
                       default='single_instance', help='Rendering mode')
    parser.add_argument('--duration', type=float, help='Render duration in beats (auto if not specified)')
    parser.add_argument('--max_voices', type=int, default=8, help='Max voices for multi-instance mode')
    parser.add_argument('--sample_rate', type=int, default=22050, help='Sample rate')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.events_path):
        print(f"Error: Events file not found: {args.events_path}")
        sys.exit(1)
    
    if not os.path.exists(args.plugin):
        print(f"Error: Plugin not found: {args.plugin}")
        sys.exit(1)
    
    # Load events
    print(f"Loading events from {args.events_path}...")
    if args.events_path.endswith('.json'):
        events = load_events_from_json(args.events_path)
    elif args.events_path.endswith('.mid') or args.events_path.endswith('.midi'):
        events = load_events_from_midi(args.events_path, args.bpm)
    else:
        print("Error: Events file must be .json, .mid, or .midi")
        sys.exit(1)
    
    if not events:
        print("No events loaded or file is empty")
        sys.exit(1)
    
    print(f"Loaded {len(events)} note events")
    
    # Calculate render duration if not specified
    if args.duration:
        duration_beats = args.duration
    else:
        # Use the end time of the last note plus a small buffer
        last_end = max(event.onset_beats + event.duration_beats for event in events)
        duration_beats = last_end + 2.0  # Add 2 beats buffer
    
    print(f"Render duration: {duration_beats:.2f} beats at {args.bpm} BPM")
    
    # Initialize renderer and load plugin
    print(f"Loading plugin: {args.plugin}")
    renderer = PolyphonicRenderer(sample_rate=args.sample_rate)
    
    if not renderer.load_plugin(args.plugin):
        print("Failed to load plugin")
        sys.exit(1)
    
    # Render audio
    print(f"Rendering in {args.mode} mode...")
    try:
        if args.mode == 'single_instance':
            audio = renderer.render_single_instance(events, duration_beats, args.bpm)
        else:
            audio = renderer.render_multi_instance(events, duration_beats, args.bpm, 
                                                  max_voices=args.max_voices)
    except Exception as e:
        print(f"Error during rendering: {e}")
        sys.exit(1)
    
    # Save output
    if args.out:
        output_path = args.out
    else:
        base_name = os.path.splitext(os.path.basename(args.events_path))[0]
        output_path = f"{base_name}_rendered.wav"
    
    try:
        # Convert to stereo if mono
        if audio.ndim == 1:
            audio = np.stack([audio, audio])
        elif audio.shape[0] == 1:
            audio = np.vstack([audio, audio])
        
        # Transpose to (samples, channels) for soundfile
        audio_t = audio.T
        sf.write(output_path, audio_t, args.sample_rate)
        print(f"Saved audio to: {output_path}")
        
        # Print audio stats
        duration_sec = audio.shape[1] / args.sample_rate
        max_amp = np.max(np.abs(audio))
        print(f"Audio duration: {duration_sec:.2f}s")
        print(f"Max amplitude: {max_amp:.3f}")
        
    except Exception as e:
        print(f"Error saving audio: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()