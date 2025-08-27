#!/usr/bin/env python3
"""
Generate clean Serum 2 audio dataset using default presets without parameter mapping.
This bypasses the parameter mapping issues to create a working audio dataset.
"""

import os
import glob
import json
import numpy as np
from pathlib import Path
import dawdreamer as daw
from scipy.io import wavfile

def generate_clean_audio_dataset(output_dir: str = "/Users/gjb/Datasets/serum2/clean_renders", num_samples: int = 20):
    """Generate audio dataset using Serum 2 default sound with different MIDI patterns."""
    
    # Plugin path
    plugin_path = "/Library/Audio/Plug-Ins/VST3/Serum2.vst3"
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating clean Serum 2 audio dataset...")
    print(f"Output directory: {output_path}")
    
    # Initialize DawDreamer
    engine = daw.RenderEngine(44100, 512)
    synth = engine.make_plugin_processor("serum2", plugin_path)
    
    # Load the graph
    graph = [(synth, [])]
    engine.load_graph(graph)
    
    print(f"Serum 2 plugin loaded successfully")
    
    # Define different musical patterns to create variety
    patterns = [
        # Single notes
        {"name": "C4_single", "notes": [(60, 0.0, 2.0)], "velocity": 80},
        {"name": "G4_single", "notes": [(67, 0.0, 2.0)], "velocity": 80},
        {"name": "C5_single", "notes": [(72, 0.0, 2.0)], "velocity": 80},
        
        # Chords
        {"name": "C_major", "notes": [(60, 0.0, 3.0), (64, 0.0, 3.0), (67, 0.0, 3.0)], "velocity": 70},
        {"name": "F_major", "notes": [(65, 0.0, 3.0), (69, 0.0, 3.0), (72, 0.0, 3.0)], "velocity": 70},
        {"name": "G_major", "notes": [(67, 0.0, 3.0), (71, 0.0, 3.0), (74, 0.0, 3.0)], "velocity": 70},
        
        # Arpeggios
        {"name": "C_arp_up", "notes": [(60, 0.0, 0.5), (64, 0.5, 0.5), (67, 1.0, 0.5), (72, 1.5, 1.0)], "velocity": 85},
        {"name": "C_arp_down", "notes": [(72, 0.0, 0.5), (67, 0.5, 0.5), (64, 1.0, 0.5), (60, 1.5, 1.0)], "velocity": 85},
        
        # Bass notes
        {"name": "C2_bass", "notes": [(36, 0.0, 2.5)], "velocity": 100},
        {"name": "F2_bass", "notes": [(41, 0.0, 2.5)], "velocity": 100},
        
        # High notes
        {"name": "C6_high", "notes": [(84, 0.0, 1.5)], "velocity": 60},
        {"name": "G6_high", "notes": [(91, 0.0, 1.5)], "velocity": 60},
        
        # Velocity variations
        {"name": "C4_soft", "notes": [(60, 0.0, 2.0)], "velocity": 40},
        {"name": "C4_loud", "notes": [(60, 0.0, 2.0)], "velocity": 127},
        
        # Rhythmic patterns
        {"name": "rhythm_1", "notes": [(60, 0.0, 0.25), (60, 0.5, 0.25), (64, 1.0, 0.5), (67, 2.0, 1.0)], "velocity": 80},
        {"name": "rhythm_2", "notes": [(48, 0.0, 0.5), (60, 0.25, 0.25), (64, 0.75, 0.25), (67, 1.5, 0.5)], "velocity": 90},
        
        # Extended chords
        {"name": "Cmaj7", "notes": [(60, 0.0, 3.0), (64, 0.0, 3.0), (67, 0.0, 3.0), (71, 0.0, 3.0)], "velocity": 65},
        {"name": "Dm7", "notes": [(62, 0.0, 3.0), (65, 0.0, 3.0), (69, 0.0, 3.0), (72, 0.0, 3.0)], "velocity": 65},
        
        # Octaves
        {"name": "C_octaves", "notes": [(48, 0.0, 2.0), (60, 0.0, 2.0), (72, 0.0, 2.0)], "velocity": 75},
        {"name": "G_octaves", "notes": [(55, 0.0, 2.0), (67, 0.0, 2.0), (79, 0.0, 2.0)], "velocity": 75},
    ]
    
    # Limit to requested number of samples
    patterns = patterns[:num_samples]
    
    # Metadata for the dataset
    dataset_metadata = {
        "total_samples": len(patterns),
        "plugin": "Serum 2",
        "sample_rate": 44100,
        "format": "WAV",
        "generation_method": "default_preset_midi_patterns",
        "samples": []
    }
    
    # Generate audio for each pattern
    for i, pattern in enumerate(patterns, 1):
        print(f"\n[{i}/{len(patterns)}] Generating: {pattern['name']}")
        
        try:
            # Clear any existing MIDI
            synth.clear_midi()
            
            # Add all notes in the pattern
            max_end_time = 0
            for note, start_time, duration in pattern["notes"]:
                synth.add_midi_note(note, pattern["velocity"], start_time, duration)
                max_end_time = max(max_end_time, start_time + duration)
                print(f"  Added note: {note} at {start_time}s for {duration}s")
            
            # Render with extra time for release
            render_duration = max_end_time + 2.0
            engine.render(render_duration)
            audio = engine.get_audio()
            
            # Clear MIDI after rendering
            synth.clear_midi()
            
            # Check if audio was generated
            if audio.size == 0:
                print(f"  ❌ No audio generated")
                continue
                
            max_amplitude = np.max(np.abs(audio))
            if max_amplitude < 1e-6:
                print(f"  ❌ Audio too quiet (max: {max_amplitude:.2e})")
                continue
            
            # Normalize if too loud
            if max_amplitude > 0.95:
                audio = audio * (0.95 / max_amplitude)
                max_amplitude = 0.95
            
            print(f"  ✅ Audio generated (max amplitude: {max_amplitude:.4f})")
            
            # Save audio file
            audio_filename = f"{pattern['name']}_{i:03d}.wav"
            audio_path = output_path / audio_filename
            
            # Save using scipy (like XML example)
            wavfile.write(str(audio_path), 44100, audio.transpose())
            print(f"  💾 Saved: {audio_filename}")
            
            # Add to metadata
            sample_metadata = {
                "filename": audio_filename,
                "pattern_name": pattern["name"],
                "notes": pattern["notes"],
                "velocity": pattern["velocity"],
                "max_amplitude": float(max_amplitude),
                "duration_seconds": float(render_duration),
                "audio_shape": list(audio.shape)
            }
            dataset_metadata["samples"].append(sample_metadata)
            
        except Exception as e:
            print(f"  ❌ Error generating {pattern['name']}: {e}")
            continue
    
    # Save metadata
    metadata_path = output_path / "clean_dataset_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(dataset_metadata, f, indent=2)
    
    print(f"\n🎵 Dataset generation complete!")
    print(f"Generated {len(dataset_metadata['samples'])} audio files")
    print(f"Output directory: {output_path}")
    print(f"Metadata saved: {metadata_path}")

if __name__ == "__main__":
    generate_clean_audio_dataset()