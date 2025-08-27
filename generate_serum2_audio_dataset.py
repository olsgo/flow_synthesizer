#!/usr/bin/env python3
"""
Generate audio dataset from Serum 2 presets
Creates 20 audio samples with clear preset associations
"""

import os
import glob
import json
import numpy as np
from pathlib import Path
from serum2_parameter_mapper import Serum2ParameterMapper
import dawdreamer as daw

def generate_audio_dataset(output_dir: str = "/Users/gjb/Datasets/serum2/renders", num_samples: int = 20):
    """Generate audio dataset from Serum 2 presets"""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize the mapper
    plugin_path = "/Library/Audio/Plug-Ins/VST3/Serum2.vst3"
    mapper = Serum2ParameterMapper(plugin_path)
    
    # Find all converted JSON presets
    preset_dir = "converted_presets"
    preset_files = []
    for root, dirs, files in os.walk(preset_dir):
        for file in files:
            if file.endswith('.json'):
                preset_files.append(os.path.join(root, file))
    
    if not preset_files:
        print(f"No preset files found in {preset_dir}")
        return
    
    # Limit to requested number of samples
    preset_files = preset_files[:num_samples]
    
    print(f"Generating audio dataset with {len(preset_files)} presets")
    print(f"Output directory: {output_dir}")
    
    # Create metadata file
    metadata = {
        "dataset_info": {
            "total_samples": len(preset_files),
            "sample_rate": 44100,
            "duration_seconds": 4.0,
            "plugin": "Serum 2",
            "generation_method": "DawDreamer with parameter mapping"
        },
        "samples": []
    }
    
    successful_renders = 0
    
    for i, preset_path in enumerate(preset_files, 1):
        preset_name = Path(preset_path).stem
        safe_name = "".join(c for c in preset_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_name = safe_name.replace(' ', '_')
        
        print(f"\n[{i}/{len(preset_files)}] Processing: {preset_name}")
        
        try:
            # Load preset parameters
            parameters_set = mapper.load_preset_with_mapping(preset_path, verbose=False)
            print(f"  Parameters set: {parameters_set}")
            
            if parameters_set == 0:
                print(f"  ⚠️  No parameters set, skipping audio generation")
                continue
            
            # Generate audio with a single note first to test
            note = 60  # C4
            duration = 3.0  # 3 seconds
            velocity = 100
            
            # Clear any existing MIDI
            mapper.synth.clear_midi()
            
            # Add a single note
            mapper.synth.add_midi_note(note, velocity, 0.0, duration)
            
            # Render audio with longer duration to allow for release
            render_duration = duration + 1.0  # Extra time for release
            mapper.engine.render(render_duration)
            audio = mapper.engine.get_audio()
            
            # Clear MIDI after rendering
            mapper.synth.clear_midi()
            
            # Check if audio was generated
            if audio.size == 0:
                print(f"  ❌ No audio generated")
                continue
                
            max_amplitude = np.max(np.abs(audio))
            if max_amplitude < 1e-6:
                print(f"  ❌ Audio too quiet (max: {max_amplitude:.2e})")
                continue
            
            # Normalize audio to prevent clipping
            if max_amplitude > 0.95:
                audio = audio * (0.95 / max_amplitude)
                max_amplitude = 0.95
            
            print(f"  ✅ Audio generated (max amplitude: {max_amplitude:.4f})")
            
            # Save audio file
            audio_filename = f"{safe_name}_{i:03d}.wav"
            audio_path = output_path / audio_filename
            
            # Save using scipy exactly like the XML example (no int16 conversion)
            try:
                from scipy.io import wavfile
                wavfile.write(str(audio_path), 44100, audio.transpose())
            except ImportError:
                # Fallback: convert to int16 only if scipy is not available
                audio_int16 = (audio * 32767).astype(np.int16)
                import wave
                import struct
                
                with wave.open(str(audio_path), 'wb') as wav_file:
                    wav_file.setnchannels(2 if len(audio_int16.shape) > 1 else 1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(44100)
                    
                    if len(audio_int16.shape) > 1:
                        # Stereo
                        for frame in audio_int16.transpose():
                            wav_file.writeframes(struct.pack('<hh', frame[0], frame[1]))
                    else:
                        # Mono
                        for sample in audio_int16:
                            wav_file.writeframes(struct.pack('<h', sample))
            
            print(f"  💾 Saved: {audio_filename}")
            
            # Add to metadata
            sample_metadata = {
                "id": i,
                "audio_file": audio_filename,
                "preset_name": preset_name,
                "preset_path": preset_path,
                "parameters_set": parameters_set,
                "max_amplitude": float(max_amplitude),
                "notes_played": notes,
                "duration_seconds": duration
            }
            
            metadata["samples"].append(sample_metadata)
            successful_renders += 1
            
        except Exception as e:
            print(f"  ❌ Error processing {preset_name}: {str(e)}")
            continue
    
    # Save metadata
    metadata_path = output_path / "dataset_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"DATASET GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Successfully rendered: {successful_renders}/{len(preset_files)} presets")
    print(f"Output directory: {output_dir}")
    print(f"Metadata file: {metadata_path}")
    
    if successful_renders > 0:
        print(f"\nGenerated files:")
        for sample in metadata["samples"]:
            print(f"  - {sample['audio_file']} (from {sample['preset_name']})")
    
    return metadata

if __name__ == "__main__":
    generate_audio_dataset()