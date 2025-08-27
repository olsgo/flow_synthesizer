#!/usr/bin/env python3
"""
Corrected Serum 2 Audio Dataset Generator
Uses proper parameter mapping to apply preset parameters without glitchy audio.
"""

import os
import json
import numpy as np
from pathlib import Path
import dawdreamer as daw
from scipy.io import wavfile
from serum2_parameter_mapping import map_preset_to_vst_parameters

def generate_corrected_dataset():
    """Generate audio dataset with corrected parameter mapping."""
    
    # Configuration
    plugin_path = "/Library/Audio/Plug-Ins/VST3/Serum2.vst3"
    preset_dir = Path("converted_presets")
    output_dir = Path("/Users/gjb/Datasets/serum2/corrected_renders")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Audio generation settings
    sample_rate = 44100
    buffer_size = 512
    note_duration = 3.0
    render_duration = 4.0  # Extra time for release
    
    print("🎵 Corrected Serum 2 Dataset Generation")
    print("=" * 50)
    
    # Find all preset files
    preset_files = list(preset_dir.glob("**/*.json"))
    print(f"Found {len(preset_files)} preset files")
    
    # Limit to first 10 for testing
    preset_files = preset_files[:10]
    print(f"Processing first {len(preset_files)} presets for testing")
    
    # Initialize DawDreamer engine
    engine = daw.RenderEngine(sample_rate, buffer_size)
    synth = engine.make_plugin_processor("serum2", plugin_path)
    graph = [(synth, [])]
    engine.load_graph(graph)
    
    # Get VST parameter mapping
    vst_params = synth.get_parameters_description()
    vst_param_name_to_index = {param["name"]: param["index"] for param in vst_params}
    
    print(f"VST has {len(vst_params)} parameters available")
    
    # Process each preset
    successful_renders = 0
    failed_renders = 0
    metadata = []
    
    for i, preset_file in enumerate(preset_files):
        print(f"\n[{i+1}/{len(preset_files)}] Processing: {preset_file.name}")
        
        try:
            # Load preset
            with open(preset_file, 'r') as f:
                preset_json = json.load(f)
            
            # Extract parameters from dawdreamer_params section
            preset_data = preset_json.get('dawdreamer_params', {})
            
            if not preset_data:
                print(f"  ⚠️  No dawdreamer_params found, skipping")
                failed_renders += 1
                continue
            
            # Map preset parameters to VST parameters
            mapped_params = map_preset_to_vst_parameters(preset_data, vst_param_name_to_index)
            
            print(f"  📊 Mapped {len(mapped_params)} parameters")
            
            # Reset plugin to default state
            engine = daw.RenderEngine(sample_rate, buffer_size)
            synth = engine.make_plugin_processor("serum2", plugin_path)
            graph = [(synth, [])]
            engine.load_graph(graph)
            
            # Apply mapped parameters
            applied_count = 0
            for param_index, param_value in mapped_params:
                try:
                    # Additional validation
                    if 0.0 <= param_value <= 1.0:
                        synth.set_parameter(param_index, param_value)
                        applied_count += 1
                    else:
                        print(f"    ⚠️  Invalid parameter value: {param_value}")
                except Exception as e:
                    print(f"    ❌ Error setting parameter {param_index}: {e}")
            
            print(f"  ✅ Applied {applied_count}/{len(mapped_params)} parameters")
            
            # Generate audio with single note
            synth.clear_midi()
            synth.add_midi_note(60, 80, 0.0, note_duration)  # C4, velocity 80
            
            # Render audio
            engine.render(render_duration)
            audio = engine.get_audio()
            synth.clear_midi()
            
            # Check audio quality
            if audio.size == 0:
                print(f"  ❌ No audio generated")
                failed_renders += 1
                continue
            
            max_amplitude = np.max(np.abs(audio))
            print(f"  🔊 Audio: max amplitude = {max_amplitude:.4f}")
            
            # Check for problematic audio
            if max_amplitude > 2.0 or np.isnan(max_amplitude) or np.isinf(max_amplitude):
                print(f"  ❌ Problematic audio detected (max={max_amplitude:.4f})")
                failed_renders += 1
                continue
            
            if max_amplitude < 0.001:
                print(f"  ⚠️  Very quiet audio (max={max_amplitude:.4f})")
            
            # Save audio
            preset_name = preset_file.stem.replace(" ", "_")
            output_filename = f"{preset_name}_{i+1:03d}.wav"
            output_path = output_dir / output_filename
            
            try:
                # Use scipy for saving (no explicit int16 conversion)
                wavfile.write(str(output_path), sample_rate, audio.transpose())
                print(f"  💾 Saved: {output_filename}")
                
                # Store metadata
                metadata.append({
                    "file": output_filename,
                    "preset_name": preset_json.get('presetName', preset_file.stem),
                    "preset_author": preset_json.get('presetAuthor', 'Unknown'),
                    "preset_file": str(preset_file),
                    "mapped_parameters": len(mapped_params),
                    "applied_parameters": applied_count,
                    "max_amplitude": float(max_amplitude),
                    "note": "C4",
                    "duration": note_duration,
                    "sample_rate": sample_rate
                })
                
                successful_renders += 1
                
            except Exception as e:
                print(f"  ❌ Error saving audio: {e}")
                failed_renders += 1
                
        except Exception as e:
            print(f"  ❌ Error processing preset: {e}")
            failed_renders += 1
    
    # Save metadata
    metadata_file = output_dir / "corrected_dataset_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump({
            "total_presets_processed": len(preset_files),
            "successful_renders": successful_renders,
            "failed_renders": failed_renders,
            "output_directory": str(output_dir),
            "generation_settings": {
                "sample_rate": sample_rate,
                "buffer_size": buffer_size,
                "note_duration": note_duration,
                "render_duration": render_duration,
                "note": "C4 (MIDI 60)",
                "velocity": 80
            },
            "files": metadata
        }, f, indent=2)
    
    print(f"\n🎵 Dataset generation complete!")
    print(f"Successful renders: {successful_renders}")
    print(f"Failed renders: {failed_renders}")
    print(f"Success rate: {successful_renders/(successful_renders+failed_renders)*100:.1f}%")
    print(f"Output directory: {output_dir}")
    print(f"Metadata saved: {metadata_file}")

if __name__ == "__main__":
    generate_corrected_dataset()