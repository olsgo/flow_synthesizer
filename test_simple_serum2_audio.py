#!/usr/bin/env python3
"""
Simple test script to generate clean audio from Serum 2 without parameter mapping.
This helps isolate audio rendering issues from parameter mapping complexity.
"""

import dawdreamer as daw
import numpy as np
from scipy.io import wavfile
from pathlib import Path

def test_simple_serum2_audio():
    """Test basic Serum 2 audio generation without preset loading."""
    
    # Plugin path
    plugin_path = "/Library/Audio/Plug-Ins/VST3/Serum2.vst3"
    
    # Create output directory
    output_dir = Path("/Users/gjb/Datasets/serum2/test_renders")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Initializing DawDreamer engine...")
    
    # Initialize engine with different settings to test
    configs = [
        {"sample_rate": 44100, "buffer_size": 512, "name": "default"},
        {"sample_rate": 44100, "buffer_size": 256, "name": "small_buffer"},
        {"sample_rate": 44100, "buffer_size": 1024, "name": "large_buffer"},
        {"sample_rate": 48000, "buffer_size": 512, "name": "high_sr"},
    ]
    
    for config in configs:
        print(f"\nTesting config: {config['name']}")
        
        try:
            # Create engine
            engine = daw.RenderEngine(config["sample_rate"], config["buffer_size"])
            synth = engine.make_plugin_processor("serum2", plugin_path)
            
            # Load graph
            graph = [(synth, [])]
            engine.load_graph(graph)
            
            print(f"  Engine initialized: {config['sample_rate']}Hz, buffer={config['buffer_size']}")
            
            # Test different note configurations
            test_cases = [
                {"note": 60, "velocity": 80, "duration": 2.0, "name": "c4_normal"},
                {"note": 48, "velocity": 100, "duration": 1.5, "name": "c3_loud"},
                {"note": 72, "velocity": 60, "duration": 3.0, "name": "c5_soft"},
            ]
            
            for test_case in test_cases:
                print(f"    Testing: {test_case['name']}")
                
                # Clear any existing MIDI
                synth.clear_midi()
                
                # Add note
                synth.add_midi_note(
                    test_case["note"], 
                    test_case["velocity"], 
                    0.0, 
                    test_case["duration"]
                )
                
                # Render with extra time for release
                render_duration = test_case["duration"] + 2.0
                engine.render(render_duration)
                
                # Get audio
                audio = engine.get_audio()
                
                # Clear MIDI
                synth.clear_midi()
                
                # Check audio
                if audio.size == 0:
                    print(f"      ❌ No audio generated")
                    continue
                    
                max_amplitude = np.max(np.abs(audio))
                if max_amplitude < 1e-6:
                    print(f"      ❌ Audio too quiet: {max_amplitude:.2e}")
                    continue
                    
                print(f"      ✅ Audio generated: max={max_amplitude:.4f}, shape={audio.shape}")
                
                # Save audio file
                filename = f"{config['name']}_{test_case['name']}.wav"
                filepath = output_dir / filename
                
                # Save using scipy (like XML example)
                wavfile.write(str(filepath), config["sample_rate"], audio.transpose())
                print(f"      💾 Saved: {filename}")
                
        except Exception as e:
            print(f"  ❌ Error with config {config['name']}: {e}")
            continue
    
    print(f"\n🎵 Test complete! Check audio files in: {output_dir}")

if __name__ == "__main__":
    test_simple_serum2_audio()