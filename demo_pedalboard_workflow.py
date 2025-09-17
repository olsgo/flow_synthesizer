#!/usr/bin/env python3
"""
Demonstration script showing how to use the Pedalboard migration functionality.

This script shows practical examples of:
1. Loading plugins and capturing initial states
2. Parameter manipulation and state management
3. Rendering audio with different presets
4. Round-trip binary state capture and restore

Usage:
    # Basic demo with mock data (no real plugins needed)
    python demo_pedalboard_workflow.py --demo

    # Real plugin demo (requires actual plugins)
    python demo_pedalboard_workflow.py --plugin /path/to/plugin.vst3
"""

import argparse
import json
import os
import sys
import tempfile
import numpy as np
from pathlib import Path

# Add the code directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))

def demo_mock_workflow():
    """Demonstrate the workflow with mock data (no real plugins required)."""
    
    print("🎭 Demo: Pedalboard Workflow (Mock Mode)")
    print("="*60)
    
    from pedalboard_renderer import PBRenderer
    
    # Initialize renderer
    print("1. Initializing PBRenderer...")
    renderer = PBRenderer(sample_rate=22050, buffer_size=512)
    print(f"   ✓ Sample rate: {renderer.sr} Hz")
    print(f"   ✓ Buffer size: {renderer.buffer_size} samples")
    
    # Show API methods available
    print("\n2. Available methods:")
    methods = [m for m in dir(renderer) if not m.startswith('_') and callable(getattr(renderer, m))]
    for method in methods:
        print(f"   • {method}()")
    
    # Demonstrate parameter handling
    print("\n3. Parameter format demonstration:")
    mock_params = [
        (0, 0.5),   # Oscillator 1 Volume: 50%
        (1, 0.8),   # Filter Cutoff: 80%
        (2, 0.2),   # Envelope Attack: 20%
        (3, 1.0),   # Master Volume: 100%
    ]
    
    params_json = json.dumps(mock_params)
    print(f"   Parameters as JSON: {params_json}")
    
    # Show parameter dictionary format
    param_dict = {f"param_{idx}": val for idx, val in mock_params}
    print(f"   Parameters as dict: {json.dumps(param_dict, indent=2)}")
    
    # Binary state simulation
    print("\n4. Binary state management:")
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
        # Simulate binary state data
        mock_state = b"MOCK_PLUGIN_STATE_" + json.dumps(mock_params).encode()
        tmp_file.write(mock_state)
        tmp_file.flush()
        
        print(f"   ✓ Mock state saved: {tmp_file.name}")
        print(f"   ✓ State size: {len(mock_state)} bytes")
        
        # Read it back
        with open(tmp_file.name, 'rb') as f:
            loaded_state = f.read()
        
        if loaded_state == mock_state:
            print("   ✓ Binary state round-trip successful!")
        
        # Cleanup
        os.unlink(tmp_file.name)
    
    print("\n5. Output structure demonstration:")
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "demo_output"
        output_dir.mkdir()
        
        # Create mock output files
        files_created = []
        
        # Metadata CSV
        metadata_file = output_dir / "metadata.csv"
        with open(metadata_file, 'w') as f:
            f.write("preset,audio,params_json,state_bin,plugin_path\n")
            f.write("demo.json,demo.wav,\"[[0,0.5],[1,0.8]]\",demo.bin,mock_plugin.vst3\n")
        files_created.append(metadata_file)
        
        # Parameter index map
        param_map = {"oscillator_volume": 0, "filter_cutoff": 1, "envelope_attack": 2}
        param_file = output_dir / "parameter_index_map.json"
        with open(param_file, 'w') as f:
            json.dump(param_map, f, indent=2)
        files_created.append(param_file)
        
        # Mock audio file
        audio_file = output_dir / "demo.wav"
        with open(audio_file, 'wb') as f:
            f.write(b"RIFF....WAVE")  # Mock WAV header
        files_created.append(audio_file)
        
        # Mock binary state
        state_file = output_dir / "demo.bin"
        with open(state_file, 'wb') as f:
            f.write(mock_state)
        files_created.append(state_file)
        
        print(f"   Output directory: {output_dir}")
        for file_path in files_created:
            relative_path = file_path.relative_to(output_dir)
            size = file_path.stat().st_size
            print(f"   ✓ {relative_path} ({size} bytes)")
    
    print(f"\n✅ Demo completed successfully!")
    print(f"💡 Use --plugin flag to test with real plugins.")

def demo_real_plugin(plugin_path):
    """Demonstrate the workflow with a real plugin."""
    
    print(f"🎵 Demo: Pedalboard Workflow (Real Plugin)")
    print("="*60)
    print(f"Plugin: {plugin_path}")
    
    from pedalboard_renderer import PBRenderer
    
    try:
        # Initialize and load plugin
        print("\n1. Loading plugin...")
        renderer = PBRenderer(sample_rate=22050, buffer_size=512)
        
        if not renderer.load_plugin(plugin_path):
            print(f"❌ Failed to load plugin: {plugin_path}")
            return False
        
        plugin_name = Path(plugin_path).stem
        print(f"   ✓ Plugin loaded: {plugin_name}")
        
        # Get parameter information
        print("\n2. Analyzing plugin parameters...")
        params = renderer.get_parameters_description()
        param_count = renderer.get_plugin_parameter_size()
        
        print(f"   ✓ Total parameters: {param_count}")
        
        # Show first few parameters
        for i, param in enumerate(params[:5]):
            name = param.get('name', f'Param_{i}')
            range_info = param.get('range', [0, 1])
            default = param.get('default', 0.5)
            print(f"   • {i}: {name} (range: {range_info}, default: {default})")
        
        if len(params) > 5:
            print(f"   ... and {len(params) - 5} more parameters")
        
        # Capture initial state
        print("\n3. Capturing initial state...")
        with tempfile.NamedTemporaryFile(suffix='_init.bin', delete=False) as tmp_file:
            if renderer.save_state(tmp_file.name):
                state_size = Path(tmp_file.name).stat().st_size
                print(f"   ✓ Initial state saved: {tmp_file.name}")
                print(f"   ✓ State size: {state_size} bytes")
                
                # Test state round-trip
                print("\n4. Testing state round-trip...")
                if renderer.load_state(tmp_file.name):
                    print("   ✓ State loaded successfully")
                    
                    # Verify state by saving again and comparing
                    with tempfile.NamedTemporaryFile(suffix='_verify.bin', delete=False) as verify_file:
                        if renderer.save_state(verify_file.name):
                            original_size = Path(tmp_file.name).stat().st_size
                            verify_size = Path(verify_file.name).stat().st_size
                            
                            if original_size == verify_size:
                                print("   ✓ State round-trip verified (sizes match)")
                            else:
                                print(f"   ⚠️  State sizes differ: {original_size} vs {verify_size}")
                        
                        os.unlink(verify_file.name)
                else:
                    print("   ❌ Failed to load state")
                
                os.unlink(tmp_file.name)
            else:
                print("   ❌ Failed to save initial state")
        
        # Get current parameter values
        print("\n5. Current parameter values...")
        current_patch = renderer.get_patch()
        print(f"   ✓ Retrieved {len(current_patch)} parameter values")
        
        # Show a few parameter values
        for i, (idx, value) in enumerate(current_patch[:3]):
            param_name = params[idx].get('name', f'Param_{idx}') if idx < len(params) else f'Param_{idx}'
            print(f"   • {idx}: {param_name} = {value:.3f}")
        
        # Test parameter modification
        print("\n6. Testing parameter modification...")
        if current_patch:
            # Modify first parameter
            modified_patch = current_patch.copy()
            if modified_patch:
                old_value = modified_patch[0][1]
                new_value = 0.7 if old_value < 0.5 else 0.3
                modified_patch[0] = (modified_patch[0][0], new_value)
                
                renderer.set_patch([modified_patch[0]])
                
                # Verify the change
                updated_patch = renderer.get_patch()
                if updated_patch and abs(updated_patch[0][1] - new_value) < 0.01:
                    print(f"   ✓ Parameter modified: {old_value:.3f} → {new_value:.3f}")
                else:
                    print(f"   ⚠️  Parameter modification may not have worked as expected")
        
        # Test rendering
        print("\n7. Testing audio rendering...")
        try:
            audio = renderer.render_patch(midi_note=60, note_len_sec=1.0, render_len_sec=2.0)
            if audio is not None and audio.size > 0:
                channels, samples = audio.shape
                duration = samples / renderer.sr
                print(f"   ✓ Audio rendered: {channels} channels, {samples} samples ({duration:.2f}s)")
                
                # Check if audio contains signal (not just silence)
                max_amplitude = np.max(np.abs(audio))
                if max_amplitude > 0.001:
                    print(f"   ✓ Audio signal detected (max: {max_amplitude:.3f})")
                else:
                    print(f"   ⚠️  Audio appears to be silent (max: {max_amplitude:.6f})")
            else:
                print("   ❌ Audio rendering failed or returned empty audio")
        except Exception as e:
            print(f"   ❌ Audio rendering error: {e}")
        
        print(f"\n✅ Real plugin demo completed!")
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Demonstrate Pedalboard workflow")
    parser.add_argument("--demo", action="store_true", help="Run mock demo (no plugins required)")
    parser.add_argument("--plugin", help="Path to real plugin for demonstration")
    
    args = parser.parse_args()
    
    if not args.demo and not args.plugin:
        print("❌ Must specify either --demo or --plugin")
        parser.print_help()
        return 1
    
    success = True
    
    if args.demo:
        demo_mock_workflow()
    
    if args.plugin:
        if not os.path.exists(args.plugin):
            print(f"❌ Plugin file not found: {args.plugin}")
            return 1
        
        if not demo_real_plugin(args.plugin):
            success = False
    
    if success:
        print(f"\n🎉 All demonstrations completed successfully!")
        print(f"\n📚 Next steps:")
        print(f"   • Run: python capture_init_state.py --plugins MyPlugin.vst3")
        print(f"   • Run: python code/render_dataset_pb.py plugin.vst3 presets/ output/")
        print(f"   • Read: docs/pedalboard_migration.md")
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())