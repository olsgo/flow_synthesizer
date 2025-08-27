#!/usr/bin/env python3
"""
Demonstration script for opsix synthesizer integration.
This script shows how to use the opsix synthesizer with the flow_synthesizer project.
"""

import sys
sys.path.append('code')

import numpy as np
import json
import soundfile as sf
from synth.synthesize import create_synth, synthesize_audio

def demo_opsix_synthesis():
    """Demonstrate opsix synthesizer usage."""
    print("🎹 Opsix Synthesizer Demo")
    print("=" * 40)
    
    # Create opsix synthesizer
    print("Loading opsix synthesizer...")
    engine, param_defaults, rev_idx = create_synth(dataset="toy", synth_type="opsix")
    print(f"✓ Loaded opsix with {len(param_defaults)} parameters")
    
    # Load important parameters
    with open('code/synth/opsix_important_params.json') as f:
        important_params = json.load(f)
    print(f"✓ Using {len(important_params)} important parameters for synthesis")
    
    # Create some example parameter sets
    examples = [
        {"name": "Default", "params": {param: param_defaults[param] for param in important_params[:8]}},
        {"name": "Random 1", "params": {param: np.random.rand() for param in important_params[:8]}},
        {"name": "Random 2", "params": {param: np.random.rand() for param in important_params[:8]}},
    ]
    
    # Synthesize audio for each example
    for i, example in enumerate(examples):
        print(f"\nSynthesizing '{example['name']}'...")
        
        # Generate audio
        audio = synthesize_audio(example['params'], engine, rev_idx)
        
        # Save audio file
        filename = f"opsix_demo_{i+1}_{example['name'].lower().replace(' ', '_')}.wav"
        sf.write(filename, audio, 44100)
        
        print(f"✓ Generated {len(audio)} samples")
        print(f"✓ Audio range: [{np.min(audio):.3f}, {np.max(audio):.3f}]")
        print(f"✓ Saved as: {filename}")
        
        # Show some parameter values
        print(f"  Parameters used:")
        for param, value in list(example['params'].items())[:3]:
            print(f"    {param}: {value:.3f}")
        if len(example['params']) > 3:
            print(f"    ... and {len(example['params']) - 3} more")
    
    print(f"\n🎉 Demo complete! Generated {len(examples)} audio files.")
    print("\nTo use opsix in your own code:")
    print("```python")
    print("from synth.synthesize import create_synth, synthesize_audio")
    print("engine, defaults, rev_idx = create_synth(dataset='toy', synth_type='opsix')")
    print("audio = synthesize_audio(your_params, engine, defaults)")
    print("```")

if __name__ == "__main__":
    demo_opsix_synthesis()