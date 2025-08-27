#!/usr/bin/env python3
"""
Test script to verify opsix integration with the flow_synthesizer project.
"""

import sys
sys.path.append('code')

import numpy as np
import json
from synth.synthesize import create_synth, synthesize_audio, synthesize_batch

def test_opsix_integration():
    """Test opsix synthesizer integration."""
    print("Testing opsix integration...")
    
    # Create opsix synthesizer
    try:
        engine, param_defaults, rev_idx = create_synth(dataset="toy", synth_type="opsix")
        print(f"✓ Opsix engine created successfully")
        print(f"✓ Parameter defaults loaded: {len(param_defaults)} parameters")
        print(f"✓ Reverse index created: {len(rev_idx)} mappings")
    except Exception as e:
        print(f"✗ Failed to create opsix engine: {e}")
        return False
    
    # Load important parameters for testing
    try:
        with open('code/synth/opsix_important_params.json') as f:
            important_params = json.load(f)
        print(f"✓ Important parameters loaded: {len(important_params)} parameters")
    except Exception as e:
        print(f"✗ Failed to load important parameters: {e}")
        return False
    
    # Test single audio synthesis
    try:
        # Create a test parameter set (using defaults)
        test_params = {param: param_defaults[param] for param in important_params[:16]}
        
        # Synthesize audio
        audio = synthesize_audio(test_params, engine, rev_idx)
        print(f"✓ Single audio synthesis successful: {len(audio)} samples")
        print(f"  Audio range: [{np.min(audio):.3f}, {np.max(audio):.3f}]")
    except Exception as e:
        print(f"✗ Single audio synthesis failed: {e}")
        return False
    
    # Test batch synthesis
    try:
        # Create a batch of random parameter values
        batch_size = 4
        param_count = min(16, len(important_params))
        
        # Generate random parameter values (0-1 range)
        batch_params = np.random.rand(batch_size, param_count)
        
        # Synthesize batch
        audio_batch = synthesize_batch(
            batch_params,
            important_params[:param_count],
            engine,
            param_defaults,
            rev_idx,
            n_outs=2
        )
        
        print(f"✓ Batch synthesis successful: {len(audio_batch)} audio files")
        for i, audio in enumerate(audio_batch[:2]):
            print(f"  Audio {i}: {len(audio)} samples, range [{np.min(audio):.3f}, {np.max(audio):.3f}]")
            
    except Exception as e:
        print(f"✗ Batch synthesis failed: {e}")
        return False
    
    print("\n🎉 All opsix integration tests passed!")
    return True

def test_diva_compatibility():
    """Test that Diva still works after changes."""
    print("\nTesting Diva compatibility...")
    
    try:
        # Test Diva with new interface
        engine, param_defaults, rev_idx = create_synth(dataset="toy", synth_type="diva")
        print(f"✓ Diva engine still works: {len(param_defaults)} parameters")
        
        # Test default behavior (should still be Diva)
        engine2, param_defaults2, rev_idx2 = create_synth(dataset="toy")
        print(f"✓ Default behavior preserved: {len(param_defaults2)} parameters")
        
        return True
    except Exception as e:
        print(f"✗ Diva compatibility test failed: {e}")
        return False

if __name__ == "__main__":
    success = True
    
    # Test opsix integration
    success &= test_opsix_integration()
    
    # Test Diva compatibility
    success &= test_diva_compatibility()
    
    if success:
        print("\n🎊 All tests passed! Opsix integration is ready.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)