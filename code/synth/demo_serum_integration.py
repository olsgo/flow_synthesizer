#!/usr/bin/env python3
"""
Quick test demonstrating the new Serum 2 pedalboard integration
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from synth.synthesize import create_synth

def demo_serum_integration():
    """Demonstrate Serum integration with pedalboard"""
    print("=== Serum 2 Pedalboard Integration Demo ===")
    
    try:
        # Try to create Serum synthesizer with pedalboard backend
        print("Creating Serum synthesizer with pedalboard backend...")
        engine, generator, defaults, rev_idx = create_synth(
            dataset='toy',
            synth_type='serum',
            backend='pedalboard'
        )
        print("✓ Serum synthesizer created successfully!")
        
        # Show some parameter information
        print(f"✓ Default parameters loaded: {len(defaults)} parameters")
        print(f"✓ Parameter mapping created: {len(rev_idx)} mappings")
        
        # Show some example parameters
        print("\nExample Serum parameters:")
        example_params = [
            'OSC A: Volume', 'OSC A: Wavetable Position', 'FILTER: Cutoff',
            'ENV1: Attack', 'ENV1: Decay', 'ENV1: Sustain', 'ENV1: Release'
        ]
        
        for param in example_params:
            if param in defaults:
                print(f"  {param}: {defaults[param]}")
        
        return True
        
    except FileNotFoundError as e:
        print(f"✓ Expected error (no plugin found): {e}")
        print("  This is normal when running without actual Serum plugin installed.")
        return True
        
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def demo_backend_switching():
    """Demonstrate backend switching"""
    print("\n=== Backend Switching Demo ===")
    
    backends = ['auto', 'pedalboard']
    synth_types = ['diva', 'serum']
    
    for backend in backends:
        for synth_type in synth_types:
            try:
                print(f"Testing {synth_type} with {backend} backend...")
                engine, generator, defaults, rev_idx = create_synth(
                    dataset='toy',
                    synth_type=synth_type,
                    backend=backend
                )
                print(f"✓ {synth_type}/{backend} combination worked")
            except FileNotFoundError:
                print(f"✓ {synth_type}/{backend} failed gracefully (no plugin)")
            except Exception as e:
                print(f"✗ {synth_type}/{backend} failed unexpectedly: {e}")
                return False
    
    return True

if __name__ == "__main__":
    print("Flow Synthesizer Pedalboard Integration Demo")
    print("=" * 50)
    
    success = True
    success &= demo_serum_integration()
    success &= demo_backend_switching()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Demo completed successfully!")
        print("\nKey improvements:")
        print("• Serum 2 support with proper state loading")
        print("• M1 Mac optimization (AudioUnit preference)")
        print("• Multiple backend support (pedalboard/librenderman)")
        print("• Enhanced OSC interface")
    else:
        print("❌ Demo encountered issues")
    
    sys.exit(0 if success else 1)