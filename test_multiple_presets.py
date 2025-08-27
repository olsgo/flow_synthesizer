#!/usr/bin/env python3
"""
Test script to validate Serum 2 preset loading with multiple presets
using the parameter mapping approach inspired by the xml_synth_sound_rendering example.
"""

import os
import glob
import json
from serum2_parameter_mapper import Serum2ParameterMapper

def test_multiple_presets():
    """Test parameter mapping with multiple Serum 2 presets"""
    
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
    
    print(f"Found {len(preset_files)} preset files to test")
    
    results = []
    
    # Test first 5 presets to avoid overwhelming output
    test_presets = preset_files[:5]
    
    for i, preset_path in enumerate(test_presets, 1):
        preset_name = os.path.basename(preset_path)
        print(f"\n=== Testing preset {i}/{len(test_presets)}: {preset_name} ===")
        
        try:
            # Load and apply preset
            parameters_set = mapper.load_preset_with_mapping(preset_path, verbose=False)
            
            # Test audio generation
            success = mapper.test_preset_loading(preset_path)
            
            if success and parameters_set > 0:
                print(f"✅ SUCCESS: {parameters_set} parameters set")
                print(f"   Audio generation test: PASSED")
            else:
                print(f"❌ FAILED: {parameters_set} parameters set, audio test: {'PASSED' if success else 'FAILED'}")
            
            results.append({
                'preset': preset_name,
                'success': success and parameters_set > 0,
                'parameters_set': parameters_set,
                'audio_test': success
            })
            
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            results.append({
                'preset': preset_name,
                'success': False,
                'error': str(e)
            })
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    successful = sum(1 for r in results if r['success'])
    total = len(results)
    
    print(f"Overall success rate: {successful}/{total} ({100*successful/total:.1f}%)")
    
    if successful > 0:
        avg_params = sum(r['parameters_set'] for r in results if r['success']) / successful
        audio_tests_passed = sum(1 for r in results if r['audio_test'])
        print(f"Average parameters set per preset: {avg_params:.1f}")
        print(f"Audio generation tests passed: {audio_tests_passed}/{total}")
    
    # Show any failures
    failures = [r for r in results if not r['success']]
    if failures:
        print(f"\nFailed presets ({len(failures)}):")
        for failure in failures:
            params = failure.get('parameters_set', 0)
            audio = failure.get('audio_test', False)
            print(f"  - {failure['preset']}: {params} params, audio: {'PASS' if audio else 'FAIL'}")
    
    return results

if __name__ == "__main__":
    test_multiple_presets()