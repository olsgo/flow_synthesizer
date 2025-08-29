#!/usr/bin/env python3
"""
Test script for Ableton Live integration with PolyMAX parameter prediction.
This script tests the connection and communication with Ableton Live.
"""

import sys
import os
import time
import json
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ableton_integration import AbletonPolyMAXIntegration

def test_ableton_connection():
    """
    Test basic connection to Ableton Live.
    """
    print("=== Testing Ableton Live Connection ===")
    
    try:
        # Initialize integration (without model for connection test)
        integration = AbletonPolyMAXIntegration(model_path=None)
        
        # Test connection
        print("Checking Ableton Live connection...")
        is_connected = integration.check_ableton_connection()
        
        if is_connected:
            print("✅ Successfully connected to Ableton Live!")
            return True
        else:
            print("❌ Failed to connect to Ableton Live.")
            print("Please ensure:")
            print("1. Ableton Live is running")
            print("2. The PolyMAX remote script is installed and active")
            print("3. The remote script server is listening on port 9001")
            return False
            
    except Exception as e:
        print(f"❌ Error testing connection: {e}")
        return False

def test_parameter_sending():
    """
    Test sending sample parameters to Ableton Live.
    """
    print("\n=== Testing Parameter Sending ===")
    
    try:
        integration = AbletonPolyMAXIntegration(model_path=None)
        
        # Create sample parameters (normalized 0-1 values)
        sample_params = {
            "osc1_wave": 0.5,
            "osc1_pitch": 0.0,
            "osc2_wave": 0.3,
            "osc2_pitch": 0.1,
            "filter_cutoff": 0.7,
            "filter_resonance": 0.4,
            "env_attack": 0.2,
            "env_decay": 0.6,
            "env_sustain": 0.8,
            "env_release": 0.5
        }
        
        print(f"Sending sample parameters: {json.dumps(sample_params, indent=2)}")
        
        success = integration.apply_parameters_to_ableton(sample_params)
        
        if success:
            print("✅ Successfully sent parameters to Ableton Live!")
            print("Check your PolyMAX plugin in Live to see if parameters changed.")
            return True
        else:
            print("❌ Failed to send parameters to Ableton Live.")
            return False
            
    except Exception as e:
        print(f"❌ Error sending parameters: {e}")
        return False

def test_with_audio_file():
    """
    Test the full pipeline with an audio file (if model is available).
    """
    print("\n=== Testing Full Pipeline with Audio File ===")
    
    # Check if we have a trained model
    model_path = "outputs_optimized/models/best_model.pth"
    if not os.path.exists(model_path):
        print("⚠️  No trained model found. Skipping audio file test.")
        print(f"Expected model at: {model_path}")
        return False
    
    # Check if we have test audio files
    test_audio_dir = "/Users/gjb/Datasets/polymax/render"
    if not os.path.exists(test_audio_dir):
        print("⚠️  No test audio directory found. Skipping audio file test.")
        return False
    
    # Find a test audio file
    audio_files = list(Path(test_audio_dir).glob("*.wav"))
    if not audio_files:
        print("⚠️  No audio files found in test directory. Skipping audio file test.")
        return False
    
    test_file = audio_files[0]
    print(f"Using test audio file: {test_file.name}")
    
    try:
        # Initialize with model
        integration = AbletonPolyMAXIntegration(model_path=model_path)
        
        # Predict and apply parameters
        print("Predicting parameters from audio...")
        success = integration.predict_and_apply_to_ableton(str(test_file))
        
        if success:
            print("✅ Successfully predicted and applied parameters to Ableton Live!")
            print("Check your PolyMAX plugin in Live to hear the predicted sound.")
            return True
        else:
            print("❌ Failed to predict and apply parameters.")
            return False
            
    except Exception as e:
        print(f"❌ Error in full pipeline test: {e}")
        return False

def main():
    """
    Run all integration tests.
    """
    print("PolyMAX Ableton Live Integration Test")
    print("====================================")
    
    # Test 1: Basic connection
    connection_ok = test_ableton_connection()
    
    if not connection_ok:
        print("\n❌ Connection test failed. Please fix connection issues before proceeding.")
        return
    
    # Test 2: Parameter sending
    time.sleep(1)  # Brief pause
    params_ok = test_parameter_sending()
    
    # Test 3: Full pipeline (if model available)
    time.sleep(1)  # Brief pause
    pipeline_ok = test_with_audio_file()
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"Connection Test: {'✅ PASS' if connection_ok else '❌ FAIL'}")
    print(f"Parameter Test: {'✅ PASS' if params_ok else '❌ FAIL'}")
    print(f"Pipeline Test: {'✅ PASS' if pipeline_ok else '⚠️  SKIP' if not pipeline_ok else '❌ FAIL'}")
    
    if connection_ok and params_ok:
        print("\n🎉 Integration is working! You can now use the PolyMAX prediction system with Ableton Live.")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main()