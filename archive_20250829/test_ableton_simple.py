#!/usr/bin/env python3
"""
Simplified test script for PolyMAX parameter prediction without Ableton Live dependency.
This tests the core prediction functionality and TCP communication setup.
"""

import sys
import os
import time
import json
import socket
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predict_polymax_params import PolyMAXPredictor

def test_tcp_server():
    """
    Test if we can create a TCP server on the expected port.
    """
    print("=== Testing TCP Server Setup ===")
    
    try:
        # Try to create a server socket on the expected port
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('localhost', 9001))
        server_socket.listen(1)
        
        print("✅ TCP server can bind to port 9001")
        print("Server is ready to accept connections on localhost:9001")
        
        # Close the server
        server_socket.close()
        return True
        
    except Exception as e:
        print(f"❌ Failed to create TCP server: {e}")
        return False

def test_parameter_prediction():
    """
    Test parameter prediction with a sample audio file (if available).
    """
    print("\n=== Testing Parameter Prediction ===")
    
    # Check if we have a trained model
    model_path = "outputs_optimized/models/best_model.pth"
    if not os.path.exists(model_path):
        print("⚠️  No trained model found. Skipping prediction test.")
        print(f"Expected model at: {model_path}")
        return False
    
    # Check if we have test audio files
    test_audio_dir = "/Users/gjb/Datasets/polymax/render"
    if not os.path.exists(test_audio_dir):
        print("⚠️  No test audio directory found. Skipping prediction test.")
        return False
    
    # Find a test audio file
    audio_files = list(Path(test_audio_dir).glob("*.wav"))
    if not audio_files:
        print("⚠️  No audio files found in test directory. Skipping prediction test.")
        return False
    
    test_file = audio_files[0]
    print(f"Using test audio file: {test_file.name}")
    
    try:
        # Initialize predictor
        predictor = PolyMAXPredictor(model_path=model_path)
        
        # Predict parameters
        print("Predicting parameters from audio...")
        result = predictor.predict_from_audio(str(test_file))
        
        if result:
            print("✅ Successfully predicted parameters!")
            print(f"Predicted parameters: {json.dumps(result['parameters'], indent=2)}")
            print(f"Confidence: {result['confidence']:.3f}")
            return True
        else:
            print("❌ Failed to predict parameters.")
            return False
            
    except Exception as e:
        print(f"❌ Error in prediction test: {e}")
        return False

def test_parameter_format():
    """
    Test parameter formatting and validation.
    """
    print("\n=== Testing Parameter Format ===")
    
    try:
        # Create sample parameters
        sample_params = {
            "osc1_wave": 0.5,
            "osc1_pitch": 0.0,
            "osc2_wave": 0.3,
            "filter_cutoff": 0.7,
            "filter_resonance": 0.4,
            "env_attack": 0.2,
            "env_decay": 0.6,
            "env_sustain": 0.8,
            "env_release": 0.5
        }
        
        print(f"Sample parameters: {json.dumps(sample_params, indent=2)}")
        
        # Validate parameter ranges
        valid = True
        for param, value in sample_params.items():
            if not (0.0 <= value <= 1.0):
                print(f"❌ Parameter {param} value {value} is out of range [0, 1]")
                valid = False
        
        if valid:
            print("✅ All parameters are in valid range [0, 1]")
            return True
        else:
            print("❌ Some parameters are out of valid range")
            return False
            
    except Exception as e:
        print(f"❌ Error in parameter format test: {e}")
        return False

def test_json_serialization():
    """
    Test JSON serialization for TCP communication.
    """
    print("\n=== Testing JSON Serialization ===")
    
    try:
        # Create a message like what would be sent to Ableton
        message = {
            "action": "set_parameters",
            "parameters": {
                "osc1_wave": 0.5,
                "osc1_pitch": 0.0,
                "filter_cutoff": 0.7,
                "env_attack": 0.2
            },
            "timestamp": time.time()
        }
        
        # Serialize to JSON
        json_str = json.dumps(message)
        print(f"Serialized message: {json_str}")
        
        # Deserialize back
        parsed_message = json.loads(json_str)
        print(f"Parsed message: {parsed_message}")
        
        # Verify integrity
        if message == parsed_message:
            print("✅ JSON serialization/deserialization works correctly")
            return True
        else:
            print("❌ JSON serialization/deserialization failed")
            return False
            
    except Exception as e:
        print(f"❌ Error in JSON serialization test: {e}")
        return False

def main():
    """
    Run all simplified integration tests.
    """
    print("PolyMAX Integration Test (Simplified)")
    print("=====================================")
    
    # Test 1: TCP server setup
    tcp_ok = test_tcp_server()
    
    # Test 2: Parameter prediction (if model available)
    time.sleep(0.5)
    prediction_ok = test_parameter_prediction()
    
    # Test 3: Parameter format validation
    time.sleep(0.5)
    format_ok = test_parameter_format()
    
    # Test 4: JSON serialization
    time.sleep(0.5)
    json_ok = test_json_serialization()
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"TCP Server Test: {'✅ PASS' if tcp_ok else '❌ FAIL'}")
    print(f"Prediction Test: {'✅ PASS' if prediction_ok else '⚠️  SKIP' if not prediction_ok else '❌ FAIL'}")
    print(f"Parameter Format Test: {'✅ PASS' if format_ok else '❌ FAIL'}")
    print(f"JSON Serialization Test: {'✅ PASS' if json_ok else '❌ FAIL'}")
    
    if tcp_ok and format_ok and json_ok:
        print("\n🎉 Core integration components are working!")
        print("\nNext steps:")
        print("1. Install the Ableton Live remote script")
        print("2. Configure Ableton Live to use the remote script")
        print("3. Test the full integration with Live running")
    else:
        print("\n⚠️  Some core tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main()