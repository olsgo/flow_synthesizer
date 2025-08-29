#!/usr/bin/env python3
"""
Ableton Live Integration for PolyMAX Parameter Prediction

This script provides integration between the PolyMAX parameter prediction model
and Ableton Live, allowing automatic application of predicted parameters to the
PolyMAX VST plugin.

Usage:
    python ableton_integration.py --audio_file path/to/audio.wav
    python ableton_integration.py --audio_file path/to/audio.wav --apply_live
    python ableton_integration.py --batch_dir path/to/audio/files/
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent))

from predict_polymax_params import PolyMAXPredictor
from ableton_polymax_remote import PolyMAXClient

class AbletonPolyMAXIntegration:
    """
    Integration class for PolyMAX prediction and Ableton Live control
    """
    
    def __init__(self, model_path: Optional[str] = None, ableton_host: str = 'localhost', ableton_port: int = 9001):
        """
        Initialize the integration system
        
        Args:
            model_path: Path to trained PolyMAX model
            ableton_host: Ableton Live remote script host
            ableton_port: Ableton Live remote script port
        """
        self.predictor = None
        if model_path:
            try:
                self.predictor = PolyMAXPredictor(model_path)
            except Exception as e:
                print(f"Warning: Failed to load model '{model_path}': {e}")
        self.ableton_client = PolyMAXClient(ableton_host, ableton_port)
        
        print(f"Initialized PolyMAX-Ableton integration")
        print(f"Model: {model_path if model_path else 'None (connection-only mode)'}")
        print(f"Ableton: {ableton_host}:{ableton_port}")
    
    def check_ableton_connection(self) -> bool:
        """
        Check if Ableton Live remote script is available
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            status = self.ableton_client.get_status()
            if status.get('status') == 'success':
                print(f"✓ Connected to Ableton Live")
                if status.get('polymax_found'):
                    track_name = status.get('track_name', 'Unknown')
                    param_count = status.get('parameter_count', 0)
                    print(f"✓ PolyMAX found on track '{track_name}' ({param_count} parameters)")
                    return True
                else:
                    print("⚠ PolyMAX plugin not found in Live set")
                    return False
            else:
                print(f"✗ Ableton connection failed: {status.get('message')}")
                return False
        except Exception as e:
            print(f"✗ Ableton connection error: {e}")
            return False
    
    def predict_and_apply(self, audio_file: str, apply_to_live: bool = False, 
                         save_params: bool = True) -> Dict[str, Any]:
        """
        Predict parameters from audio and optionally apply to Ableton Live
        
        Args:
            audio_file: Path to input audio file
            apply_to_live: Whether to send parameters to Ableton Live
            save_params: Whether to save predicted parameters to JSON
        
        Returns:
            Dictionary containing prediction results and status
        """
        try:
            print(f"\nProcessing: {audio_file}")
            if self.predictor is None:
                return {
                    'success': False,
                    'error': 'Model not loaded. Provide a valid model_path to enable prediction.',
                    'file': audio_file
                }
            
            # Predict parameters
            print("Predicting PolyMAX parameters...")
            result = self.predictor.predict_from_file(audio_file)
            
            if not result['success']:
                return {
                    'success': False,
                    'error': result.get('error', 'Prediction failed'),
                    'file': audio_file
                }
            
            parameters = result['parameters']
            confidence = result['confidence']
            
            print(f"✓ Prediction completed (confidence: {confidence:.3f})")
            print(f"  Predicted {len(parameters)} parameters")
            
            # Save parameters if requested
            if save_params:
                output_file = self._save_parameters(audio_file, result)
                print(f"✓ Parameters saved to: {output_file}")
            
            # Apply to Ableton Live if requested
            ableton_result = None
            if apply_to_live:
                print("Applying parameters to Ableton Live...")
                ableton_result = self.ableton_client.send_parameters(parameters)
                
                if ableton_result.get('status') == 'success':
                    applied_count = ableton_result.get('applied_count', 0)
                    print(f"✓ Applied {applied_count} parameters to PolyMAX")
                else:
                    print(f"✗ Failed to apply parameters: {ableton_result.get('message')}")
            
            return {
                'success': True,
                'file': audio_file,
                'parameters': parameters,
                'confidence': confidence,
                'ableton_result': ableton_result,
                'output_file': output_file if save_params else None
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'file': audio_file
            }
    
    def batch_process(self, audio_dir: str, apply_to_live: bool = False, 
                     delay_between: float = 2.0) -> List[Dict[str, Any]]:
        """
        Process multiple audio files in batch
        
        Args:
            audio_dir: Directory containing audio files
            apply_to_live: Whether to apply each prediction to Ableton Live
            delay_between: Delay in seconds between Live parameter updates
        
        Returns:
            List of processing results for each file
        """
        audio_extensions = {'.wav', '.mp3', '.flac', '.aiff', '.m4a'}
        audio_files = []
        
        # Find all audio files
        for ext in audio_extensions:
            audio_files.extend(Path(audio_dir).glob(f'*{ext}'))
            audio_files.extend(Path(audio_dir).glob(f'*{ext.upper()}'))
        
        if not audio_files:
            print(f"No audio files found in {audio_dir}")
            return []
        
        print(f"Found {len(audio_files)} audio files to process")
        
        results = []
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n[{i}/{len(audio_files)}] Processing: {audio_file.name}")
            
            result = self.predict_and_apply(
                str(audio_file), 
                apply_to_live=apply_to_live,
                save_params=True
            )
            
            results.append(result)
            
            # Add delay between Live updates to avoid overwhelming the plugin
            if apply_to_live and i < len(audio_files):
                print(f"Waiting {delay_between}s before next update...")
                time.sleep(delay_between)
        
        # Print summary
        successful = sum(1 for r in results if r['success'])
        print(f"\n=== Batch Processing Summary ===")
        print(f"Total files: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(results) - successful}")
        
        if apply_to_live:
            applied = sum(1 for r in results if r.get('ableton_result', {}).get('status') == 'success')
            print(f"Applied to Live: {applied}")
        
        return results
    
    def _save_parameters(self, audio_file: str, prediction_result: Dict[str, Any]) -> str:
        """
        Save predicted parameters to JSON file
        
        Args:
            audio_file: Original audio file path
            prediction_result: Full prediction result
        
        Returns:
            Path to saved JSON file
        """
        audio_path = Path(audio_file)
        output_file = audio_path.parent / f"{audio_path.stem}_polymax_params.json"
        
        # Create output data
        output_data = {
            'source_audio': str(audio_path),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_info': {
                'type': 'PolyMAX_VAE',
                'confidence': prediction_result['confidence']
            },
            'parameters': prediction_result['parameters'],
            'metadata': {
                'parameter_count': len(prediction_result['parameters']),
                'prediction_time': prediction_result.get('prediction_time', 0)
            }
        }
        
        # Save to JSON
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        return str(output_file)
    
    def interactive_mode(self):
        """
        Run in interactive mode for real-time parameter prediction
        """
        print("\n=== PolyMAX Interactive Mode ===")
        print("Commands:")
        print("  predict <audio_file> - Predict parameters")
        print("  apply <audio_file> - Predict and apply to Live")
        print("  status - Check Ableton connection")
        print("  quit - Exit")
        
        while True:
            try:
                command = input("\n> ").strip().split()
                
                if not command:
                    continue
                
                if command[0] == 'quit':
                    break
                
                elif command[0] == 'status':
                    self.check_ableton_connection()
                
                elif command[0] == 'predict' and len(command) > 1:
                    audio_file = ' '.join(command[1:])
                    if os.path.exists(audio_file):
                        result = self.predict_and_apply(audio_file, apply_to_live=False)
                        if result['success']:
                            print(f"Confidence: {result['confidence']:.3f}")
                            print(f"Parameters: {len(result['parameters'])}")
                    else:
                        print(f"File not found: {audio_file}")
                
                elif command[0] == 'apply' and len(command) > 1:
                    audio_file = ' '.join(command[1:])
                    if os.path.exists(audio_file):
                        result = self.predict_and_apply(audio_file, apply_to_live=True)
                        if result['success']:
                            ableton_status = result.get('ableton_result', {}).get('status')
                            print(f"Live update: {ableton_status}")
                    else:
                        print(f"File not found: {audio_file}")
                
                else:
                    print("Unknown command. Type 'quit' to exit.")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("Goodbye!")


def main():
    parser = argparse.ArgumentParser(description='PolyMAX-Ableton Live Integration')
    parser.add_argument('--model', default='outputs_optimized/models/best_model.pth',
                       help='Path to trained PolyMAX model')
    parser.add_argument('--audio_file', help='Single audio file to process')
    parser.add_argument('--batch_dir', help='Directory of audio files to process')
    parser.add_argument('--apply_live', action='store_true',
                       help='Apply predicted parameters to Ableton Live')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--ableton_host', default='localhost',
                       help='Ableton Live remote script host')
    parser.add_argument('--ableton_port', type=int, default=9001,
                       help='Ableton Live remote script port')
    parser.add_argument('--delay', type=float, default=2.0,
                       help='Delay between Live updates in batch mode')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        print("Make sure training has completed and model is saved.")
        return 1
    
    # Initialize integration
    try:
        integration = AbletonPolyMAXIntegration(
            args.model, 
            args.ableton_host, 
            args.ableton_port
        )
    except Exception as e:
        print(f"Error initializing integration: {e}")
        return 1
    
    # Check Ableton connection if Live integration is requested
    if args.apply_live:
        if not integration.check_ableton_connection():
            print("\nAbleton Live connection failed.")
            print("Make sure:")
            print("1. Ableton Live is running")
            print("2. PolyMAX remote script is installed and selected")
            print("3. PolyMAX VST is loaded on a track")
            return 1
    
    # Run based on arguments
    if args.interactive:
        integration.interactive_mode()
    
    elif args.audio_file:
        if not os.path.exists(args.audio_file):
            print(f"Error: Audio file not found: {args.audio_file}")
            return 1
        
        result = integration.predict_and_apply(
            args.audio_file, 
            apply_to_live=args.apply_live
        )
        
        if not result['success']:
            print(f"Error: {result['error']}")
            return 1
    
    elif args.batch_dir:
        if not os.path.isdir(args.batch_dir):
            print(f"Error: Directory not found: {args.batch_dir}")
            return 1
        
        results = integration.batch_process(
            args.batch_dir,
            apply_to_live=args.apply_live,
            delay_between=args.delay
        )
        
        # Check if any processing failed
        failed = [r for r in results if not r['success']]
        if failed:
            print(f"\nWarning: {len(failed)} files failed to process")
            return 1
    
    else:
        print("Error: Specify --audio_file, --batch_dir, or --interactive")
        parser.print_help()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
