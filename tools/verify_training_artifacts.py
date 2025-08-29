#!/usr/bin/env python3
"""
Verify Training Artifacts for PolyMAX Dataset

This script verifies that all training artifacts are complete and properly formatted:
- params_schema.json
- params/*.json files
- manifest.csv
- audio files
"""

import os
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainingArtifactsVerifier:
    """Verify all training artifacts for PolyMAX dataset."""
    
    def __init__(self):
        self.project_dir = '/Users/gjb/Projects/flow_synthesizer'
        self.audio_dir = '/Users/gjb/Datasets/polymax/render'
        self.params_dir = os.path.join(self.project_dir, 'params')
        self.schema_path = os.path.join(self.project_dir, 'params_schema.json')
        self.manifest_path = os.path.join(self.project_dir, 'manifest.csv')
        
        self.errors = []
        self.warnings = []
        
    def log_error(self, message: str):
        """Log an error message."""
        self.errors.append(message)
        logger.error(message)
        
    def log_warning(self, message: str):
        """Log a warning message."""
        self.warnings.append(message)
        logger.warning(message)
        
    def verify_params_schema(self) -> bool:
        """Verify the parameter schema file."""
        logger.info("Verifying params_schema.json...")
        
        if not os.path.exists(self.schema_path):
            self.log_error(f"Parameter schema file not found: {self.schema_path}")
            return False
            
        try:
            with open(self.schema_path, 'r') as f:
                schema = json.load(f)
                
            # Check required fields
            required_fields = ['plugin_name', 'total_parameters', 'parameter_order']
            for field in required_fields:
                if field not in schema:
                    self.log_error(f"Missing required field in schema: {field}")
                    return False
                    
            # Verify parameter count
            param_count = len(schema['parameter_order'])
            if param_count != schema['total_parameters']:
                self.log_error(f"Parameter count mismatch: {param_count} vs {schema['total_parameters']}")
                return False
                
            logger.info(f"✓ Schema valid with {param_count} parameters")
            return True
            
        except Exception as e:
            self.log_error(f"Failed to load parameter schema: {e}")
            return False
            
    def verify_parameter_files(self) -> Tuple[bool, int]:
        """Verify all parameter JSON files."""
        logger.info("Verifying parameter files...")
        
        if not os.path.exists(self.params_dir):
            self.log_error(f"Parameters directory not found: {self.params_dir}")
            return False, 0
            
        # Load schema for validation
        try:
            with open(self.schema_path, 'r') as f:
                schema = json.load(f)
            expected_param_count = len(schema['parameter_order'])
        except Exception as e:
            self.log_error(f"Cannot load schema for validation: {e}")
            return False, 0
            
        param_files = [f for f in os.listdir(self.params_dir) if f.endswith('.json')]
        valid_files = 0
        
        for param_file in param_files:
            param_path = os.path.join(self.params_dir, param_file)
            
            try:
                with open(param_path, 'r') as f:
                    param_data = json.load(f)
                    
                # Check required fields
                required_fields = ['preset_name', 'parameter_vector', 'parameter_count']
                for field in required_fields:
                    if field not in param_data:
                        self.log_error(f"Missing field '{field}' in {param_file}")
                        continue
                        
                # Verify parameter vector
                param_vector = param_data['parameter_vector']
                if not isinstance(param_vector, list):
                    self.log_error(f"Parameter vector is not a list in {param_file}")
                    continue
                    
                if len(param_vector) != expected_param_count:
                    self.log_error(f"Parameter vector length mismatch in {param_file}: {len(param_vector)} vs {expected_param_count}")
                    continue
                    
                # Check parameter values are in [0,1] range
                for i, val in enumerate(param_vector):
                    if not isinstance(val, (int, float)):
                        self.log_error(f"Non-numeric parameter value at index {i} in {param_file}")
                        break
                    if not (0.0 <= val <= 1.0):
                        self.log_warning(f"Parameter value {val} outside [0,1] range at index {i} in {param_file}")
                        
                valid_files += 1
                
            except Exception as e:
                self.log_error(f"Failed to load parameter file {param_file}: {e}")
                
        logger.info(f"✓ {valid_files}/{len(param_files)} parameter files valid")
        return valid_files == len(param_files), len(param_files)
        
    def verify_manifest(self) -> Tuple[bool, int]:
        """Verify the manifest CSV file."""
        logger.info("Verifying manifest.csv...")
        
        if not os.path.exists(self.manifest_path):
            self.log_error(f"Manifest file not found: {self.manifest_path}")
            return False, 0
            
        try:
            with open(self.manifest_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
            if not rows:
                self.log_error("Manifest file is empty")
                return False, 0
                
            # Check required columns
            required_cols = ['split', 'audio_path', 'params_path', 'preset_name']
            missing_cols = [col for col in required_cols if col not in rows[0]]
            if missing_cols:
                self.log_error(f"Missing columns in manifest: {missing_cols}")
                return False, 0
                
            # Verify file references
            missing_audio = 0
            missing_params = 0
            train_count = 0
            test_count = 0
            
            for i, row in enumerate(rows):
                # Check split values
                if row['split'] not in ['train', 'test']:
                    self.log_error(f"Invalid split value '{row['split']}' in row {i+1}")
                elif row['split'] == 'train':
                    train_count += 1
                else:
                    test_count += 1
                    
                # Check file existence
                if not os.path.exists(row['audio_path']):
                    missing_audio += 1
                    if missing_audio <= 3:  # Only log first few
                        self.log_error(f"Missing audio file: {row['audio_path']}")
                        
                if not os.path.exists(row['params_path']):
                    missing_params += 1
                    if missing_params <= 3:  # Only log first few
                        self.log_error(f"Missing parameter file: {row['params_path']}")
                        
            if missing_audio > 3:
                self.log_error(f"... and {missing_audio - 3} more missing audio files")
            if missing_params > 3:
                self.log_error(f"... and {missing_params - 3} more missing parameter files")
                
            logger.info(f"✓ Manifest has {len(rows)} entries: {train_count} train, {test_count} test")
            logger.info(f"  Missing files: {missing_audio} audio, {missing_params} params")
            
            return missing_audio == 0 and missing_params == 0, len(rows)
            
        except Exception as e:
            self.log_error(f"Failed to load manifest: {e}")
            return False, 0
            
    def verify_audio_files(self) -> Tuple[bool, int]:
        """Verify audio files exist and have reasonable properties."""
        logger.info("Verifying audio files...")
        
        if not os.path.exists(self.audio_dir):
            self.log_error(f"Audio directory not found: {self.audio_dir}")
            return False, 0
            
        audio_files = [f for f in os.listdir(self.audio_dir) if f.endswith('.wav')]
        
        if not audio_files:
            self.log_error("No audio files found")
            return False, 0
            
        # Check file sizes
        small_files = 0
        large_files = 0
        
        for audio_file in audio_files:
            audio_path = os.path.join(self.audio_dir, audio_file)
            file_size = os.path.getsize(audio_path)
            
            # Reasonable size checks (assuming 4-second audio at 44.1kHz, 16-bit, stereo)
            # Expected size: ~700KB, allow range 100KB - 5MB
            if file_size < 100_000:  # 100KB
                small_files += 1
                if small_files <= 3:
                    self.log_warning(f"Small audio file ({file_size} bytes): {audio_file}")
            elif file_size > 5_000_000:  # 5MB
                large_files += 1
                if large_files <= 3:
                    self.log_warning(f"Large audio file ({file_size} bytes): {audio_file}")
                    
        if small_files > 3:
            self.log_warning(f"... and {small_files - 3} more small audio files")
        if large_files > 3:
            self.log_warning(f"... and {large_files - 3} more large audio files")
            
        logger.info(f"✓ Found {len(audio_files)} audio files")
        return True, len(audio_files)
        
    def run_verification(self) -> bool:
        """Run complete verification of all training artifacts."""
        logger.info("=== Training Artifacts Verification ===")
        
        # Verify each component
        schema_ok = self.verify_params_schema()
        params_ok, param_count = self.verify_parameter_files()
        manifest_ok, manifest_count = self.verify_manifest()
        audio_ok, audio_count = self.verify_audio_files()
        
        # Summary
        logger.info("\n=== Verification Summary ===")
        logger.info(f"Parameter schema: {'✓' if schema_ok else '✗'}")
        logger.info(f"Parameter files: {'✓' if params_ok else '✗'} ({param_count} files)")
        logger.info(f"Manifest file: {'✓' if manifest_ok else '✗'} ({manifest_count} entries)")
        logger.info(f"Audio files: {'✓' if audio_ok else '✗'} ({audio_count} files)")
        
        if self.errors:
            logger.error(f"\nFound {len(self.errors)} errors:")
            for error in self.errors[:10]:  # Show first 10
                logger.error(f"  {error}")
            if len(self.errors) > 10:
                logger.error(f"  ... and {len(self.errors) - 10} more errors")
                
        if self.warnings:
            logger.warning(f"\nFound {len(self.warnings)} warnings:")
            for warning in self.warnings[:5]:  # Show first 5
                logger.warning(f"  {warning}")
            if len(self.warnings) > 5:
                logger.warning(f"  ... and {len(self.warnings) - 5} more warnings")
                
        all_ok = schema_ok and params_ok and manifest_ok and audio_ok
        
        if all_ok:
            logger.info("\n🎉 All training artifacts verified successfully!")
            logger.info("Dataset is ready for flow synthesizer training.")
        else:
            logger.error("\n❌ Verification failed. Please fix the errors above.")
            
        return all_ok
        
def main():
    """Main function to verify training artifacts."""
    verifier = TrainingArtifactsVerifier()
    success = verifier.run_verification()
    return 0 if success else 1
    
if __name__ == '__main__':
    exit(main())