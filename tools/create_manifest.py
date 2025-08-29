#!/usr/bin/env python3
"""
Create Manifest CSV for PolyMAX Dataset

This script creates a manifest.csv file that maps audio files to their corresponding
parameter files with train/test splits for the flow synthesizer training.
"""

import os
import csv
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ManifestCreator:
    """Create manifest CSV for PolyMAX dataset."""
    
    def __init__(self):
        self.audio_dir = '/Users/gjb/Datasets/polymax/render'
        self.params_dir = '/Users/gjb/Projects/flow_synthesizer/params'
        self.manifest_path = '/Users/gjb/Projects/flow_synthesizer/manifest.csv'
        self.train_split = 0.8  # 80% train, 20% test
        
    def find_matching_files(self) -> List[Dict[str, str]]:
        """Find matching audio and parameter files."""
        audio_files = []
        param_files = []
        
        # Find all audio files
        if os.path.exists(self.audio_dir):
            for file in os.listdir(self.audio_dir):
                if file.endswith('.wav'):
                    audio_files.append(file)
                    
        # Find all parameter files
        if os.path.exists(self.params_dir):
            for file in os.listdir(self.params_dir):
                if file.endswith('.json'):
                    param_files.append(file)
                    
        logger.info(f"Found {len(audio_files)} audio files and {len(param_files)} parameter files")
        
        # Match files by stem name
        matched_files = []
        audio_stems = {Path(f).stem: f for f in audio_files}
        param_stems = {Path(f).stem: f for f in param_files}
        
        for stem in audio_stems:
            if stem in param_stems:
                # Get preset name from parameter file
                param_path = os.path.join(self.params_dir, param_stems[stem])
                preset_name = stem
                
                try:
                    with open(param_path, 'r') as f:
                        param_data = json.load(f)
                        preset_name = param_data.get('preset_name', stem)
                except Exception as e:
                    logger.warning(f"Could not read preset name from {param_path}: {e}")
                
                matched_files.append({
                    'stem': stem,
                    'audio_file': audio_stems[stem],
                    'param_file': param_stems[stem],
                    'preset_name': preset_name,
                    'audio_path': os.path.join(self.audio_dir, audio_stems[stem]),
                    'params_path': os.path.join(self.params_dir, param_stems[stem])
                })
            else:
                logger.warning(f"No matching parameter file for audio: {stem}")
                
        for stem in param_stems:
            if stem not in audio_stems:
                logger.warning(f"No matching audio file for parameters: {stem}")
                
        logger.info(f"Successfully matched {len(matched_files)} file pairs")
        return matched_files
        
    def create_train_test_split(self, matched_files: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Create train/test split for the dataset."""
        # Shuffle files for random split
        random.seed(42)  # For reproducible splits
        shuffled_files = matched_files.copy()
        random.shuffle(shuffled_files)
        
        # Calculate split point
        train_count = int(len(shuffled_files) * self.train_split)
        
        # Assign splits
        for i, file_info in enumerate(shuffled_files):
            file_info['split'] = 'train' if i < train_count else 'test'
            
        train_files = [f for f in shuffled_files if f['split'] == 'train']
        test_files = [f for f in shuffled_files if f['split'] == 'test']
        
        logger.info(f"Split: {len(train_files)} train, {len(test_files)} test")
        return shuffled_files
        
    def create_manifest_csv(self, file_data: List[Dict[str, str]]) -> bool:
        """Create the manifest CSV file."""
        try:
            with open(self.manifest_path, 'w', newline='') as csvfile:
                fieldnames = [
                    'split',
                    'audio_path', 
                    'params_path',
                    'preset_name',
                    'stem',
                    'audio_file',
                    'param_file'
                ]
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for file_info in file_data:
                    writer.writerow({
                        'split': file_info['split'],
                        'audio_path': file_info['audio_path'],
                        'params_path': file_info['params_path'],
                        'preset_name': file_info['preset_name'],
                        'stem': file_info['stem'],
                        'audio_file': file_info['audio_file'],
                        'param_file': file_info['param_file']
                    })
                    
            logger.info(f"Created manifest CSV: {self.manifest_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create manifest CSV: {e}")
            return False
            
    def validate_manifest(self) -> bool:
        """Validate the created manifest file."""
        try:
            with open(self.manifest_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                rows = list(reader)
                
            if not rows:
                logger.error("Manifest file is empty")
                return False
                
            # Check required columns
            required_cols = ['split', 'audio_path', 'params_path', 'preset_name']
            if not all(col in rows[0] for col in required_cols):
                logger.error(f"Missing required columns in manifest")
                return False
                
            # Validate file existence
            missing_files = []
            for row in rows:
                if not os.path.exists(row['audio_path']):
                    missing_files.append(f"Audio: {row['audio_path']}")
                if not os.path.exists(row['params_path']):
                    missing_files.append(f"Params: {row['params_path']}")
                    
            if missing_files:
                logger.warning(f"Found {len(missing_files)} missing files:")
                for missing in missing_files[:5]:  # Show first 5
                    logger.warning(f"  {missing}")
                if len(missing_files) > 5:
                    logger.warning(f"  ... and {len(missing_files) - 5} more")
                    
            # Count splits
            train_count = sum(1 for row in rows if row['split'] == 'train')
            test_count = sum(1 for row in rows if row['split'] == 'test')
            
            logger.info(f"Manifest validation complete:")
            logger.info(f"  Total entries: {len(rows)}")
            logger.info(f"  Train: {train_count} ({train_count/len(rows)*100:.1f}%)")
            logger.info(f"  Test: {test_count} ({test_count/len(rows)*100:.1f}%)")
            logger.info(f"  Missing files: {len(missing_files)}")
            
            return len(missing_files) == 0
            
        except Exception as e:
            logger.error(f"Failed to validate manifest: {e}")
            return False
            
    def create_manifest(self) -> bool:
        """Main function to create the manifest file."""
        logger.info("Creating PolyMAX dataset manifest...")
        
        # Find matching files
        matched_files = self.find_matching_files()
        if not matched_files:
            logger.error("No matching audio/parameter file pairs found")
            return False
            
        # Create train/test split
        split_files = self.create_train_test_split(matched_files)
        
        # Create CSV
        if not self.create_manifest_csv(split_files):
            return False
            
        # Validate
        return self.validate_manifest()
        
def main():
    """Main function to create the manifest."""
    creator = ManifestCreator()
    success = creator.create_manifest()
    
    if success:
        print("\n=== Manifest Creation Complete ===")
        print(f"Manifest file created: {creator.manifest_path}")
        print("Ready for flow synthesizer training!")
        return 0
    else:
        print("\n=== Manifest Creation Failed ===")
        return 1
        
if __name__ == '__main__':
    exit(main())