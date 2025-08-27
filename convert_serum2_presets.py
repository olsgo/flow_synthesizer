#!/usr/bin/env python3
"""
Serum 2 Preset Converter

This script recursively finds .SerumPreset files and converts them to JSON format
using a Python implementation based on the serum-preset-packager approach.
The converted presets can then be used with DawDreamer to apply settings to synthesizer plugins.
"""

import os
import subprocess
import json
import struct
import zstandard as zstd
import cbor2
from pathlib import Path
import argparse
from typing import List, Dict, Any, Optional
import hashlib


class Serum2PresetDecoder:
    """Decoder for Serum 2 .SerumPreset files."""
    
    def __init__(self):
        self.xfer_json_header = b'XferJson'
        self.serum2_header = b'SERUM2PRESET'
    
    def decode_preset_file(self, preset_path: Path) -> Optional[Dict[str, Any]]:
        """Decode a .SerumPreset file to JSON-compatible dictionary."""
        try:
            with open(preset_path, 'rb') as f:
                data = f.read()
            
            # Check for XferJson format (newer format)
            if data.startswith(self.xfer_json_header):
                return self._decode_xfer_json_format(data, preset_path)
            # Check for SERUM2PRESET format (older format)
            elif data.startswith(self.serum2_header):
                return self._decode_serum2_format(data, preset_path)
            else:
                print(f"Warning: {preset_path} doesn't have recognized magic header")
                return None
            
        except Exception as e:
            print(f"Error processing {preset_path}: {e}")
            return None
    
    def _decode_xfer_json_format(self, data: bytes, preset_path: Path) -> Optional[Dict[str, Any]]:
        """Decode XferJson format preset."""
        try:
            # Find the end of the JSON data
            json_end = data.find(b'}') + 1
            if json_end <= 0:
                print(f"No JSON end found in {preset_path}")
                return None
            
            # Extract and parse JSON metadata
            # Find the actual start of JSON (after XferJson header and length info)
            json_start = data.find(b'{')  # Find the opening brace
            if json_start == -1:
                print(f"No JSON start found in {preset_path}")
                return None
            
            json_data = data[json_start:json_end]
            try:
                metadata = json.loads(json_data.decode('utf-8'))
            except Exception as e:
                print(f"Error parsing JSON metadata from {preset_path}: {e}")
                return None
            
            # Process binary data section after JSON
            remaining_data = data[json_end:]
            preset_data = {}
            
            if len(remaining_data) >= 12:  # Need at least 12 bytes for header
                try:
                    # Parse the binary header after JSON
                    # Format appears to be: [1 byte] [4 bytes length] [4 bytes unknown] [4 bytes compressed size]
                    header_offset = 0
                    
                    # Skip any padding bytes
                    while header_offset < len(remaining_data) and remaining_data[header_offset] in [0x01, 0x00]:
                        header_offset += 1
                    
                    if header_offset + 12 <= len(remaining_data):
                        # Read header information
                        header_data = remaining_data[header_offset:header_offset + 12]
                        
                        # Try to find zstandard compressed data
                        zstd_magic = b'\x28\xB5\x2F\xFD'
                        zstd_pos = remaining_data.find(zstd_magic)
                        
                        if zstd_pos >= 0:
                            # Extract compressed data
                            compressed_data = remaining_data[zstd_pos:]
                            
                            try:
                                # Decompress with zstandard
                                dctx = zstd.ZstdDecompressor()
                                decompressed_data = dctx.decompress(compressed_data)
                                
                                # Try to decode as CBOR
                                try:
                                    cbor_data = cbor2.loads(decompressed_data)
                                    preset_data = cbor_data
                                except Exception as cbor_error:
                                    # If CBOR fails, try to parse as binary parameter data
                                    preset_data = self._parse_binary_parameters(decompressed_data)
                                    
                            except Exception as decomp_error:
                                print(f"Decompression failed for {preset_path}: {decomp_error}")
                                preset_data = {"error": "decompression_failed"}
                        else:
                            print(f"No zstandard data found in {preset_path}")
                            preset_data = {"error": "no_compressed_data"}
                    else:
                        print(f"Insufficient header data in {preset_path}")
                        preset_data = {"error": "insufficient_header"}
                        
                except Exception as e:
                    print(f"Error parsing binary section of {preset_path}: {e}")
                    preset_data = {"error": str(e)}
            
            # Structure the result
            result = {
                **metadata,  # Include all JSON metadata at top level
                "data": preset_data
            }
            
            return result
            
        except Exception as e:
            print(f"Error processing XferJson format {preset_path}: {e}")
            return None
    
    def _parse_binary_parameters(self, data: bytes) -> Dict[str, Any]:
        """Parse binary parameter data from decompressed preset data."""
        parameters = {}
        
        try:
            # Try to parse as structured binary data
            offset = 0
            param_index = 0
            
            while offset + 4 <= len(data):
                # Try to read 4-byte float values
                try:
                    value = struct.unpack('<f', data[offset:offset+4])[0]
                    if -1000.0 <= value <= 1000.0:  # Reasonable parameter range
                        parameters[f"param_{param_index}"] = value
                        param_index += 1
                except:
                    pass
                offset += 4
            
            # If we found parameters, return them
            if parameters:
                return {"parameters": parameters}
            
            # Fallback: return hex dump for analysis
            return {"raw_hex": data[:1024].hex()}  # First 1KB as hex
            
        except Exception as e:
            return {"parse_error": str(e)}
    
    def _decode_serum2_format(self, data: bytes, preset_path: Path) -> Optional[Dict[str, Any]]:
        """Decode SERUM2PRESET format preset files (compressed CBOR)."""
        try:
            # Skip magic header
            offset = len(self.serum2_header)
            
            # Read version (4 bytes, little endian)
            if len(data) < offset + 4:
                print(f"Error: {preset_path} is too short")
                return None
            
            version = struct.unpack('<I', data[offset:offset+4])[0]
            offset += 4
            
            # Read compressed data length (4 bytes, little endian)
            if len(data) < offset + 4:
                print(f"Error: {preset_path} is too short")
                return None
            
            compressed_length = struct.unpack('<I', data[offset:offset+4])[0]
            offset += 4
            
            # Read compressed data
            if len(data) < offset + compressed_length:
                print(f"Error: {preset_path} compressed data is incomplete")
                return None
            
            compressed_data = data[offset:offset+compressed_length]
            
            # Decompress using zstandard
            try:
                dctx = zstd.ZstdDecompressor()
                decompressed_data = dctx.decompress(compressed_data)
            except Exception as e:
                print(f"Error decompressing {preset_path}: {e}")
                return None
            
            # Decode CBOR data
            try:
                preset_data = cbor2.loads(decompressed_data)
            except Exception as e:
                print(f"Error decoding CBOR from {preset_path}: {e}")
                return None
            
            # Create metadata
            metadata = {
                "fileType": "SerumPreset",
                "version": version,
                "presetName": preset_path.stem,
                "product": "Serum2",
                "vendor": "Xfer Records",
                "hash": hashlib.md5(data).hexdigest()
            }
            
            # Structure the result
            result = {
                "metadata": metadata,
                "data": preset_data
            }
            
            return result
            
        except Exception as e:
            print(f"Error processing SERUM2PRESET format {preset_path}: {e}")
            return None


class Serum2PresetConverter:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.decoder = Serum2PresetDecoder()
        
    def find_preset_files(self) -> List[Path]:
        """Recursively find all .SerumPreset files in the input directory."""
        preset_files = []
        for root, dirs, files in os.walk(self.input_dir):
            for file in files:
                if file.endswith('.SerumPreset'):
                    preset_files.append(Path(root) / file)
        return preset_files
    
    def convert_preset_to_json(self, preset_path: Path) -> Optional[Dict[str, Any]]:
        """Convert a single .SerumPreset file to JSON using Python decoder."""
        preset_data = self.decoder.decode_preset_file(preset_path)
        
        if preset_data:
            # Extract parameters for DawDreamer
            dawdreamer_params = {}
            if 'data' in preset_data:
                data = preset_data['data']
                if 'parameters' in data:
                    # Direct parameters from binary parsing
                    for param_name, param_value in data['parameters'].items():
                        dawdreamer_params[param_name] = float(param_value)
                else:
                    # Extract from CBOR structure
                    self._extract_section_parameters('', data, dawdreamer_params)
            
            # Add dawdreamer_params to preset_data
            if dawdreamer_params:
                preset_data['dawdreamer_params'] = dawdreamer_params
        
        return preset_data
    
    def save_json_preset(self, preset_data: Dict[str, Any], original_path: Path):
        """Save the converted preset data as JSON file."""
        try:
            # Try to create relative path structure in output directory
            rel_path = original_path.relative_to(self.input_dir)
            output_path = self.output_dir / rel_path.with_suffix('.json')
        except ValueError:
            # If the file is not in the input directory (e.g., during testing),
            # just use the filename
            output_path = self.output_dir / original_path.with_suffix('.json').name
        
        # Create parent directories if they don't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(preset_data, f, indent=2, ensure_ascii=False)
        
        print(f"Converted: {original_path} -> {output_path}")
        return output_path
    
    def convert_all_presets(self, limit=None):
        """Convert all found preset files to JSON format."""
        preset_files = self.find_preset_files()
        print(f"Found {len(preset_files)} .SerumPreset files")
        
        if limit:
            preset_files = preset_files[:limit]
            print(f"Converting first {len(preset_files)} presets")
        
        successful_conversions = 0
        failed_conversions = 0
        
        for preset_file in preset_files:
            print(f"Converting: {preset_file}")
            preset_data = self.convert_preset_to_json(preset_file)
            
            if preset_data is not None:
                self.save_json_preset(preset_data, preset_file)
                successful_conversions += 1
            else:
                failed_conversions += 1
        
        print(f"\nConversion complete:")
        print(f"Successful: {successful_conversions}")
        print(f"Failed: {failed_conversions}")
        print(f"Total: {len(preset_files)}")
    
    def extract_parameters_for_dawdreamer(self, json_preset_path: Path) -> Dict[str, float]:
        """Extract parameter values from JSON preset for use with DawDreamer."""
        with open(json_preset_path, 'r', encoding='utf-8') as f:
            preset_data = json.load(f)
        
        # Extract parameters from the preset data structure
        parameters = {}
        
        # Navigate through the preset data structure to extract parameter values
        if 'data' in preset_data:
            data = preset_data['data']
            
            # Handle the new parameter structure
            if 'parameters' in data:
                # Direct parameters from binary parsing
                for param_name, param_value in data['parameters'].items():
                    parameters[param_name] = float(param_value)
            else:
                # Extract various parameter sections (legacy format)
                for section_name, section_data in data.items():
                    if isinstance(section_data, dict):
                        # Handle different parameter structures
                        self._extract_section_parameters(section_name, section_data, parameters)
        
        return parameters
    
    def _extract_section_parameters(self, section_name: str, section_data: dict, parameters: dict):
        """Recursively extract parameters from a section."""
        for key, value in section_data.items():
            if isinstance(value, (int, float)):
                # Direct numeric parameter
                param_name = f"{section_name}.{key}"
                parameters[param_name] = float(value)
            elif isinstance(value, dict):
                # Nested parameters
                if key == 'plainParams' and isinstance(value, dict):
                    # Handle plainParams specifically
                    for param_name, param_value in value.items():
                        if isinstance(param_value, (int, float)):
                            full_param_name = f"{section_name}.{param_name}"
                            parameters[full_param_name] = float(param_value)
                else:
                    # Recursively handle nested dictionaries
                    nested_section = f"{section_name}.{key}"
                    self._extract_section_parameters(nested_section, value, parameters)
            elif isinstance(value, list):
                # Handle arrays of parameters
                for i, item in enumerate(value):
                    if isinstance(item, (int, float)):
                        param_name = f"{section_name}.{key}[{i}]"
                        parameters[param_name] = float(item)
                    elif isinstance(item, dict):
                        nested_section = f"{section_name}.{key}[{i}]"
                        self._extract_section_parameters(nested_section, item, parameters)


def install_dependencies():
    """Install required Python dependencies."""
    try:
        import zstandard
        import cbor2
        print("Dependencies already installed.")
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Installing required packages...")
        
        packages = ['zstandard', 'cbor2']
        for package in packages:
            try:
                subprocess.check_call(['pip', 'install', package])
                print(f"Successfully installed {package}")
            except subprocess.CalledProcessError:
                print(f"Failed to install {package}. Please install manually: pip install {package}")
                return False
    return True


def main():
    """Main function to handle command line arguments and run conversion."""
    parser = argparse.ArgumentParser(description='Convert Serum 2 presets to JSON format')
    parser.add_argument('--input-dir', '-i', 
                       default='/Library/Audio/Presets/Xfer Records/Serum 2 Presets/Presets/Factory',
                       help='Input directory containing .SerumPreset files')
    parser.add_argument('--output-dir', '-o',
                       default='/Users/gjb/Projects/flow_synthesizer/converted_presets',
                       help='Output directory for converted JSON files')
    parser.add_argument('--install-deps', action='store_true',
                       help='Install required dependencies')
    parser.add_argument('--extract-params', action='store_true',
                       help='Extract parameters for DawDreamer integration')
    parser.add_argument('--limit', '-l', type=int, default=25,
                       help='Limit number of presets to convert (default: 25)')
    
    args = parser.parse_args()
    
    if args.install_deps:
        install_dependencies()
        return
    
    # Ensure input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        print("Please provide a valid directory containing .SerumPreset files")
        return
    
    # Create converter and run conversion
    converter = Serum2PresetConverter(args.input_dir, args.output_dir)
    
    print(f"Converting presets from {args.input_dir} to {args.output_dir}")
    print(f"Limiting conversion to {args.limit} presets")
    converter.convert_all_presets(limit=args.limit)
    
    if args.extract_params:
        print("\nExtracting parameters for DawDreamer integration...")
        # Find converted JSON files and extract parameters
        json_files = list(Path(args.output_dir).rglob('*.json'))
        for json_file in json_files[:5]:  # Process first 5 for testing
            params = converter.extract_parameters_for_dawdreamer(json_file)
            if params:
                print(f"Extracted {len(params)} parameters from {json_file.name}")
    
    print("\nConversion complete!")


if __name__ == '__main__':
    main()