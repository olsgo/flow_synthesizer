#!/usr/bin/env python3
"""
POC-Inspired Serum 2 Preset Loader

Based on strategies from poc-audio-pedalboard project for VST3 plugin state management.
This loader focuses on proper raw_state handling, XML extraction, and parameter tracking
for Serum 2 preset loading and audio synthesis.
"""

import os
import sys
import json
import hashlib
import argparse
import struct
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import cbor2
import zstandard

# Add the code directory to Python path
sys.path.append(str(Path(__file__).parent / 'code'))
try:
    from serum2_mapping import SerumPresetMapper
    HAVE_SERUM_MAPPER = True
except Exception:
    HAVE_SERUM_MAPPER = False

try:
    import pedalboard
    from pedalboard import VST3Plugin, load_plugin
    PEDALBOARD_AVAILABLE = True
except ImportError:
    print("Error: pedalboard is required but not available")
    PEDALBOARD_AVAILABLE = False
    sys.exit(1)

try:
    from pedalboard import MIDIMessage
    HAVE_MIDI_MESSAGE = True
except ImportError:
    HAVE_MIDI_MESSAGE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VST3StateManager:
    """
    VST3 state management utilities based on poc-audio-pedalboard strategies
    """
    
    @staticmethod
    def is_vst3_xml(raw_state: bytes) -> bool:
        """
        Check if the raw state looks like VST3 XML.
        
        This function checks if the 9th to 13th bytes are '<?xml' and the last byte is a null byte (0x00).
        
        Args:
            raw_state: The raw state to check.
            
        Returns:
            True if the raw state looks like VST3 XML, False otherwise.
        """
        return (
            len(raw_state) > 13 and
            raw_state[8:13] == b'<?xml' and
            raw_state[-1] == 0x00
        )
    
    @staticmethod
    def extract_vst3_xml(raw_state: bytes) -> Optional[str]:
        """
        Extract the XML content from the VST3 raw state if it is valid.
        
        Args:
            raw_state: The raw state to extract XML from.
            
        Returns:
            The extracted XML content if valid, otherwise None.
        """
        if VST3StateManager.is_vst3_xml(raw_state):
            try:
                # Extract the bytes between the 8-byte header and the null byte
                xml_part = raw_state[8:-1].decode('utf-8')
                return xml_part
            except UnicodeDecodeError:
                pass
        return None
    
    @staticmethod
    def save_state_to_file(raw_state: bytes, filepath: str) -> bool:
        """
        Save raw plugin state to file
        
        Args:
            raw_state: Plugin raw state bytes
            filepath: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'wb') as f:
                f.write(raw_state)
            return True
        except Exception as e:
            print(f"Error saving state to {filepath}: {e}")
            return False
    
    @staticmethod
    def load_state_from_file(filepath: str) -> Optional[bytes]:
        """
        Load raw plugin state from file
        
        Args:
            filepath: Input file path
            
        Returns:
            Raw state bytes if successful, None otherwise
        """
        try:
            with open(filepath, 'rb') as f:
                return f.read()
        except Exception as e:
            print(f"Error loading state from {filepath}: {e}")
            return None


class PluginParameterTracker:
    """
    Track plugin parameter changes for debugging and verification
    """
    
    def __init__(self, plugin):
        self.plugin = plugin
        self.snapshots = {}
    
    def capture_parameters(self, snapshot_name: str) -> Dict[str, Any]:
        """
        Capture current plugin parameters
        
        Args:
            snapshot_name: Name for this parameter snapshot
            
        Returns:
            Dictionary of parameter names to values
        """
        try:
            params = {}
            for name, param in self.plugin.parameters.items():
                try:
                    params[name] = {
                        'raw_value': param.raw_value,
                        'string_value': param.string_value,
                        'range': (param.min_value, param.max_value)
                    }
                except Exception as e:
                    params[name] = {'error': str(e)}
            
            self.snapshots[snapshot_name] = params
            return params
            
        except Exception as e:
            print(f"Error capturing parameters: {e}")
            return {}
    
    def compare_snapshots(self, snapshot1: str, snapshot2: str) -> Dict[str, Any]:
        """
        Compare two parameter snapshots
        
        Args:
            snapshot1: First snapshot name
            snapshot2: Second snapshot name
            
        Returns:
            Dictionary of differences
        """
        if snapshot1 not in self.snapshots or snapshot2 not in self.snapshots:
            return {'error': 'One or both snapshots not found'}
        
        params1 = self.snapshots[snapshot1]
        params2 = self.snapshots[snapshot2]
        
        differences = {}
        
        # Find changed parameters
        for name in params1:
            if name in params2:
                val1 = params1[name].get('raw_value')
                val2 = params2[name].get('raw_value')
                if val1 != val2:
                    differences[name] = {
                        'before': val1,
                        'after': val2,
                        'change': val2 - val1 if isinstance(val1, (int, float)) and isinstance(val2, (int, float)) else 'N/A'
                    }
        
        return differences


class POCInspiredSerumLoader:
    """
    Serum 2 preset loader inspired by poc-audio-pedalboard strategies
    """
    
    def __init__(self, plugin_path: str = '/Library/Audio/Plug-Ins/VST3/Serum2.vst3'):
        self.plugin_path = plugin_path
        self.plugin = None
        self.state_manager = VST3StateManager()
        self.param_tracker = None
        self.sample_rate = 44100
        self.parameter_mapping = self._create_parameter_mapping()
        self.initial_raw_state: Optional[bytes] = None
        
        self._load_plugin()
    
    def _unpack_serum_preset(self, preset_path: str) -> Optional[Dict[str, Any]]:
        """
        Unpack a .SerumPreset file and extract parameter data
        
        Args:
            preset_path: Path to .SerumPreset file
            
        Returns:
            Dictionary with preset data if successful, None otherwise
        """
        try:
            with open(preset_path, 'rb') as f:
                data = f.read()
            
            logger.debug(f"Read {len(data)} bytes from {preset_path}")
            logger.debug(f"First 32 bytes: {data[:32].hex()}")
            
            # Check for XferJson format
            if data.startswith(b'XferJson'):
                logger.debug("Detected XferJson format")
                # Skip the header (8 bytes "XferJson" + 8 bytes header info)
                json_data = data[16:]
                
                # Skip any leading null bytes
                while json_data and json_data[0] == 0:
                    json_data = json_data[1:]
                
                try:
                    # Find the end of the JSON (look for closing brace followed by null or end)
                    # First, try to find a reasonable end point
                    json_str = json_data.decode('utf-8', errors='ignore')
                    
                    # Find the last closing brace that could end the JSON
                    brace_count = 0
                    end_pos = -1
                    for i, char in enumerate(json_str):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_pos = i + 1
                                break
                    
                    if end_pos > 0:
                        json_str = json_str[:end_pos]
                    
                    metadata = json.loads(json_str)
                    logger.debug(f"Successfully decoded XferJson metadata: {preset_path}")
                    if isinstance(metadata, dict):
                        logger.debug(f"Metadata keys: {list(metadata.keys())}")
                    
                    # Extract binary data after JSON
                    json_bytes_end = 16 + 1 + end_pos  # header + null + json length
                    binary_data = data[json_bytes_end:]
                    logger.debug(f"Found {len(binary_data)} bytes of binary data")
                    
                    # Try to decompress and decode the binary data as CBOR
                    try:
                        # The binary data has an 8-byte header, then zstd compressed CBOR
                        if len(binary_data) > 8 and binary_data[8:12] == b'\x28\xb5\x2f\xfd':
                            decompressed = zstandard.ZstdDecompressor().decompress(binary_data[8:])
                            cbor_data = cbor2.loads(decompressed)
                            
                            logger.debug(f"Successfully decoded CBOR data with {len(cbor_data)} sections")
                            
                            # Extract parameters from CBOR data
                            parameters = {}
                            for section_name, section_data in cbor_data.items():
                                if isinstance(section_data, dict) and 'plainParams' in section_data:
                                    plain_params = section_data['plainParams']
                                    if isinstance(plain_params, dict):
                                        for param_name, param_value in plain_params.items():
                                            if isinstance(param_value, (int, float)):
                                                parameters[param_name] = param_value
                            
                            logger.debug(f"Extracted {len(parameters)} parameters from CBOR data")
                            
                            return {
                                'metadata': metadata,
                                'parameters': parameters,
                                'cbor_data': cbor_data,
                                'preset_type': 'xfer_json'
                            }
                    except Exception as e:
                        logger.warning(f"Failed to decode CBOR data: {e}")
                    
                    # Return both metadata and binary data
                    return {
                        'metadata': metadata,
                        'binary_data': binary_data,
                        'preset_type': 'xfer_json'
                    }
                except Exception as e:
                    logger.debug(f"Failed to decode XferJson: {e}")
            
            # Check for CBOR format (starts with specific bytes)
            if data[:4] == b'\x9f\x64\x64\x61\x74\x61':
                # Try CBOR decoding
                try:
                    preset_data = cbor2.loads(data)
                    logger.info(f"Successfully decoded CBOR preset: {preset_path}")
                    if isinstance(preset_data, dict):
                        logger.debug(f"Decoded data keys: {list(preset_data.keys())}")
                    return preset_data
                except Exception as e:
                    logger.warning(f"CBOR decode failed: {e}")
            
            # Check for Zstandard compression
            try:
                dctx = zstandard.ZstdDecompressor()
                decompressed = dctx.decompress(data)
                logger.debug(f"Decompressed to {len(decompressed)} bytes")
                
                # Try CBOR on decompressed data
                try:
                    preset_data = cbor2.loads(decompressed)
                    logger.info(f"Successfully decoded Zstd+CBOR preset: {preset_path}")
                    if isinstance(preset_data, dict):
                        logger.debug(f"Decoded data keys: {list(preset_data.keys())}")
                    return preset_data
                except Exception as e:
                    logger.debug(f"Zstd+CBOR decode failed: {e}")
                    # Try JSON on decompressed data
                    try:
                        preset_data = json.loads(decompressed.decode('utf-8'))
                        logger.info(f"Successfully decoded Zstd+JSON preset: {preset_path}")
                        if isinstance(preset_data, dict):
                            logger.debug(f"Decoded data keys: {list(preset_data.keys())}")
                        return preset_data
                    except Exception as e2:
                        logger.debug(f"Zstd+JSON decode failed: {e2}")
            except Exception as e:
                logger.debug(f"Zstandard decompression failed: {e}")
            
            # Try direct JSON decode
            try:
                preset_data = json.loads(data.decode('utf-8'))
                logger.info(f"Successfully decoded JSON preset: {preset_path}")
                if isinstance(preset_data, dict):
                    logger.debug(f"Decoded data keys: {list(preset_data.keys())}")
                return preset_data
            except Exception as e:
                logger.debug(f"Direct JSON decode failed: {e}")
            
            logger.error(f"Could not decode preset file: {preset_path}")
            return None
            
        except Exception as e:
            logger.error(f"Error reading preset file {preset_path}: {e}")
            return None
    
    def load_serum_preset(self, preset_path: str) -> bool:
        """
        Load a .SerumPreset file
        
        Args:
            preset_path: Path to .SerumPreset file
            
        Returns:
            True if successful, False otherwise
        """
        preset_data = self._unpack_serum_preset(preset_path)
        if preset_data is None:
            return False
        
        # Handle XferJson format with CBOR parameters
        if isinstance(preset_data, dict) and preset_data.get('preset_type') == 'xfer_json':
            # PRIMARY: Apply decoded parameters from the preset (if available)
            parameters = preset_data.get('parameters')
            if parameters:
                logger.info(f"Applying decoded preset parameters ({len(parameters)})")
                if self.load_json_preset(parameters):
                    # After a successful apply, extract the 512 parameters for metadata
                    extracted = self._extract_comprehensive_parameters()
                    if extracted:
                        preset_data['comprehensive_parameters'] = extracted
                    return True
                else:
                    logger.warning("Decoded parameters did not change plugin state; trying fallbacks")
            
            # FALLBACK 1: Try to apply parameters from nested CBOR data
            cbor_data = preset_data.get('cbor_data')
            if cbor_data:
                logger.info("Attempting to apply parameters from nested CBOR data")
                nested_params = self._extract_nested_parameters(cbor_data)
                if nested_params and self.load_json_preset(nested_params):
                    extracted = self._extract_comprehensive_parameters()
                    if extracted:
                        preset_data['comprehensive_parameters'] = extracted
                    return True
            
            # FALLBACK 2: As a last resort, try raw_state assignment with full preset bytes
            try:
                with open(preset_path, 'rb') as f:
                    full_bytes = f.read()
                self.param_tracker.capture_parameters('before_binary_load')
                self.plugin.raw_state = full_bytes
                self.param_tracker.capture_parameters('after_binary_load')
                changes = self.param_tracker.compare_snapshots('before_binary_load', 'after_binary_load')
                if changes:
                    logger.info(f"✓ Raw preset applied via raw_state - {len(changes)} parameters changed")
                    extracted = self._extract_comprehensive_parameters()
                    if extracted:
                        preset_data['comprehensive_parameters'] = extracted
                    return True
                else:
                    logger.warning("Raw preset assigned but no parameter changes detected")
            except Exception as e:
                logger.warning(f"Failed raw_state fallback: {e}")
        
        # Extract parameters from CBOR data if available (generic path)
        cbor_data = preset_data.get('cbor_data', {})
        if cbor_data:
            logger.info(f"Extracting parameters from CBOR data ({len(cbor_data)} top-level entries)")
            nested_params = self._extract_nested_parameters(cbor_data)
            if nested_params:
                logger.info(f"Found {len(nested_params)} nested kParam entries; applying")
                if self.load_json_preset(nested_params):
                    extracted = self._extract_comprehensive_parameters()
                    if extracted:
                        preset_data['comprehensive_parameters'] = extracted
                    return True
        
        # Extract parameters from preset data structure (fallback)
        parameters = {}
        
        # Handle different preset data structures
        if isinstance(preset_data, dict):
            # Look for common parameter containers
            for key in ['parameters', 'params', 'data', 'preset']:
                if key in preset_data:
                    if isinstance(preset_data[key], dict):
                        parameters.update(preset_data[key])
                    break
            else:
                # Use the preset data directly if no container found
                parameters = preset_data
        
        if not parameters:
            logger.warning(f"No parameters found in preset: {preset_path}")
            return False
        
        logger.info(f"Loading preset with {len(parameters)} parameters")
        if self.load_json_preset(parameters):
            extracted = self._extract_comprehensive_parameters()
            if extracted:
                preset_data['comprehensive_parameters'] = extracted
            return True
        return False
    
    def _load_plugin(self):
        """
        Load the Serum 2 VST3 plugin using poc-audio-pedalboard approach
        """
        try:
            # Use load_plugin for automatic detection, then specify plugin_name for Serum 2
            if 'Serum2.vst3' in self.plugin_path:
                self.plugin = VST3Plugin(self.plugin_path, plugin_name="Serum 2")
            else:
                self.plugin = load_plugin(self.plugin_path)
            
            print(f"✓ Successfully loaded plugin: {self.plugin.name}")
            print(f"  Is instrument: {self.plugin.is_instrument}")
            print(f"  Is effect: {self.plugin.is_effect}")
            print(f"  Parameter count: {len(self.plugin.parameters)}")
            
            # Initialize parameter tracker
            self.param_tracker = PluginParameterTracker(self.plugin)
            
            # Capture initial state
            self.param_tracker.capture_parameters('initial')
            try:
                self.initial_raw_state = bytes(self.plugin.raw_state)
            except Exception:
                self.initial_raw_state = None
            
        except Exception as e:
            print(f"✗ Failed to load plugin {self.plugin_path}: {e}")
            raise
    
    def get_plugin_state_info(self) -> Dict[str, Any]:
        """
        Get detailed information about current plugin state
        
        Returns:
            Dictionary with state information
        """
        if not self.plugin:
            return {'error': 'No plugin loaded'}
        
        try:
            raw_state = self.plugin.raw_state
            
            info = {
                'raw_state_length': len(raw_state),
                'is_vst3_xml': self.state_manager.is_vst3_xml(raw_state),
                'state_hash': hashlib.md5(raw_state).hexdigest(),
                'parameter_count': len(self.plugin.parameters)
            }
            
            # Try to extract XML if it's VST3 format
            if info['is_vst3_xml']:
                xml_content = self.state_manager.extract_vst3_xml(raw_state)
                if xml_content:
                    info['xml_length'] = len(xml_content)
                    info['xml_preview'] = xml_content[:200] + '...' if len(xml_content) > 200 else xml_content
            
            return info
            
        except Exception as e:
            return {'error': str(e)}
    
    def save_current_state(self, filepath: str, include_xml: bool = True) -> bool:
        """
        Save current plugin state to file(s)
        
        Args:
            filepath: Base filepath for state files
            include_xml: Whether to also save extracted XML
            
        Returns:
            True if successful, False otherwise
        """
        if not self.plugin:
            print("No plugin loaded")
            return False

    def reset_plugin_state(self) -> bool:
        """Reset plugin to its initial state captured at load time."""
        if not self.plugin:
            return False
        if not self.initial_raw_state:
            # Nothing to reset to
            return True
        try:
            self.plugin.raw_state = self.initial_raw_state
            # Refresh tracker snapshot for clarity
            self.param_tracker.capture_parameters('after_reset')
            return True
        except Exception as e:
            logger.warning(f"Failed to reset plugin state: {e}")
            return False
        
        try:
            raw_state = self.plugin.raw_state
            
            # Save raw state
            success = self.state_manager.save_state_to_file(raw_state, filepath)
            if success:
                print(f"✓ Saved raw state to {filepath}")
            
            # Save XML if requested and available
            if include_xml and self.state_manager.is_vst3_xml(raw_state):
                xml_content = self.state_manager.extract_vst3_xml(raw_state)
                if xml_content:
                    xml_filepath = filepath.replace('.bin', '.xml')
                    try:
                        with open(xml_filepath, 'w') as f:
                            f.write(xml_content)
                        print(f"✓ Saved XML content to {xml_filepath}")
                    except Exception as e:
                        print(f"✗ Failed to save XML: {e}")
            
            return success
            
        except Exception as e:
            print(f"✗ Error saving state: {e}")
            return False
    
    def load_state_from_file(self, filepath: str) -> bool:
        """
        Load plugin state from file
        
        Args:
            filepath: Path to state file
            
        Returns:
            True if successful, False otherwise
        """
        if not self.plugin:
            print("No plugin loaded")
            return False
        
        raw_state = self.state_manager.load_state_from_file(filepath)
        if raw_state is None:
            return False
        
        try:
            # Capture state before loading
            self.param_tracker.capture_parameters('before_load')
            
            # Load the state
            self.plugin.raw_state = raw_state
            
            # Capture state after loading
            self.param_tracker.capture_parameters('after_load')
            
            print(f"✓ Loaded state from {filepath}")
            
            # Show parameter changes
            changes = self.param_tracker.compare_snapshots('before_load', 'after_load')
            if changes:
                print(f"  {len(changes)} parameters changed")
                # Show first few changes
                for i, (name, change) in enumerate(changes.items()):
                    if i >= 5:  # Limit output
                        print(f"  ... and {len(changes) - 5} more")
                        break
                    print(f"    {name}: {change['before']} → {change['after']}")
            else:
                print("  No parameter changes detected")
            
            return True
            
        except Exception as e:
            print(f"✗ Error loading state: {e}")
            return False
    
    def _create_parameter_mapping(self) -> dict:
        """Create comprehensive mapping for 512 essential Serum 2 parameters"""
        try:
            # Load the 512 essential parameters mapping
            mapping_file = Path(__file__).parent / 'serum2_512_parameter_mapping.json'
            if mapping_file.exists():
                with open(mapping_file, 'r') as f:
                    comprehensive_mapping = json.load(f)
                logger.info(f"Loaded comprehensive mapping with {len(comprehensive_mapping)} parameters")
                return comprehensive_mapping
            else:
                logger.warning(f"512-parameter mapping file not found: {mapping_file}")
        except Exception as e:
            logger.error(f"Error loading comprehensive parameter mapping: {e}")
        
        # Fallback to basic mapping if comprehensive mapping fails
        logger.info("Using fallback basic parameter mapping")
        return {
            # Main section
            "main: Volume": "main_vol",
            "main: Bypass": "amp",
            
            # OSC A section
            "OSC A: Volume": "a_level",
            "OSC A: Pan": "a_pan",
            "OSC A: Coarse": "a_coarse_pitch",
            "OSC A: Fine": "a_fine_cents",
            "OSC A: Phase": "a_phase",
            "OSC A: Wavetable Position": "a_wt_pos",
            "OSC A: Unison": "a_unison",
            
            # OSC B section
            "OSC B: Volume": "b_level",
            "OSC B: Pan": "b_pan",
            "OSC B: Coarse": "b_coarse_pitch",
            "OSC B: Fine": "b_fine_cents",
            "OSC B: Phase": "b_phase",
            "OSC B: Wavetable Position": "b_wt_pos",
            "OSC B: Unison": "b_unison",
            
            # SUB section
            "SUB: Volume": "sub_level",
            "SUB: Pan": "sub_pan",
            
            # NOISE section
            "NOISE: Volume": "noise_level",
            "NOISE: Pan": "noise_pan",
            
            # FILTER section
            "FILTER: Cutoff": "filter_1_freq_hz",
            "FILTER: Resonance": "filter_1_res",
            "FILTER: Drive": "filter_1_drive",
            
            # ENV1 section
            "ENV1: Attack": "env_1_attack",
            "ENV1: Decay": "env_1_decay",
            "ENV1: Sustain": "env_1_sustain",
            "ENV1: Release": "env_1_release",
        }
    
    def _extract_nested_parameters(self, cbor_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract nested kParam entries from CBOR data structure.
        
        Serum presets store parameters in nested structures like:
        - Env0.plainParams.kParamCurve1
        - Global0.plainParams.kParamMasterVolume
        - OSC0.plainParams.kParamLevel
        
        Args:
            cbor_data: The decoded CBOR data from the preset
            
        Returns:
            Dictionary of kParam names to values
        """
        nested_params = {}
        
        def extract_kparams(data, path=""):
            if isinstance(data, dict):
                for key, value in data.items():
                    current_path = f"{path}.{key}" if path else key
                    # If this is a kParam, include the context in the key (e.g., Env0.kParamAttack)
                    if isinstance(key, str) and key.lower().startswith('kparam') and isinstance(value, (int, float)):
                        nested_params[current_path] = value
                        logger.debug(f"Found parameter: {current_path} = {value}")
                    elif isinstance(value, dict):
                        extract_kparams(value, current_path)

        extract_kparams(cbor_data)

        logger.info(f"Extracted {len(nested_params)} kParam entries from CBOR data (with context)")
        return nested_params
    
    def _extract_comprehensive_parameters(self) -> Dict[str, Any]:
        """
        Extract all 512 essential parameters directly from the plugin state
        
        Returns:
            Dictionary of extracted parameters with their current values
        """
        extracted_params = {}
        
        if not self.plugin:
            logger.error("Plugin not loaded, cannot extract parameters")
            return extracted_params
        
        try:
            # Load the list of 512 essential parameter names
            param_names_file = Path(__file__).parent / 'serum2_512_parameter_names.txt'
            if not param_names_file.exists():
                logger.error(f"512 parameter names file not found: {param_names_file}")
                return extracted_params
            
            with open(param_names_file, 'r') as f:
                essential_param_names = [line.strip() for line in f if line.strip()]
            
            logger.info(f"Loading {len(essential_param_names)} essential parameters")
            
            # Extract current values for all essential parameters
            if hasattr(self.plugin, 'parameters'):
                plugin_params = self.plugin.parameters
                
                for param_name in essential_param_names:
                    try:
                        if param_name in plugin_params:
                            param_obj = plugin_params[param_name]
                            # Get the current value
                            if hasattr(param_obj, 'value'):
                                extracted_params[param_name] = param_obj.value
                            elif hasattr(param_obj, 'raw_value'):
                                extracted_params[param_name] = param_obj.raw_value
                            else:
                                extracted_params[param_name] = 0.0  # Default value
                        else:
                            logger.debug(f"Parameter {param_name} not found in plugin")
                            extracted_params[param_name] = 0.0  # Default value
                    except Exception as e:
                        logger.debug(f"Error extracting parameter {param_name}: {e}")
                        extracted_params[param_name] = 0.0  # Default value
            
            logger.info(f"Successfully extracted {len(extracted_params)} parameters")
            return extracted_params
            
        except Exception as e:
            logger.error(f"Error in comprehensive parameter extraction: {e}")
            return extracted_params
    
    def _try_set_parameter(self, param_name: str, value: Any) -> bool:
        """
        Try to set a parameter with various name mappings
        
        Args:
            param_name: Parameter name to try
            value: Value to set
            
        Returns:
            True if successful, False otherwise
        """
        # Specific parameter mappings for known conversions
        param_mappings = {
            # Main/Master parameters
            'kparammastervolume': 'main_vol',
            'kparamvolume': 'main_vol',
            'kparammainvolume': 'main_vol',
            'kparamlevelout': 'main_vol',
            
            # Oscillator A parameters
            'kparamoscalevel': 'a_level',
            'kparamalevel': 'a_level',
            'kparamoscavolume': 'a_level',
            'kparamoscapan': 'a_pan',
            'kparamapan': 'a_pan',
            'kparamoscacoarse': 'a_coarse_pitch',
            'kparamacoarse': 'a_coarse_pitch',
            'kparamoscafine': 'a_fine_cents',
            'kparamafine': 'a_fine_cents',
            'kparamoscaphase': 'a_phase',
            'kparamaphase': 'a_phase',
            'kparamoscawtpos': 'a_wt_pos',
            'kparamawtpos': 'a_wt_pos',
            'kparamoscawarp': 'a_warp',
            'kparamawarp': 'a_warp',
            'kparamoscaunison': 'a_unison',
            'kparamaunison': 'a_unison',
            'kparamoscadetune': 'a_uni_detune',
            'kparamadetune': 'a_uni_detune',
            
            # Oscillator B parameters
            'kparamoscblevel': 'b_level',
            'kparamblevel': 'b_level',
            'kparamoscbvolume': 'b_level',
            'kparamoscbpan': 'b_pan',
            'kparambpan': 'b_pan',
            'kparamoscbcoarse': 'b_coarse_pitch',
            'kparambcoarse': 'b_coarse_pitch',
            'kparamoscbfine': 'b_fine_cents',
            'kparambfine': 'b_fine_cents',
            'kparamoscbphase': 'b_phase',
            'kparambphase': 'b_phase',
            'kparamoscbwtpos': 'b_wt_pos',
            'kparambwtpos': 'b_wt_pos',
            'kparamoscbwarp': 'b_warp',
            'kparambwarp': 'b_warp',
            'kparamoscbunison': 'b_unison',
            'kparambunison': 'b_unison',
            'kparamoscbdetune': 'b_uni_detune',
            'kparambdetune': 'b_uni_detune',
            
            # Sub oscillator parameters
            'kparamsublevel': 'sub_level',
            'kparamsubvolume': 'sub_level',
            'kparamsubpan': 'sub_pan',
            'kparamsuboctave': 'sub_octave_oct',
            'kparamsubcoarse': 'sub_coarse_pitch',
            'kparamsubfine': 'sub_fine_pitch',
            'kparamsubphase': 'sub_phase',
            
            # Noise parameters
            'kparamnoiselevel': 'noise_level',
            'kparamnoisevolume': 'noise_level',
            'kparamnoisepan': 'noise_pan',
            'kparamnoisecolor': 'noise_pitch',
            'kparamnoisepitch': 'noise_pitch',
            
            # Filter parameters
            'kparamfilterfreq': 'filter_1_freq_hz',
            'kparamfilter1freq': 'filter_1_freq_hz',
            'kparamfreq': 'filter_1_freq_hz',
            'kparamfilterres': 'filter_1_res',
            'kparamfilter1res': 'filter_1_res',
            'kparamreson': 'filter_1_res',
            'kparamreso': 'filter_1_res',
            'kparamfilterdrive': 'filter_1_drive',
            'kparamfilter1drive': 'filter_1_drive',
            'kparamdrive': 'filter_1_drive',
            'kparamfiltertype': 'filter_1_type',
            'kparamfilter1type': 'filter_1_type',
            'kparamtype': 'filter_1_type',
            'kparamfiltermix': 'filter_1_wet',
            'kparamfilter1mix': 'filter_1_wet',
            'kparamfilterwet': 'filter_1_wet',
            'kparamfilterbalance': 'filter_balance',
            'kparamstereo': 'filter_1_stereo',
            
            # Envelope 1 parameters
            'kparamenv1attack': 'env_1_attack',
            'kparamattack': 'env_1_attack',
            'kparamenv1decay': 'env_1_decay',
            'kparamdecay': 'env_1_decay',
            'kparamenv1sustain': 'env_1_sustain',
            'kparamsustain': 'env_1_sustain',
            'kparamenv1release': 'env_1_release',
            'kparamrelease': 'env_1_release',
            'kparamenv1curve1': 'env_1_attack_curve',
            'kparamcurve1': 'env_1_attack_curve',
            'kparamenv1curve2': 'env_1_decay_curve',
            'kparamcurve2': 'env_1_decay_curve',
            'kparamenv1curve3': 'env_1_release_curve',
            'kparamcurve3': 'env_1_release_curve',
            
            # Envelope 2 parameters
            'kparamenv2attack': 'env_2_attack',
            'kparamenv2decay': 'env_2_decay',
            'kparamenv2sustain': 'env_2_sustain',
            'kparamenv2release': 'env_2_release',
            'kparamenv2curve1': 'env_2_attack_curve',
            'kparamenv2curve2': 'env_2_decay_curve',
            'kparamenv2curve3': 'env_2_release_curve',
            
            # LFO parameters
            'kparamlfo1rate': 'lfo_1_rate',
            'kparamlfo1shape': 'lfo_1_shape',
            'kparamlfo1amount': 'lfo_1_amount',
            'kparamlfo2rate': 'lfo_2_rate',
            'kparamlfo2shape': 'lfo_2_shape',
            'kparamlfo2amount': 'lfo_2_amount',
            
            # Effects parameters
            'kparamfxbus1level': 'fx_bus_1_level',
            'kparamfxbus2level': 'fx_bus_2_level',
            'kparamfx1level': 'fx_1_level',
            'kparamfx2level': 'fx_2_level',
            'kparamreverblevel': 'reverb_level',
            'kparamdelaylevel': 'delay_level',
            'kparamchoruslevel': 'chorus_level',
            
            # Modulation parameters
            'kparammodwheel': 'mod_wheel',
            'kparamviaenv1': 'via_env_1',
            'kparamviaenv2': 'via_env_2',
            'kparamvialfo1': 'via_lfo_1',
            'kparamvialfo2': 'via_lfo_2',
            
            # General parameters
            'kparampolycount': 'poly_count',
            'kparamvalue': 'value',
            'kparamamount': 'amount',
            'kparamenable': 'enable',
            'kparambypass': 'bypass',
            'kparammute': 'mute',
            'kparamsolo': 'solo',
            'kparamgain': 'gain',
            'kparamlevel': 'level',
        }
        
        # Normalize parameter name for lookup
        normalized_name = param_name.lower().replace(' ', '').replace('_', '').replace('-', '')
        
        def _apply_by_plugin_name(pname: str, v: Any) -> bool:
            try:
                p = self.plugin.parameters.get(pname)
                if p is None:
                    return False
                # Coerce/scaling based on parameter metadata
                try:
                    min_v = getattr(p, 'min_value', 0.0)
                    max_v = getattr(p, 'max_value', 1.0)
                except Exception:
                    min_v, max_v = 0.0, 1.0
                # Normalize boolean
                try:
                    is_bool = bool(getattr(p, 'is_boolean', False))
                except Exception:
                    is_bool = False
                val = v
                # Discrete/enum mapping via valid_values if available
                try:
                    valid_values = getattr(p, 'valid_values', None)
                except Exception:
                    valid_values = None
                # If value looks normalized and parameter expects a wider range, scale
                try:
                    if not is_bool and isinstance(val, (int, float)):
                        fval = float(val)
                        if valid_values:
                            # Map normalized [0,1] to nearest discrete option
                            try:
                                seq = list(valid_values)
                                if 0.0 <= fval <= 1.0:
                                    idx = int(round(fval * (len(seq) - 1)))
                                    idx = max(0, min(len(seq) - 1, idx))
                                    fval = seq[idx]
                                else:
                                    # Choose nearest discrete value
                                    fval = min(seq, key=lambda x: abs(x - fval))
                            except Exception:
                                pass
                        elif 0.0 <= fval <= 1.0 and (max_v - min_v) > 1.0:
                            fval = min_v + fval * (max_v - min_v)
                        # Clamp
                        if max_v is not None and min_v is not None:
                            if fval < min_v:
                                fval = min_v
                            if fval > max_v:
                                fval = max_v
                        val = fval
                    if is_bool:
                        val = 1.0 if (bool(val) or (isinstance(val, (int, float)) and val > 0.5)) else 0.0
                except Exception:
                    pass
                setattr(self.plugin, pname, val)
                logger.debug(f"✓ Set parameter {pname} = {val}")
                return True
            except Exception as e:
                logger.debug(f"Failed to set {pname}: {e}")
                return False

        # Try direct parameter name
        if param_name in self.plugin.parameters:
            if _apply_by_plugin_name(param_name, value):
                return True
        
        # Try mapped parameter name
        if normalized_name in param_mappings:
            mapped_name = param_mappings[normalized_name]
            if mapped_name in self.plugin.parameters:
                if _apply_by_plugin_name(mapped_name, value):
                    return True
        
        # Try lowercase version
        lower_name = param_name.lower()
        if lower_name in self.plugin.parameters:
            if _apply_by_plugin_name(lower_name, value):
                return True
        
        # Try removing 'kParam' prefix
        if param_name.lower().startswith('kparam'):
            clean_name = param_name[6:]  # Remove 'kParam'
            if clean_name in self.plugin.parameters:
                if _apply_by_plugin_name(clean_name, value):
                    return True
        
        logger.debug(f"✗ Could not find parameter: {param_name}")
        return False

    def load_json_preset(self, json_data: Dict[str, Any]) -> bool:
        """
        Load preset from JSON parameter data
        
        Args:
            json_data: Dictionary of parameter names to values
            
        Returns:
            True if successful, False otherwise
        """
        if not self.plugin:
            logger.error("No plugin loaded")
            return False
        
        try:
            # If available, use the SerumPresetMapper for robust name mapping and scaling
            if HAVE_SERUM_MAPPER:
                try:
                    mapper = SerumPresetMapper(self.plugin)
                    plugin_values, diag = mapper.map_params(json_data)
                    if plugin_values:
                        self.param_tracker.capture_parameters('before_json_load')
                        applied_count = 0
                        for pname, pval in plugin_values.items():
                            if pname in self.plugin.parameters:
                                try:
                                    setattr(self.plugin, pname, pval)
                                    applied_count += 1
                                except Exception:
                                    pass
                        self.param_tracker.capture_parameters('after_json_load')
                        changes = self.param_tracker.compare_snapshots('before_json_load', 'after_json_load')
                        logger.info(f"Applied {applied_count} mapped parameters (mapper)")
                        if changes:
                            logger.info(f"✓ Preset loaded (mapper) - {len(changes)} parameters changed")
                            return True
                        else:
                            logger.warning("No parameters changed via mapper; falling back to heuristic mapping")
                except Exception as e:
                    logger.warning(f"SerumPresetMapper failed, using fallback mapping: {e}")
            # Capture state before loading
            self.param_tracker.capture_parameters('before_json_load')
            
            applied_count = 0
            failed_count = 0
            
            for param_name, param_data in json_data.items():
                # Handle nested parameter structures
                if isinstance(param_data, dict):
                    # Check for plainParams with non-default values
                    if 'plainParams' in param_data and param_data['plainParams'] != 'default':
                        # Try to extract actual parameter values
                        for key, value in param_data.items():
                            if key != 'plainParams' and isinstance(value, (int, float)):
                                # Handle macro parameters (Macro0 -> macro_1)
                                if param_name.startswith('Macro') and 'kParamValue' in key:
                                    try:
                                        macro_num = int(param_name[5:]) + 1  # Macro0 -> 1
                                        macro_param = f'macro_{macro_num}'
                                        if self._try_set_parameter(macro_param, value):
                                            applied_count += 1
                                        else:
                                            failed_count += 1
                                    except (ValueError, IndexError):
                                        failed_count += 1
                                
                                # Handle modulation slot parameters (ModSlot0 -> mod_1_amount)
                                elif param_name.startswith('ModSlot') and ('kParamValue' in key or 'kParamAmount' in key):
                                    try:
                                        mod_num = int(param_name[7:]) + 1  # ModSlot0 -> 1
                                        mod_param = f'mod_{mod_num}_amount'
                                        if self._try_set_parameter(mod_param, value):
                                            applied_count += 1
                                        else:
                                            failed_count += 1
                                    except (ValueError, IndexError):
                                        failed_count += 1
                                
                                # Try generic parameter mapping
                                else:
                                    full_param_name = f"{param_name}_{key}"
                                    if self._try_set_parameter(full_param_name, value):
                                        applied_count += 1
                                    elif self._try_set_parameter(f"{param_name}.{key}", value):
                                        applied_count += 1
                                    else:
                                        failed_count += 1
                    else:
                        # Skip default parameters
                        failed_count += 1
                else:
                    # Direct parameter value
                    if self._try_set_parameter(param_name, param_data):
                        applied_count += 1
                    else:
                        failed_count += 1
            
            # Capture state after loading
            self.param_tracker.capture_parameters('after_json_load')
            
            logger.info(f"Applied {applied_count} parameters, {failed_count} failed")
            
            # Show parameter changes
            changes = self.param_tracker.compare_snapshots('before_json_load', 'after_json_load')
            if changes:
                logger.info(f"✓ Preset loaded successfully - {len(changes)} parameters changed")
                return True
            else:
                logger.warning("No parameters changed")
                return False
            
        except Exception as e:
            logger.error(f"Error loading JSON preset: {e}")
            return False
    
    def render_audio(self, midi_note: int = 60, midi_velocity: int = 100, 
                    note_length: float = 1.0, render_length: float = 2.0) -> Optional[np.ndarray]:
        """
        Render audio using current plugin state
        
        Args:
            midi_note: MIDI note number (0-127)
            midi_velocity: MIDI velocity (0-127)
            note_length: Duration of note in seconds
            render_length: Total render time in seconds
            
        Returns:
            Audio array if successful, None otherwise
        """
        if not self.plugin:
            print("No plugin loaded")
            return None
        
        try:
            # Create MIDI messages
            if HAVE_MIDI_MESSAGE:
                midi_messages = [
                    MIDIMessage.note_on(note=int(midi_note), velocity=int(midi_velocity), time=0.0),
                    MIDIMessage.note_off(note=int(midi_note), velocity=int(midi_velocity), time=float(note_length)),
                ]
            else:
                # Compatibility fallback
                class _CompatMidi:
                    def __init__(self, status, note, velocity, time):
                        self._data = bytes([status, note, velocity])
                        self.time = time
                    def bytes(self):
                        return self._data
                midi_messages = [
                    _CompatMidi(0x90, int(midi_note), int(midi_velocity), 0.0),
                    _CompatMidi(0x80, int(midi_note), int(midi_velocity), float(note_length)),
                ]
            
            # Render audio
            audio_out = self.plugin(
                midi_messages,
                duration=render_length,
                sample_rate=self.sample_rate,
                num_channels=2,
                reset=False
            )
            
            # Ensure correct format (channels, samples)
            if audio_out.ndim == 1:
                # Mono to stereo
                audio_out = np.stack([audio_out, audio_out])
            elif audio_out.shape[0] > audio_out.shape[1]:
                # Transpose if needed (samples, channels) -> (channels, samples)
                audio_out = audio_out.T
            
            return audio_out
            
        except Exception as e:
            print(f"✗ Error rendering audio: {e}")
            return None
    
    def test_preset_loading_workflow(self, preset_file: str = None) -> Dict[str, Any]:
        """
        Test complete preset loading workflow
        
        Args:
            preset_file: Optional JSON preset file to test
            
        Returns:
            Dictionary with test results
        """
        results = {
            'plugin_loaded': self.plugin is not None,
            'initial_state_info': self.get_plugin_state_info(),
            'preset_loaded': False,
            'audio_rendered': False,
            'state_changed': False
        }
        
        if not self.plugin:
            return results
        
        # Test JSON preset loading if provided
        if preset_file and os.path.exists(preset_file):
            try:
                with open(preset_file, 'r') as f:
                    json_data = json.load(f)
                
                results['preset_loaded'] = self.load_json_preset(json_data)
                
                # Check if state actually changed
                changes = self.param_tracker.compare_snapshots('initial', 'after_json_load')
                results['state_changed'] = len(changes) > 0
                results['parameter_changes'] = len(changes)
                
            except Exception as e:
                results['preset_error'] = str(e)
        
        # Test audio rendering
        audio = self.render_audio()
        if audio is not None:
            results['audio_rendered'] = True
            results['audio_shape'] = audio.shape
            results['audio_rms'] = float(np.sqrt(np.mean(audio**2)))
            results['audio_max'] = float(np.max(np.abs(audio)))
        
        return results

    def generate_dataset(self, presets_dir: str, output_dir: str, 
                        limit: Optional[int] = None, test_mode: bool = False) -> Dict[str, Any]:
        """
        Generate dataset from .SerumPreset files
        
        Args:
            presets_dir: Directory containing .SerumPreset files
            output_dir: Directory to save rendered audio and metadata
            limit: Maximum number of presets to process
            test_mode: If True, run in test mode with detailed logging
            
        Returns:
            Dictionary with generation results
        """
        if not self.plugin:
            logger.error("No plugin loaded")
            return {'error': 'No plugin loaded'}
        
        # Discover preset files
        preset_files = []
        for root, dirs, files in os.walk(presets_dir):
            for file in files:
                if file.endswith('.SerumPreset'):
                    preset_files.append(os.path.join(root, file))
        
        if not preset_files:
            logger.error(f"No .SerumPreset files found in {presets_dir}")
            return {'error': 'No presets found'}
        
        logger.info(f"Discovered {len(preset_files)} presets")
        
        if limit:
            preset_files = preset_files[:limit]
            logger.info(f"Limited to {limit} presets")
        
        if test_mode:
            logger.info("Running in test mode")
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        renders_dir = os.path.join(output_dir, 'renders')
        os.makedirs(renders_dir, exist_ok=True)
        
        # Process presets
        results = {
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'presets': []
        }
        
        for i, preset_path in enumerate(preset_files):
            preset_name = os.path.splitext(os.path.basename(preset_path))[0]
            logger.info(f"Processing preset {i+1}/{len(preset_files)}: {preset_name}")
            
            try:
                # Reset to a known default state before applying a new preset
                self.reset_plugin_state()
                # Load preset
                logger.info(f"Loading preset: {preset_name}")
                if not self.load_serum_preset(preset_path):
                    logger.warning(f"Failed to load preset: {preset_name}")
                    results['failed'] += 1
                    results['presets'].append({
                        'name': preset_name,
                        'path': preset_path,
                        'status': 'failed',
                        'error': 'Failed to load preset'
                    })
                    continue
                
                # Render audio
                try:
                    audio = self.render_audio(
                        midi_note=60,
                        midi_velocity=100,
                        note_length=3.0,
                        render_length=4.0
                    )
                    
                    if audio is None:
                        logger.error(f"Failed to render audio for: {preset_name}")
                        results['failed'] += 1
                        results['presets'].append({
                            'name': preset_name,
                            'path': preset_path,
                            'status': 'failed',
                            'error': 'Failed to render audio'
                        })
                        continue
                    
                    # Save audio
                    audio_filename = f"{i:06d}.wav"
                    audio_path = os.path.join(renders_dir, audio_filename)
                    
                    # Ensure audio is in correct format (samples, channels)
                    if audio.shape[0] < audio.shape[1]:
                        audio = audio.T
                    
                    sf.write(audio_path, audio, self.sample_rate)
                    
                    logger.info(f"Successfully processed: {preset_name}")
                    results['successful'] += 1
                    results['presets'].append({
                        'name': preset_name,
                        'path': preset_path,
                        'audio_file': audio_filename,
                        'status': 'success',
                        'audio_rms': float(np.sqrt(np.mean(audio**2))),
                        'audio_max': float(np.max(np.abs(audio)))
                    })
                    
                except Exception as e:
                    logger.error(f"Failed to render audio: {e}")
                    results['failed'] += 1
                    results['presets'].append({
                        'name': preset_name,
                        'path': preset_path,
                        'status': 'failed',
                        'error': f'Audio rendering failed: {str(e)}'
                    })
                
            except Exception as e:
                logger.error(f"Error processing preset {preset_name}: {e}")
                results['failed'] += 1
                results['presets'].append({
                    'name': preset_name,
                    'path': preset_path,
                    'status': 'failed',
                    'error': str(e)
                })
            
            results['processed'] += 1
        
        # Save metadata
        metadata_path = os.path.join(output_dir, 'dataset_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("Dataset generation complete!")
        logger.info(f"Processed: {results['processed']} presets")
        logger.info(f"Successful: {results['successful']} presets")
        logger.info(f"Failed: {results['failed']} presets")
        logger.info(f"Skipped: {results['skipped']} presets")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="POC-Inspired Serum 2 Preset Loader")
    parser.add_argument('--plugin-path', default='/Library/Audio/Plug-Ins/VST3/Serum2.vst3',
                       help='Path to Serum 2 VST3 plugin')
    parser.add_argument('--preset-file', help='JSON preset file to test')
    parser.add_argument('--presets-dir', default='/Library/Audio/Presets/Xfer Records/Serum 2 Presets/Presets/Factory',
                       help='Directory containing .SerumPreset files')
    parser.add_argument('--output-dir', default='/Users/gjb/Datasets/serum2',
                       help='Output directory for dataset')
    parser.add_argument('--generate-dataset', action='store_true',
                       help='Generate dataset from .SerumPreset files')
    parser.add_argument('--limit', type=int, help='Limit number of presets to process')
    parser.add_argument('--test-mode', action='store_true',
                       help='Run in test mode with detailed logging')
    parser.add_argument('--save-state', help='Save current state to file')
    parser.add_argument('--load-state', help='Load state from file')
    parser.add_argument('--test-workflow', action='store_true',
                       help='Run complete preset loading workflow test')
    parser.add_argument('--render-audio', help='Render audio to WAV file')
    parser.add_argument('--enumerate-parameters', action='store_true',
                       help='List all plugin parameters')
    
    args = parser.parse_args()
    
    # Initialize loader
    try:
        loader = POCInspiredSerumLoader(args.plugin_path)
    except Exception as e:
        print(f"Failed to initialize loader: {e}")
        return 1
    
    # Show initial state info
    print("\n=== Initial Plugin State ===")
    state_info = loader.get_plugin_state_info()
    for key, value in state_info.items():
        print(f"{key}: {value}")
    
    # Load state from file if requested
    if args.load_state:
        print(f"\n=== Loading State from {args.load_state} ===")
        loader.load_state_from_file(args.load_state)
    
    # Enumerate parameters if requested
    if args.enumerate_parameters:
        print("\n=== Plugin Parameters ===")
        if loader.plugin:
            for i, (name, param) in enumerate(loader.plugin.parameters.items()):
                try:
                    print(f"{i:3d}: {name} = {param.raw_value} ({param.min_value} to {param.max_value})")
                except Exception as e:
                    print(f"{i:3d}: {name} = ERROR: {e}")
        return 0
    
    # Generate dataset if requested
    if args.generate_dataset:
        print("\n=== Generating Dataset ===")
        if args.test_mode:
            logging.getLogger().setLevel(logging.DEBUG)
        
        results = loader.generate_dataset(
            presets_dir=args.presets_dir,
            output_dir=args.output_dir,
            limit=args.limit,
            test_mode=args.test_mode
        )
        
        print(f"\nDataset Generation Results:")
        print(f"Total processed: {results['processed']}")
        print(f"Successfully processed: {results['successful']}")
        print(f"Failed: {results['failed']}")
        if 'skipped' in results:
            print(f"Skipped: {results['skipped']}")
        if results['processed'] > 0:
            print(f"Success rate: {results['successful']/results['processed']:.1%}")
        else:
            print("No presets processed")
        
        return 0
    
    # Test workflow if requested
    if args.test_workflow:
        print("\n=== Testing Preset Loading Workflow ===")
        results = loader.test_preset_loading_workflow(args.preset_file)
        print(json.dumps(results, indent=2))
    
    # Load JSON preset if provided
    elif args.preset_file:
        print(f"\n=== Loading JSON Preset {args.preset_file} ===")
        try:
            with open(args.preset_file, 'r') as f:
                json_data = json.load(f)
            loader.load_json_preset(json_data)
        except Exception as e:
            print(f"Error loading preset: {e}")
    
    # Save state if requested
    if args.save_state:
        print(f"\n=== Saving State to {args.save_state} ===")
        loader.save_current_state(args.save_state)
    
    # Render audio if requested
    if args.render_audio:
        print(f"\n=== Rendering Audio to {args.render_audio} ===")
        audio = loader.render_audio()
        if audio is not None:
            try:
                # Transpose for soundfile (samples, channels)
                audio_for_save = audio.T
                sf.write(args.render_audio, audio_for_save, loader.sample_rate)
                print(f"✓ Audio saved to {args.render_audio}")
                print(f"  Shape: {audio.shape}, RMS: {np.sqrt(np.mean(audio**2)):.4f}")
            except Exception as e:
                print(f"✗ Error saving audio: {e}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
