#!/usr/bin/env python3
"""
Helper utilities for PolyMAX VST3 preset loading and processing.

Provides utility functions for VST3 state management, preset file handling,
and audio processing specific to the PolyMAX synthesizer.
"""

import os
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import logging
import json
from pedalboard import VST3Plugin

logger = logging.getLogger(__name__)


def is_vst3_xml(data: bytes) -> bool:
    """Check if binary data contains VST3 XML content."""
    try:
        # Look for XML declaration and VST3-specific tags
        xml_markers = [b'<?xml', b'<VST3Preset', b'<Preset']
        return any(marker in data for marker in xml_markers)
    except Exception:
        return False


def extract_vst3_xml(data: bytes) -> Optional[str]:
    """Extract XML content from VST3 binary data."""
    try:
        # Find XML start
        xml_start = data.find(b'<?xml')
        if xml_start == -1:
            # Try alternative XML start
            xml_start = data.find(b'<VST3Preset')
            if xml_start == -1:
                xml_start = data.find(b'<Preset')
                
        if xml_start == -1:
            return None
            
        # Find XML end - look for common VST3 closing tags
        end_tags = [b'</VST3Preset>', b'</Preset>', b'</Root>']
        xml_end = -1
        
        for end_tag in end_tags:
            pos = data.find(end_tag, xml_start)
            if pos != -1:
                xml_end = pos + len(end_tag)
                break
                
        if xml_end == -1:
            # If no closing tag found, try to extract until end of data
            xml_end = len(data)
            
        xml_data = data[xml_start:xml_end]
        return xml_data.decode('utf-8', errors='ignore')
        
    except Exception as e:
        logger.debug(f"Failed to extract XML: {e}")
        return None


def parse_vst3_state_chunk(file_path: str) -> Optional[Dict[str, Any]]:
    """Parse VST3 state chunk from .vstpreset file.
    
    UAD .vstpreset files contain:
    - Binary header with VST3 signature
    - Key-value pairs in a custom format
    - JSON payload embedded in plugin_state_payload
    """
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
            
        logger.debug(f"Reading {len(data)} bytes from {file_path}")
        
        # Look for the JSON start pattern - specifically look for '"controls"'
        # which appears to be the start of the actual JSON content
        json_pattern = b'"controls"'
        controls_pos = data.find(json_pattern)
        
        if controls_pos == -1:
            logger.warning(f"No controls section found in {file_path}")
            return None
            
        # Search backwards from controls to find the opening brace
        json_start = -1
        for i in range(controls_pos, -1, -1):
            if data[i:i+1] == b'{':
                json_start = i
                break
                
        if json_start == -1:
            logger.warning(f"No JSON opening brace found before controls in {file_path}")
            return None
            
        # Extract JSON data starting from the opening brace
        json_data = data[json_start:]
        
        # Find the end of JSON by counting braces
        brace_count = 0
        json_end = 0
        
        for i, byte in enumerate(json_data):
            if byte == ord('{'):
                brace_count += 1
            elif byte == ord('}'):
                brace_count -= 1
                if brace_count == 0:
                    json_end = i + 1
                    break
                    
        if json_end == 0:
            logger.warning(f"Incomplete JSON data in {file_path}")
            return None
             
        # Decode JSON string
        json_str = json_data[:json_end].decode('utf-8', errors='ignore')
         
        # Parse JSON
        parsed_data = json.loads(json_str)
         
        logger.debug(f"Successfully parsed JSON from {file_path}: {len(json_str)} characters")
        return parsed_data
        
    except json.JSONDecodeError as e:
        logger.warning(f"JSON decode error in {file_path}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Error parsing VST3 state chunk from {file_path}: {e}")
        return None


def compare_plugin_states(state1: bytes, state2: bytes) -> Dict[str, Any]:
    """Compare two VST3 plugin states and return differences."""
    comparison = {
        'size_diff': len(state2) - len(state1),
        'identical': state1 == state2,
        'common_length': min(len(state1), len(state2)),
        'differences': []
    }
    
    if not comparison['identical']:
        # Find byte-level differences
        common_len = comparison['common_length']
        diff_count = 0
        
        for i in range(common_len):
            if state1[i] != state2[i] and diff_count < 100:  # Limit to first 100 differences
                comparison['differences'].append({
                    'position': i,
                    'byte1': state1[i],
                    'byte2': state2[i]
                })
                diff_count += 1
                
        comparison['total_byte_differences'] = diff_count
        
    return comparison


def validate_vstpreset_file(file_path: str) -> Dict[str, Any]:
    """Validate and analyze a .vstpreset file."""
    validation = {
        'valid': False,
        'file_exists': False,
        'file_size': 0,
        'has_vst3_header': False,
        'has_xml_content': False,
        'preset_name': None,
        'errors': []
    }
    
    try:
        if not os.path.exists(file_path):
            validation['errors'].append(f"File does not exist: {file_path}")
            return validation
            
        validation['file_exists'] = True
        validation['file_size'] = os.path.getsize(file_path)
        validation['preset_name'] = Path(file_path).stem
        
        with open(file_path, 'rb') as f:
            data = f.read()
            
        # Check for VST3 preset signatures
        vst3_signatures = [b'VST3', b'<?xml', b'<VST3Preset']
        validation['has_vst3_header'] = any(sig in data[:1024] for sig in vst3_signatures)
        
        # Check for XML content
        validation['has_xml_content'] = is_vst3_xml(data)
        
        # Basic validation passed
        validation['valid'] = (
            validation['file_size'] > 0 and 
            (validation['has_vst3_header'] or validation['has_xml_content'])
        )
        
    except Exception as e:
        validation['errors'].append(f"Error validating file: {e}")
        
    return validation


def extract_preset_parameters_from_xml(xml_content: str) -> Dict[str, float]:
    """Extract parameter values from VST3 preset XML content."""
    parameters = {}
    
    try:
        root = ET.fromstring(xml_content)
        
        # Common VST3 XML parameter structures
        param_selectors = [
            './/Param',
            './/Parameter', 
            './/Value',
            './/Obj[@class="Param"]',
            './/Obj[@class="Parameter"]'
        ]
        
        for selector in param_selectors:
            for param_elem in root.findall(selector):
                # Try different attribute names for parameter ID and value
                param_id = (
                    param_elem.get('id') or 
                    param_elem.get('name') or 
                    param_elem.get('key')
                )
                
                param_value = (
                    param_elem.get('value') or 
                    param_elem.get('val') or 
                    param_elem.text
                )
                
                if param_id and param_value:
                    try:
                        parameters[param_id] = float(param_value)
                    except (ValueError, TypeError):
                        # Try to handle normalized values or special formats
                        if isinstance(param_value, str):
                            if param_value.lower() in ['true', 'on', 'yes']:
                                parameters[param_id] = 1.0
                            elif param_value.lower() in ['false', 'off', 'no']:
                                parameters[param_id] = 0.0
                                
    except ET.ParseError as e:
        logger.warning(f"XML parsing error: {e}")
    except Exception as e:
        logger.error(f"Error extracting parameters from XML: {e}")
        
    return parameters


def apply_parameters_to_plugin(plugin: VST3Plugin, parameters: Dict[str, float]) -> Dict[str, bool]:
    """Apply parameter values to a VST3 plugin."""
    results = {}
    
    for param_name, param_value in parameters.items():
        try:
            if param_name in plugin.parameters:
                plugin.parameters[param_name].raw_value = param_value
                results[param_name] = True
            else:
                # Try to find similar parameter names (case-insensitive, etc.)
                found = False
                for plugin_param in plugin.parameters.keys():
                    if plugin_param.lower() == param_name.lower():
                        plugin.parameters[plugin_param].raw_value = param_value
                        results[param_name] = True
                        found = True
                        break
                        
                if not found:
                    results[param_name] = False
                    logger.debug(f"Parameter not found in plugin: {param_name}")
                    
        except Exception as e:
            results[param_name] = False
            logger.warning(f"Failed to set parameter {param_name}: {e}")
            
    return results


def get_plugin_info(plugin: VST3Plugin) -> Dict[str, Any]:
    """Get comprehensive information about a VST3 plugin."""
    info = {
        'name': plugin.name,
        'is_instrument': plugin.is_instrument,
        'is_effect': plugin.is_effect,
        'parameter_count': len(plugin.parameters),
        'parameters': {},
        'state_size': len(plugin.raw_state)
    }
    
    # Get parameter details
    for param_name, param in plugin.parameters.items():
        try:
            info['parameters'][param_name] = {
                'name': param.name,
                'raw_value': param.raw_value,
                'string_value': str(param),
                'range': getattr(param, 'range', None)
            }
        except Exception as e:
            logger.debug(f"Error getting info for parameter {param_name}: {e}")
            
    return info


def create_audio_filename(preset_name: str, extension: str = 'wav') -> str:
    """Create a safe filename for audio output."""
    # Remove or replace problematic characters
    safe_name = preset_name.replace(' ', '_')
    safe_name = ''.join(c for c in safe_name if c.isalnum() or c in '_-')
    return f"{safe_name}.{extension}"


def create_state_filename(preset_name: str, extension: str = 'npz') -> str:
    """Create a safe filename for state output."""
    safe_name = preset_name.replace(' ', '_')
    safe_name = ''.join(c for c in safe_name if c.isalnum() or c in '_-')
    return f"{safe_name}_state.{extension}"


def create_safe_filename(preset_name: str) -> str:
    """Create a safe filename from preset name."""
    safe_name = "".join(c for c in preset_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_name = safe_name.replace(' ', '_')
    return safe_name


def find_vstpreset_files(directory: str, recursive: bool = True) -> List[str]:
    """Find all .vstpreset files in a directory."""
    preset_files = []
    
    try:
        if recursive:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith('.vstpreset'):
                        preset_files.append(os.path.join(root, file))
        else:
            for file in os.listdir(directory):
                if file.lower().endswith('.vstpreset'):
                    preset_files.append(os.path.join(directory, file))
                    
    except Exception as e:
        logger.error(f"Error finding preset files in {directory}: {e}")
        
    return sorted(preset_files)


def estimate_processing_time(num_presets: int, render_duration: float = 4.0) -> Dict[str, float]:
    """Estimate processing time for batch preset processing."""
    # Rough estimates based on typical processing overhead
    load_time_per_preset = 0.5  # seconds
    render_time_per_preset = render_duration + 0.5  # render duration + overhead
    save_time_per_preset = 0.2  # seconds
    
    total_time_per_preset = load_time_per_preset + render_time_per_preset + save_time_per_preset
    total_estimated_time = num_presets * total_time_per_preset
    
    return {
        'num_presets': num_presets,
        'time_per_preset': total_time_per_preset,
        'total_time_seconds': total_estimated_time,
        'total_time_minutes': total_estimated_time / 60,
        'total_time_hours': total_estimated_time / 3600
    }


def print_parameter_properties(param) -> None:
    """Print detailed properties of a VST3 parameter."""
    print(f"Parameter: {param.name}")
    print(f"  Raw value: {param.raw_value}")
    print(f"  String value: {str(param)}")
    
    # Try to access additional properties that may be available
    properties = ['range', 'units', 'label', 'min_value', 'max_value', 'default_value']
    for prop in properties:
        try:
            value = getattr(param, prop, 'N/A')
            print(f"  {prop}: {value}")
        except Exception:
            pass
            
    print()


def filter_installed_plugins_by_names(plugin_names: List[str]) -> None:
    """Find and display installed audio plugins by name."""
    common_plugin_paths = [
        '/Library/Audio/Plug-Ins/VST3',
        '/Library/Audio/Plug-Ins/VST',
        '/Library/Audio/Plug-Ins/Components',
        '~/Library/Audio/Plug-Ins/VST3',
        '~/Library/Audio/Plug-Ins/VST',
        '~/Library/Audio/Plug-Ins/Components'
    ]
    
    for plugin_name in plugin_names:
        print(f"\n{plugin_name}:")
        found = False
        
        for plugin_path in common_plugin_paths:
            expanded_path = os.path.expanduser(plugin_path)
            if os.path.exists(expanded_path):
                for item in os.listdir(expanded_path):
                    if plugin_name.lower() in item.lower():
                        full_path = os.path.join(expanded_path, item)
                        plugin_type = Path(plugin_path).name.upper()
                        print(f"  {plugin_type}: {full_path}")
                        found = True
                        
        if not found:
            print(f"  No installations found for {plugin_name}")