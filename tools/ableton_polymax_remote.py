#!/usr/bin/env python3
"""
Ableton Live Remote Script for PolyMAX Parameter Control

This script creates a remote control interface for Ableton Live that can:
1. Receive predicted PolyMAX parameters from the inference script
2. Automatically apply them to the PolyMAX VST plugin in Live
3. Handle parameter mapping and validation
4. Provide feedback on parameter application status

Usage:
1. Place this script in Ableton Live's Remote Scripts directory
2. Select it as a control surface in Live's preferences
3. Use the companion client script to send parameter updates
"""

import Live
import json
import socket
import threading
import time
from typing import Dict, List, Optional, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PolyMAXRemoteScript:
    """
    Ableton Live Remote Script for controlling PolyMAX VST parameters
    """
    
    def __init__(self, c_instance):
        self.c_instance = c_instance
        self.application = Live.Application.get_application()
        self.song = self.application.get_document()
        
        # Server configuration
        self.server_host = 'localhost'
        self.server_port = 9001
        self.server_socket = None
        self.server_thread = None
        self.running = False
        
        # PolyMAX plugin reference
        self.polymax_device = None
        self.polymax_track = None
        
        # Parameter mapping (VST parameter names to indices)
        self.parameter_mapping = self._create_parameter_mapping()
        
        logger.info("PolyMAX Remote Script initialized")
        self._start_server()
        self._find_polymax_device()
    
    def _create_parameter_mapping(self) -> Dict[str, int]:
        """
        Create mapping from parameter names to VST parameter indices
        This should match the parameter structure used in training
        """
        # Based on PolyMAX VST parameter structure
        # These indices may need adjustment based on actual VST implementation
        return {
            'osc1_wave': 0,
            'osc1_pitch': 1,
            'osc1_fine': 2,
            'osc1_level': 3,
            'osc2_wave': 4,
            'osc2_pitch': 5,
            'osc2_fine': 6,
            'osc2_level': 7,
            'osc_mix': 8,
            'filter_cutoff': 9,
            'filter_resonance': 10,
            'filter_type': 11,
            'filter_env_amount': 12,
            'amp_attack': 13,
            'amp_decay': 14,
            'amp_sustain': 15,
            'amp_release': 16,
            'filter_attack': 17,
            'filter_decay': 18,
            'filter_sustain': 19,
            'filter_release': 20,
            'lfo1_rate': 21,
            'lfo1_amount': 22,
            'lfo1_destination': 23,
            'lfo2_rate': 24,
            'lfo2_amount': 25,
            'lfo2_destination': 26,
            'reverb_amount': 27,
            'delay_amount': 28,
            'chorus_amount': 29,
            'master_volume': 30
        }
    
    def _start_server(self):
        """
        Start TCP server to receive parameter updates
        """
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.server_host, self.server_port))
            self.server_socket.listen(1)
            
            self.running = True
            self.server_thread = threading.Thread(target=self._server_loop)
            self.server_thread.daemon = True
            self.server_thread.start()
            
            logger.info(f"Parameter server started on {self.server_host}:{self.server_port}")
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
    
    def _server_loop(self):
        """
        Main server loop to handle incoming connections
        """
        while self.running:
            try:
                client_socket, address = self.server_socket.accept()
                logger.info(f"Client connected from {address}")
                
                # Handle client in separate thread
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket,)
                )
                client_thread.daemon = True
                client_thread.start()
                
            except Exception as e:
                if self.running:
                    logger.error(f"Server error: {e}")
                break
    
    def _handle_client(self, client_socket):
        """
        Handle individual client connections
        """
        try:
            while True:
                data = client_socket.recv(4096)
                if not data:
                    break
                
                # Parse JSON message
                try:
                    message = json.loads(data.decode('utf-8'))
                    self._process_message(message, client_socket)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON received: {e}")
                    self._send_response(client_socket, {
                        'status': 'error',
                        'message': 'Invalid JSON format'
                    })
                
        except Exception as e:
            logger.error(f"Client handling error: {e}")
        finally:
            client_socket.close()
    
    def _process_message(self, message: Dict[str, Any], client_socket):
        """
        Process incoming parameter update messages
        """
        try:
            command = message.get('command')
            
            if command == 'set_parameters':
                parameters = message.get('parameters', {})
                success = self._apply_parameters(parameters)
                
                response = {
                    'status': 'success' if success else 'error',
                    'message': 'Parameters applied' if success else 'Failed to apply parameters',
                    'applied_count': len(parameters) if success else 0
                }
                
            elif command == 'get_status':
                response = {
                    'status': 'success',
                    'polymax_found': self.polymax_device is not None,
                    'track_name': self.polymax_track.name if self.polymax_track else None,
                    'parameter_count': len(self.polymax_device.parameters) if self.polymax_device else 0
                }
                
            elif command == 'find_polymax':
                found = self._find_polymax_device()
                response = {
                    'status': 'success',
                    'found': found,
                    'track_name': self.polymax_track.name if found else None
                }
                
            else:
                response = {
                    'status': 'error',
                    'message': f'Unknown command: {command}'
                }
            
            self._send_response(client_socket, response)
            
        except Exception as e:
            logger.error(f"Message processing error: {e}")
            self._send_response(client_socket, {
                'status': 'error',
                'message': str(e)
            })
    
    def _send_response(self, client_socket, response: Dict[str, Any]):
        """
        Send JSON response to client
        """
        try:
            response_json = json.dumps(response)
            client_socket.send(response_json.encode('utf-8'))
        except Exception as e:
            logger.error(f"Failed to send response: {e}")
    
    def _find_polymax_device(self) -> bool:
        """
        Find PolyMAX VST device in the current Live set
        """
        try:
            for track in self.song.tracks:
                for device in track.devices:
                    # Check if device is PolyMAX VST
                    if hasattr(device, 'class_name') and 'polymax' in device.class_name.lower():
                        self.polymax_device = device
                        self.polymax_track = track
                        logger.info(f"Found PolyMAX on track: {track.name}")
                        return True
                    
                    # Alternative check by device name
                    if hasattr(device, 'name') and 'polymax' in device.name.lower():
                        self.polymax_device = device
                        self.polymax_track = track
                        logger.info(f"Found PolyMAX on track: {track.name}")
                        return True
            
            logger.warning("PolyMAX device not found")
            return False
            
        except Exception as e:
            logger.error(f"Error finding PolyMAX device: {e}")
            return False
    
    def _apply_parameters(self, parameters: Dict[str, float]) -> bool:
        """
        Apply predicted parameters to PolyMAX VST
        """
        if not self.polymax_device:
            logger.error("PolyMAX device not found")
            return False
        
        try:
            applied_count = 0
            
            for param_name, value in parameters.items():
                if param_name in self.parameter_mapping:
                    param_index = self.parameter_mapping[param_name]
                    
                    # Ensure parameter index is valid
                    if param_index < len(self.polymax_device.parameters):
                        # Clamp value to valid range [0.0, 1.0]
                        clamped_value = max(0.0, min(1.0, float(value)))
                        
                        # Apply parameter value
                        self.polymax_device.parameters[param_index].value = clamped_value
                        applied_count += 1
                        
                        logger.debug(f"Applied {param_name}: {clamped_value}")
                    else:
                        logger.warning(f"Parameter index {param_index} out of range for {param_name}")
                else:
                    logger.warning(f"Unknown parameter: {param_name}")
            
            logger.info(f"Applied {applied_count}/{len(parameters)} parameters")
            return applied_count > 0
            
        except Exception as e:
            logger.error(f"Error applying parameters: {e}")
            return False
    
    def disconnect(self):
        """
        Clean up resources when script is disconnected
        """
        self.running = False
        
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        logger.info("PolyMAX Remote Script disconnected")


# Required functions for Ableton Live Remote Script
def create_instance(c_instance):
    """
    Create instance of the remote script
    """
    return PolyMAXRemoteScript(c_instance)


# Client helper class for sending parameters from inference script
class PolyMAXClient:
    """
    Client class for sending parameter updates to Ableton Live
    """
    
    def __init__(self, host='localhost', port=9001):
        self.host = host
        self.port = port
    
    def send_parameters(self, parameters: Dict[str, float]) -> Dict[str, Any]:
        """
        Send predicted parameters to Ableton Live
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((self.host, self.port))
                
                message = {
                    'command': 'set_parameters',
                    'parameters': parameters
                }
                
                sock.send(json.dumps(message).encode('utf-8'))
                response = sock.recv(4096)
                
                return json.loads(response.decode('utf-8'))
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Connection failed: {e}'
            }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get status of PolyMAX remote script
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((self.host, self.port))
                
                message = {'command': 'get_status'}
                sock.send(json.dumps(message).encode('utf-8'))
                response = sock.recv(4096)
                
                return json.loads(response.decode('utf-8'))
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Connection failed: {e}'
            }


if __name__ == '__main__':
    # Example usage of client
    client = PolyMAXClient()
    
    # Test parameters
    test_params = {
        'osc1_level': 0.8,
        'filter_cutoff': 0.6,
        'filter_resonance': 0.3,
        'amp_attack': 0.1,
        'amp_release': 0.4
    }
    
    print("Testing PolyMAX parameter client...")
    status = client.get_status()
    print(f"Status: {status}")
    
    if status.get('status') == 'success':
        result = client.send_parameters(test_params)
        print(f"Parameter update result: {result}")
    else:
        print("Remote script not available")