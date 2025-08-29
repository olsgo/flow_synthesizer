#!/usr/bin/env python3
"""
Standalone TCP server for testing Ableton Live integration.
This server simulates the parameter prediction service and communicates with the Ableton remote script.
"""

import socket
import json
import threading
import time
import sys
from datetime import datetime

class PolyMAXTestServer:
    def __init__(self, host='localhost', port=9001):
        self.host = host
        self.port = port
        self.server_socket = None
        self.running = False
        self.clients = []
        
    def start_server(self):
        """Start the TCP server."""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.running = True
            
            print(f"🚀 PolyMAX Test Server started on {self.host}:{self.port}")
            print("Waiting for Ableton Live to connect...")
            print("Press Ctrl+C to stop the server\n")
            
            while self.running:
                try:
                    client_socket, client_address = self.server_socket.accept()
                    print(f"✅ New connection from {client_address}")
                    
                    # Handle client in a separate thread
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, client_address)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                    
                except socket.error as e:
                    if self.running:
                        print(f"❌ Socket error: {e}")
                    break
                    
        except Exception as e:
            print(f"❌ Failed to start server: {e}")
        finally:
            self.stop_server()
    
    def handle_client(self, client_socket, client_address):
        """Handle communication with a connected client."""
        self.clients.append(client_socket)
        
        try:
            while self.running:
                # Receive data from client
                data = client_socket.recv(1024)
                if not data:
                    break
                
                try:
                    # Parse JSON message
                    message = json.loads(data.decode('utf-8'))
                    print(f"📨 Received from {client_address}: {message}")
                    
                    # Handle different message types
                    response = self.process_message(message)
                    
                    # Send response
                    if response:
                        response_json = json.dumps(response)
                        client_socket.send(response_json.encode('utf-8'))
                        print(f"📤 Sent response: {response}")
                        
                except json.JSONDecodeError:
                    print(f"⚠️  Invalid JSON received from {client_address}")
                except Exception as e:
                    print(f"❌ Error handling message: {e}")
                    
        except Exception as e:
            print(f"❌ Client error: {e}")
        finally:
            print(f"🔌 Client {client_address} disconnected")
            if client_socket in self.clients:
                self.clients.remove(client_socket)
            client_socket.close()
    
    def process_message(self, message):
        """Process incoming messages and generate appropriate responses."""
        action = message.get('action')
        
        if action == 'ping':
            return {
                'status': 'pong',
                'timestamp': time.time(),
                'server_info': 'PolyMAX Test Server v1.0'
            }
            
        elif action == 'get_status':
            return {
                'status': 'ready',
                'model_loaded': False,  # No actual model in test server
                'connected_clients': len(self.clients),
                'timestamp': time.time()
            }
            
        elif action == 'predict_parameters':
            # Simulate parameter prediction with random values
            audio_file = message.get('audio_file', 'unknown')
            print(f"🎵 Simulating parameter prediction for: {audio_file}")
            
            # Generate sample parameters
            sample_params = {
                'osc1_wave': 0.6,
                'osc1_pitch': 0.2,
                'osc1_level': 0.8,
                'osc2_wave': 0.4,
                'osc2_pitch': -0.1,
                'osc2_level': 0.6,
                'filter_cutoff': 0.7,
                'filter_resonance': 0.3,
                'filter_env_amount': 0.5,
                'env_attack': 0.1,
                'env_decay': 0.4,
                'env_sustain': 0.7,
                'env_release': 0.6,
                'lfo_rate': 0.3,
                'lfo_amount': 0.2,
                'reverb_level': 0.4
            }
            
            return {
                'status': 'success',
                'parameters': sample_params,
                'confidence': 0.85,
                'processing_time': 0.123,
                'timestamp': time.time()
            }
            
        elif action == 'set_parameters':
            parameters = message.get('parameters', {})
            print(f"🎛️  Received parameter update: {parameters}")
            
            return {
                'status': 'parameters_applied',
                'applied_count': len(parameters),
                'timestamp': time.time()
            }
            
        else:
            print(f"⚠️  Unknown action: {action}")
            return {
                'status': 'error',
                'message': f'Unknown action: {action}',
                'timestamp': time.time()
            }
    
    def broadcast_message(self, message):
        """Send a message to all connected clients."""
        if not self.clients:
            print("📡 No clients connected to broadcast to")
            return
            
        message_json = json.dumps(message)
        disconnected_clients = []
        
        for client in self.clients:
            try:
                client.send(message_json.encode('utf-8'))
            except:
                disconnected_clients.append(client)
        
        # Remove disconnected clients
        for client in disconnected_clients:
            self.clients.remove(client)
            client.close()
    
    def stop_server(self):
        """Stop the TCP server."""
        print("\n🛑 Stopping server...")
        self.running = False
        
        # Close all client connections
        for client in self.clients:
            client.close()
        self.clients.clear()
        
        # Close server socket
        if self.server_socket:
            self.server_socket.close()
        
        print("✅ Server stopped")

def main():
    """Main function to run the test server."""
    print("PolyMAX TCP Test Server")
    print("=======================")
    print("This server simulates the parameter prediction service")
    print("and can communicate with the Ableton Live remote script.\n")
    
    server = PolyMAXTestServer()
    
    try:
        # Start the server in a separate thread so we can handle keyboard input
        server_thread = threading.Thread(target=server.start_server)
        server_thread.daemon = True
        server_thread.start()
        
        # Keep the main thread alive and handle user input
        while server.running:
            try:
                user_input = input("Enter command (help, broadcast, quit): ").strip().lower()
                
                if user_input == 'quit' or user_input == 'q':
                    break
                elif user_input == 'help' or user_input == 'h':
                    print("\nAvailable commands:")
                    print("  help (h)      - Show this help")
                    print("  broadcast (b) - Send test parameters to all clients")
                    print("  quit (q)      - Stop the server")
                    print("  status (s)    - Show server status\n")
                elif user_input == 'broadcast' or user_input == 'b':
                    test_message = {
                        'action': 'set_parameters',
                        'parameters': {
                            'filter_cutoff': 0.8,
                            'filter_resonance': 0.4,
                            'env_attack': 0.05,
                            'env_release': 0.7
                        },
                        'source': 'test_server',
                        'timestamp': time.time()
                    }
                    server.broadcast_message(test_message)
                    print("📡 Broadcast test parameters to all clients")
                elif user_input == 'status' or user_input == 's':
                    print(f"\n📊 Server Status:")
                    print(f"   Running: {server.running}")
                    print(f"   Connected clients: {len(server.clients)}")
                    print(f"   Server address: {server.host}:{server.port}\n")
                elif user_input:
                    print(f"Unknown command: {user_input}. Type 'help' for available commands.")
                    
            except EOFError:
                break
            except KeyboardInterrupt:
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        server.stop_server()

if __name__ == "__main__":
    main()