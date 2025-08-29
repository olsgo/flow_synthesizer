#!/usr/bin/env python3
"""
Simple TCP server for testing Ableton Live integration.
This version runs without requiring user input.
"""

import socket
import json
import threading
import time
import signal
import sys

class SimplePolyMAXServer:
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
            
            print(f"🚀 PolyMAX Server started on {self.host}:{self.port}")
            print("Waiting for Ableton Live to connect...")
            print("Server is running. Press Ctrl+C to stop.\n")
            
            while self.running:
                try:
                    self.server_socket.settimeout(1.0)  # Allow periodic checks
                    client_socket, client_address = self.server_socket.accept()
                    print(f"✅ New connection from {client_address}")
                    
                    # Handle client in a separate thread
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, client_address)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                    
                except socket.timeout:
                    continue  # Check if we should keep running
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
                try:
                    client_socket.settimeout(1.0)
                    data = client_socket.recv(1024)
                    if not data:
                        break
                    
                    # Parse JSON message
                    message = json.loads(data.decode('utf-8'))
                    print(f"📨 Received: {message}")
                    
                    # Handle different message types
                    response = self.process_message(message)
                    
                    # Send response
                    if response:
                        response_json = json.dumps(response)
                        client_socket.send(response_json.encode('utf-8'))
                        print(f"📤 Sent: {response}")
                        
                except socket.timeout:
                    continue
                except json.JSONDecodeError:
                    print(f"⚠️  Invalid JSON received")
                except Exception as e:
                    print(f"❌ Error handling message: {e}")
                    break
                    
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
                'timestamp': time.time()
            }
            
        elif action == 'get_status':
            return {
                'status': 'ready',
                'connected_clients': len(self.clients),
                'timestamp': time.time()
            }
            
        elif action == 'predict_parameters':
            print(f"🎵 Simulating parameter prediction...")
            
            # Generate sample parameters
            sample_params = {
                'filter_cutoff': 0.7,
                'filter_resonance': 0.3,
                'env_attack': 0.1,
                'env_decay': 0.4,
                'env_sustain': 0.7,
                'env_release': 0.6
            }
            
            return {
                'status': 'success',
                'parameters': sample_params,
                'confidence': 0.85,
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
            return {
                'status': 'error',
                'message': f'Unknown action: {action}',
                'timestamp': time.time()
            }
    
    def stop_server(self):
        """Stop the TCP server."""
        print("\n🛑 Stopping server...")
        self.running = False
        
        # Close all client connections
        for client in self.clients:
            try:
                client.close()
            except:
                pass
        self.clients.clear()
        
        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        print("✅ Server stopped")

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n🛑 Received interrupt signal")
    sys.exit(0)

def main():
    """Main function to run the server."""
    print("Simple PolyMAX TCP Server")
    print("=========================\n")
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    server = SimplePolyMAXServer()
    
    try:
        server.start_server()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop_server()

if __name__ == "__main__":
    main()