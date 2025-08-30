#!/usr/bin/env python3
"""
Thread-Safe Serial Port Manager
===============================

Centralized serial port management system that prevents conflicts between
multiple Arduino controllers and provides robust error handling.

Features:
- Global serial port registry with mutex protection
- Connection pooling to prevent double-opens
- Thread-safe access across all controllers
- Graceful degradation when devices fail
- Resource cleanup and recovery
"""

import serial
import threading
import time
import logging
from typing import Dict, Optional, Set, Callable
import atexit

logger = logging.getLogger(__name__)

class SerialPortManager:
    """Thread-safe serial port manager with connection pooling."""
    
    def __init__(self):
        self._connections: Dict[str, serial.Serial] = {}
        self._connection_lock = threading.RLock()
        self._port_locks: Dict[str, threading.Lock] = {}
        self._registered_cleanup: Set[str] = set()
        
        # Register cleanup on exit
        atexit.register(self.cleanup_all)
    
    def get_port_lock(self, port: str) -> threading.Lock:
        """Get or create a lock for a specific port."""
        with self._connection_lock:
            if port not in self._port_locks:
                self._port_locks[port] = threading.Lock()
            return self._port_locks[port]
    
    def acquire_port(self, port: str, baudrate: int = 9600, 
                    timeout: float = 1.0, **kwargs) -> Optional[serial.Serial]:
        """
        Acquire exclusive access to a serial port.
        
        Args:
            port: Serial port path (e.g., '/dev/ttyUSB0')
            baudrate: Baud rate (default 9600)
            timeout: Read timeout
            **kwargs: Additional serial.Serial parameters
            
        Returns:
            Serial connection or None if failed
        """
        with self.get_port_lock(port):
            with self._connection_lock:
                # Return existing connection if already open
                if port in self._connections:
                    connection = self._connections[port]
                    if connection.is_open:
                        logger.debug(f"Reusing existing connection for {port}")
                        return connection
                    else:
                        # Connection closed, remove it
                        del self._connections[port]
                
                # Create new connection
                try:
                    logger.info(f"Opening new connection to {port}")
                    
                    # Default parameters for Arduino connections
                    default_kwargs = {
                        'timeout': timeout,
                        'write_timeout': 1.0,
                        'dsrdtr': False,  # Prevent DTR resets
                        'rtscts': False
                    }
                    default_kwargs.update(kwargs)
                    
                    connection = serial.Serial(port, baudrate, **default_kwargs)
                    
                    # Brief stabilization time
                    time.sleep(0.5)
                    
                    # Clear any startup data
                    if connection.in_waiting:
                        startup_data = connection.read(connection.in_waiting)
                        logger.debug(f"Cleared startup data from {port}: {startup_data}")
                    
                    self._connections[port] = connection
                    logger.info(f"Successfully connected to {port}")
                    return connection
                    
                except Exception as e:
                    logger.error(f"Failed to connect to {port}: {e}")
                    return None
    
    def release_port(self, port: str) -> bool:
        """
        Release a serial port connection.
        
        Args:
            port: Serial port path
            
        Returns:
            True if successfully released
        """
        with self.get_port_lock(port):
            with self._connection_lock:
                if port in self._connections:
                    try:
                        connection = self._connections[port]
                        if connection.is_open:
                            connection.close()
                        del self._connections[port]
                        logger.info(f"Released connection to {port}")
                        return True
                    except Exception as e:
                        logger.error(f"Error releasing {port}: {e}")
                        # Still remove from registry even if close failed
                        if port in self._connections:
                            del self._connections[port]
                        return False
                
                logger.debug(f"Port {port} was not in connection registry")
                return True
    
    def send_command(self, port: str, command: str, 
                    expect_response: bool = False, 
                    response_timeout: float = 1.0) -> Optional[str]:
        """
        Send a command to a specific port with thread safety.
        
        Args:
            port: Serial port path
            command: Command string to send
            expect_response: Whether to wait for a response
            response_timeout: Timeout for response
            
        Returns:
            Response string if expect_response=True, otherwise None
        """
        with self.get_port_lock(port):
            if port not in self._connections:
                logger.error(f"Port {port} not connected")
                return None
            
            connection = self._connections[port]
            
            try:
                # Send command
                full_command = command.strip() + '\n'
                connection.write(full_command.encode('utf-8'))
                connection.flush()
                
                logger.debug(f"Sent to {port}: {command}")
                
                if expect_response:
                    # Wait for response
                    start_time = time.time()
                    response = ""
                    
                    while time.time() - start_time < response_timeout:
                        if connection.in_waiting:
                            response += connection.read(connection.in_waiting).decode('utf-8', errors='ignore')
                            if '\n' in response:
                                break
                        time.sleep(0.01)
                    
                    logger.debug(f"Response from {port}: {response.strip()}")
                    return response.strip()
                
                return None
                
            except Exception as e:
                logger.error(f"Error sending command to {port}: {e}")
                return None
    
    def is_connected(self, port: str) -> bool:
        """Check if a port is connected and operational."""
        with self._connection_lock:
            if port not in self._connections:
                return False
            
            connection = self._connections[port]
            return connection.is_open
    
    def get_connected_ports(self) -> Set[str]:
        """Get set of currently connected ports."""
        with self._connection_lock:
            return {port for port, conn in self._connections.items() if conn.is_open}
    
    def health_check(self, port: str) -> bool:
        """Perform a health check on a connection."""
        with self.get_port_lock(port):
            if not self.is_connected(port):
                return False
            
            try:
                # Send a gentle probe
                response = self.send_command(port, "", expect_response=True, response_timeout=0.5)
                return True  # If no exception, connection is healthy
                
            except Exception as e:
                logger.warning(f"Health check failed for {port}: {e}")
                # Try to recover
                self.release_port(port)
                return False
    
    def cleanup_all(self):
        """Cleanup all connections."""
        logger.info("Cleaning up all serial connections...")
        
        with self._connection_lock:
            ports_to_cleanup = list(self._connections.keys())
            
            for port in ports_to_cleanup:
                try:
                    self.release_port(port)
                except Exception as e:
                    logger.error(f"Error cleaning up {port}: {e}")
            
            self._connections.clear()
            self._port_locks.clear()
        
        logger.info("Serial port cleanup complete")


# Global instance
_global_serial_manager: Optional[SerialPortManager] = None

def get_serial_manager() -> SerialPortManager:
    """Get or create the global serial port manager."""
    global _global_serial_manager
    if _global_serial_manager is None:
        _global_serial_manager = SerialPortManager()
    return _global_serial_manager

def acquire_port(port: str, **kwargs) -> Optional[serial.Serial]:
    """Global function to acquire a port."""
    return get_serial_manager().acquire_port(port, **kwargs)

def release_port(port: str) -> bool:
    """Global function to release a port."""
    return get_serial_manager().release_port(port)

def send_command(port: str, command: str, **kwargs) -> Optional[str]:
    """Global function to send a command."""
    return get_serial_manager().send_command(port, command, **kwargs)

def is_connected(port: str) -> bool:
    """Global function to check connection status."""
    return get_serial_manager().is_connected(port)

def cleanup_all():
    """Global function to cleanup all connections."""
    return get_serial_manager().cleanup_all()


if __name__ == "__main__":
    # Test the serial port manager
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python serial_port_manager.py <port>")
        sys.exit(1)
    
    port = sys.argv[1]
    manager = get_serial_manager()
    
    print(f"Testing serial port manager with {port}")
    
    # Test connection
    connection = manager.acquire_port(port)
    if connection:
        print(f"Successfully connected to {port}")
        
        # Test command
        response = manager.send_command(port, "PING", expect_response=True)
        print(f"Response: {response}")
        
        # Test health check
        health = manager.health_check(port)
        print(f"Health check: {'OK' if health else 'FAILED'}")
        
        # Release
        manager.release_port(port)
        print("Connection released")
    else:
        print(f"Failed to connect to {port}")