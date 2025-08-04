#!/usr/bin/env python3
"""
🔗 Hand Control Integration Bridge
=================================

Simple bridge to connect the main LOVE_YOURSELF AI system
with the isolated hand control system via UDP.

Usage in machine.py:
    from hand_control_bridge import HandControlBridge
    
    # Initialize bridge
    hand_bridge = HandControlBridge()
    
    # Send mood updates
    hand_bridge.send_mood_update(emotional_state)
    
    # Or use auto-notification
    hand_bridge.enable_auto_notification(mood_analyzer)

Features:
- Lightweight UDP communication
- Non-blocking operation (won't slow down main AI)
- Graceful failure (won't crash if hand control not running)
- Simple integration (just 2-3 lines of code)
"""

import socket
import json
import time
import threading
from typing import Optional, Callable

class HandControlBridge:
    """Simple bridge for sending mood updates to isolated hand control system."""
    
    def __init__(self, host='localhost', port=12345):
        """Initialize the bridge."""
        self.host = host
        self.port = port
        self.enabled = True
        self.last_emotion = None
        self.send_count = 0
        
        print(f"🔗 Hand Control Bridge initialized")
        print(f"📡 Target: {host}:{port}")
        print(f"💡 Use send_mood_update(emotion) to notify hand control system")
    
    def send_mood_update(self, emotional_state: str, extra_data: dict = None) -> bool:
        """
        Send mood update to hand control system.
        
        Args:
            emotional_state: The current emotional state (e.g., 'calm_observant')
            extra_data: Optional additional data to send
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            # Prepare mood data
            mood_data = {
                'emotional_state': emotional_state,
                'timestamp': time.time(),
                'source': 'LOVE_YOURSELF_AI'
            }
            
            # Add extra data if provided
            if extra_data:
                mood_data.update(extra_data)
            
            # Send via UDP
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(0.1)  # Very short timeout
            message = json.dumps(mood_data).encode()
            sock.sendto(message, (self.host, self.port))
            sock.close()
            
            # Track successful sends
            self.send_count += 1
            self.last_emotion = emotional_state
            
            print(f"📡 Sent mood update to hand control: {emotional_state}")
            return True
            
        except Exception as e:
            # Fail silently to avoid disrupting main AI system
            if self.send_count == 0:  # Only log first failure
                print(f"📡 Hand control not available (this is normal): {e}")
            return False
    
    def send_mood_update_if_changed(self, emotional_state: str, extra_data: dict = None) -> bool:
        """
        Send mood update only if the emotional state has changed.
        
        Args:
            emotional_state: The current emotional state
            extra_data: Optional additional data to send
            
        Returns:
            bool: True if sent (because state changed), False if no change or send failed
        """
        if emotional_state != self.last_emotion:
            return self.send_mood_update(emotional_state, extra_data)
        return False
    
    def enable_auto_notification(self, mood_analyzer, check_interval: float = 1.0):
        """
        Enable automatic mood notifications from a mood analyzer object.
        
        Args:
            mood_analyzer: Object with get_current_emotion() method
            check_interval: How often to check for mood changes (seconds)
        """
        def auto_notify_loop():
            last_check_emotion = None
            
            while self.enabled:
                try:
                    if hasattr(mood_analyzer, 'get_current_emotion'):
                        current_emotion = mood_analyzer.get_current_emotion()
                        if current_emotion and current_emotion != last_check_emotion:
                            self.send_mood_update(current_emotion)
                            last_check_emotion = current_emotion
                    
                    time.sleep(check_interval)
                    
                except Exception as e:
                    print(f"📡 Auto-notification error: {e}")
                    time.sleep(check_interval)
        
        # Start background thread
        notify_thread = threading.Thread(target=auto_notify_loop, daemon=True)
        notify_thread.start()
        
        print(f"🤖 Auto-notification enabled (checking every {check_interval}s)")
    
    def disable(self):
        """Disable the bridge (stops sending updates)."""
        self.enabled = False
        print("📡 Hand Control Bridge disabled")
    
    def enable(self):
        """Re-enable the bridge."""
        self.enabled = True
        print("📡 Hand Control Bridge enabled")
    
    def get_status(self) -> dict:
        """Get bridge status information."""
        return {
            'enabled': self.enabled,
            'host': self.host,
            'port': self.port,
            'last_emotion': self.last_emotion,
            'send_count': self.send_count
        }

# === INTEGRATION EXAMPLES ===

def integrate_with_machine_py_simple():
    """
    Simple integration example for machine.py
    """
    # Initialize bridge
    hand_bridge = HandControlBridge()
    
    # Example of sending mood updates
    def on_emotion_change(new_emotion):
        """Call this whenever the AI detects an emotion change."""
        hand_bridge.send_mood_update_if_changed(new_emotion)
    
    return hand_bridge, on_emotion_change

def integrate_with_machine_py_advanced():
    """
    Advanced integration example with automatic detection
    """
    # Initialize bridge
    hand_bridge = HandControlBridge()
    
    # Example auto-notification setup
    # hand_bridge.enable_auto_notification(your_mood_analyzer_object)
    
    return hand_bridge

# === STANDALONE TESTING ===

def test_bridge():
    """Test the bridge functionality."""
    print("🧪 Testing Hand Control Bridge...")
    
    bridge = HandControlBridge()
    
    # Test emotions
    test_emotions = [
        'calm_observant',
        'alert_curious', 
        'energized_engaged',
        'quiet_detached',
        'withdrawn_distant'
    ]
    
    for emotion in test_emotions:
        print(f"🧪 Testing emotion: {emotion}")
        success = bridge.send_mood_update(emotion)
        print(f"   Result: {'✅ Sent' if success else '❌ Failed'}")
        time.sleep(0.5)
    
    print(f"🧪 Test complete. Status: {bridge.get_status()}")

if __name__ == "__main__":
    # Run standalone test
    test_bridge()
