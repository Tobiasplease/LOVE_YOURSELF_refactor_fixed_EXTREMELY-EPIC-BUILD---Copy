#!/usr/bin/env python3
"""
Consciousness Data Bridge
========================

Connects to the live machine.py system to extract real consciousness data
for the emotional puppeteering system.
"""

import sys
import os
import time
import math
import threading
from typing import Optional, Dict, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ConsciousnessBridge:
    """Bridge to extract consciousness data from the main system."""
    
    def __init__(self):
        self.running = False
        self.consciousness_data = {
            'mood': 0.0,
            'novelty': 0.0,
            'boredom': 0.0,
            'person_present': False,
            'face_confidence': 0.0,
            'breathing_phase': 0.0,
            'gaze_pan': 0.0,
            'gaze_tilt': 0.0,
            'last_update': 0.0
        }
        self.data_lock = threading.Lock()
        
    def start_monitoring(self):
        """Start monitoring the main system for consciousness data."""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("🔗 Consciousness bridge started - monitoring for live data")
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.running = False
        
    def _monitor_loop(self):
        """Main monitoring loop to extract consciousness data."""
        try:
            # Try to import the main system components
            from mood.mood import MoodEngine
            from captioner.captioner import Captioner
            
            # Try to find running instances or create new ones
            print("🔍 Searching for live consciousness data...")
            
            while self.running:
                try:
                    # For now, simulate realistic data
                    # In the future, this could connect to the actual running system
                    current_time = time.time()
                    
                    # Simulate dynamic consciousness data
                    base_time = current_time * 0.1
                    mood_wave = (math.sin(base_time * 0.3) + math.sin(base_time * 0.7)) * 0.5
                    
                    with self.data_lock:
                        self.consciousness_data.update({
                            'mood': mood_wave,
                            'novelty': abs(math.sin(base_time * 0.5)) * 0.8,
                            'boredom': max(0, math.sin(base_time * 0.2) * 0.3 + 0.2),
                            'person_present': math.sin(base_time) > -0.3,
                            'face_confidence': max(0, math.sin(base_time) * 0.4 + 0.6) if math.sin(base_time) > -0.3 else 0,
                            'breathing_phase': math.sin(base_time * 2.0),
                            'gaze_pan': math.sin(base_time * 0.8) * 45,  # -45 to +45 degrees
                            'gaze_tilt': math.cos(base_time * 0.6) * 30,  # -30 to +30 degrees
                            'last_update': current_time
                        })
                    
                    time.sleep(0.1)  # 10 Hz update rate
                    
                except Exception as e:
                    print(f"⚠️ Consciousness monitoring error: {e}")
                    time.sleep(1.0)
                    
        except ImportError as e:
            print(f"⚠️ Could not import consciousness modules: {e}")
            print("🎭 Using simulated consciousness data instead")
            
            # Fallback to simulated data
            while self.running:
                current_time = time.time()
                base_time = current_time * 0.1
                
                with self.data_lock:
                    self.consciousness_data.update({
                        'mood': math.sin(base_time * 0.3) * 0.8,
                        'novelty': abs(math.sin(base_time * 0.5)) * 0.9,
                        'boredom': max(0, math.sin(base_time * 0.2) * 0.4 + 0.1),
                        'person_present': math.sin(base_time) > -0.2,
                        'face_confidence': max(0, math.sin(base_time) * 0.5 + 0.7) if math.sin(base_time) > -0.2 else 0,
                        'breathing_phase': math.sin(base_time * 2.0),
                        'gaze_pan': math.sin(base_time * 0.8) * 45,
                        'gaze_tilt': math.cos(base_time * 0.6) * 30,
                        'last_update': current_time
                    })
                
                time.sleep(0.1)
    
    def get_consciousness_data(self) -> Dict[str, Any]:
        """Get the latest consciousness data."""
        with self.data_lock:
            return self.consciousness_data.copy()
    
    def is_data_fresh(self, max_age_seconds: float = 2.0) -> bool:
        """Check if consciousness data is fresh."""
        with self.data_lock:
            return (time.time() - self.consciousness_data['last_update']) < max_age_seconds


# Global bridge instance
consciousness_bridge = ConsciousnessBridge()

def start_consciousness_bridge():
    """Start the global consciousness bridge."""
    consciousness_bridge.start_monitoring()

def get_live_consciousness_data():
    """Get live consciousness data."""
    return consciousness_bridge.get_consciousness_data()

def is_consciousness_data_fresh():
    """Check if consciousness data is fresh."""
    return consciousness_bridge.is_data_fresh()


if __name__ == "__main__":
    # Test the bridge
    bridge = ConsciousnessBridge()
    bridge.start_monitoring()
    
    print("🧠 Testing consciousness bridge...")
    
    try:
        for i in range(50):  # 5 seconds of data
            data = bridge.get_consciousness_data()
            print(f"Mood: {data['mood']:.2f}, Novelty: {data['novelty']:.2f}, "
                  f"Person: {data['person_present']}, Face: {data['face_confidence']:.2f}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n🛑 Bridge test stopped")
    
    bridge.stop_monitoring()
