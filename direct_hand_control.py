#!/usr/bin/env python3
"""
Direct Hand Controller Integration for machine.py
================================================

Provides simple direct integration between machine.py and the hand controller
without the complexity of bridge systems and JSON file communication.

This module allows direct function calls and shared memory between the
main LOVE_YOURSELF system and the hand controller.
"""
import os
import sys
import threading
import time
from typing import Optional, Dict, Any

# Import config for debug settings
try:
    from config import config
except ImportError:
    # Fallback if config not available
    class Config:
        DEBUG_HAND_CONTROLLER = False
    config = Config()

# Add hand_control to path for imports
hand_control_path = os.path.join(os.path.dirname(__file__), 'hand_control')
if hand_control_path not in sys.path:
    sys.path.insert(0, hand_control_path)

try:
    from hand_control_interface import HandControlInterface
    HAND_CONTROL_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Hand control not available: {e}")
    HAND_CONTROL_AVAILABLE = False
    HandControlInterface = None

def debug_print(message: str):
    """Print debug message only if DEBUG_HAND_CONTROLLER is enabled."""
    if getattr(config, 'DEBUG_HAND_CONTROLLER', False):
        print(message)

class DirectHandController:
    """Direct integration with hand controller - properly handles GUI threading."""
    
    def __init__(self, headless: bool = False):
        self.headless = headless
        self.controller = None
        self.running = False
        
        # Thread-safe emotion tracking
        self.current_emotion = None  # Track current emotion to avoid unnecessary changes
        self.controller_ready = threading.Event()  # Signal when controller is ready
        
        # Emotional state mapping
        self.emotion_mapping = {
            "energized_engaged": "energized_engaged",
            "alert_curious": "alert_curious", 
            "calm_observant": "calm_observant",
            "quiet_detached": "quiet_detached",
            "withdrawn_distant": "withdrawn_distant"
        }
        
        if HAND_CONTROL_AVAILABLE:
            self._initialize_controller()
    
    def _initialize_controller(self):
        """Initialize the hand controller interface."""
        try:
            if self.headless:
                debug_print("🔇 Initializing hand controller in headless mode...")
                return self._initialize_headless_controller()
            else:
                # For GUI mode, we need to be in the main thread
                debug_print("🎮 Preparing hand controller for GUI mode...")
                # Don't create the controller here - it will be created in start()
                return True
            
        except Exception as e:
            print(f"❌ Failed to initialize hand controller: {e}")  # Always show errors
            return False
    
    def _initialize_headless_controller(self):
        """Initialize a minimal headless controller for servo control only."""
        try:
            # Import just the servo control components
            from hand_expression import HandExpressionController
            
            # Create minimal controller structure
            class HeadlessHandController:
                def __init__(self):
                    self.hand_expression = HandExpressionController()
                    self.current_emotion = "calm_observant"
                    self.dataset_directory = "C:/Users/tobia/Downloads/HandControlStandalone/movement_recordings"
                
                def change_to_emotion(self, emotion):
                    self.current_emotion = emotion
                    print(f"🎭 Headless emotion change: {emotion}")
                
                def update_reactivity(self, activity, sudden, motion):
                    # Minimal reactivity handling
                    pass
            
            self.controller = HeadlessHandController()
            print("✅ Headless hand controller initialized")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize headless controller: {e}")
            # Fallback: set controller to None and continue without it
            self.controller = None
            return False
    
    def start(self):
        """Start the hand controller in appropriate mode."""
        if not HAND_CONTROL_AVAILABLE:
            print("⚠️ Hand controller not available")
            return False
        
        if self.controller:
            print("✅ Hand controller already started")
            return True
        
        self.running = True
        
        try:
            if self.headless:
                # Headless mode 
                if not self.controller:
                    self._initialize_headless_controller()
            else:
                # GUI mode - run in separate thread with own event loop
                debug_print("🎮 Starting hand controller in separate thread...")
                self.gui_thread = threading.Thread(target=self._run_gui_thread, daemon=True)
                self.gui_thread.start()
                
                # Wait for controller to be ready
                debug_print("⏳ Waiting for hand controller to initialize...")
                if self.controller_ready.wait(timeout=10):  # Wait up to 10 seconds
                    debug_print("✅ Hand controller GUI thread started and ready")
                else:
                    print("❌ Hand controller failed to initialize within timeout")
                    return False
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to start hand controller: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _run_gui_thread(self):
        """Run the hand controller GUI in its own thread with own event loop."""
        try:
            # Optionally suppress print statements from hand controller
            original_print = print
            
            def conditional_print(*args, **kwargs):
                # Check if this print should be suppressed
                if args and isinstance(args[0], str):
                    message = args[0]
                    # Suppress common hand controller debug patterns if DEBUG_HAND_CONTROLLER is False
                    if not getattr(config, 'DEBUG_HAND_CONTROLLER', False):
                        # Suppress these patterns
                        if any(pattern in message for pattern in [
                            '🎯', '🔄', '📤', '🎲', '🔗', '✅ Loaded', '🔒', '⏳', '🎨', '🏋️', '🛑', '📁',
                            '✅ Markov chain available', '⏱️ Initial timing', '🌅 Observing environment',
                            '🌊 Starting smooth transition', '✅ Transition complete', '✅ New dataset loaded',
                            '🏁 Dataset transition', '🤖 Auto-starting Markov generation', '🎭 Emotion changed',
                            '🔧 Configuring', '🎪 Loading', '⚡ Processing', '🔍 Analyzing', '💡 Generated',
                            '🎯 Target', '🔄 Cycling', '📍 Position', '⭐ Status', '🎲 Random',
                            '🌊 Gentle diversity drift', '🌊 Applying gentle diversity easing'
                        ]):
                            return
                    
                    # Suppress emotion change details if DEBUG_EMOTION_CHANGES is False
                    if not getattr(config, 'DEBUG_EMOTION_CHANGES', False):
                        if any(pattern in message for pattern in ['🎭 Switched to', '📊 Parameters:', '🤖 Auto-starting']):
                            return
                
                # Call original print for non-suppressed messages
                original_print(*args, **kwargs)
            
            # Temporarily replace print during hand controller execution
            import builtins
            builtins.print = conditional_print
            
            try:
                # Create the controller in this thread
                self.controller = HandControlInterface()
                
                # Signal that controller is ready
                self.controller_ready.set()
                
                # Start its main loop in this thread
                self.controller.root.mainloop()
            finally:
                # Restore original print
                builtins.print = original_print
            
        except Exception as e:
            print(f"❌ GUI thread error: {e}")
            import traceback
            traceback.print_exc()
    
    def update(self):
        """Minimal update - no longer needed since hand controller runs autonomously."""
        return True
    
    def stop(self):
        """Stop the hand controller."""
        self.running = False
        
        if self.controller:
            try:
                if not self.headless and hasattr(self.controller, 'root'):
                    self.controller.root.quit()
                    self.controller.root.destroy()
                print("🛑 Hand controller stopped")
            except Exception as e:
                print(f"⚠️ Error stopping hand controller: {e}")
            finally:
                self.controller = None
    
    def set_emotion(self, emotion: str, **kwargs):
        """Set the hand controller emotion directly."""
        if not self.controller:
            return False
            
        mapped_emotion = self.emotion_mapping.get(emotion, emotion)
        
        try:
            # Call the emotion change method directly
            if hasattr(self.controller, 'change_to_emotion'):
                self.controller.change_to_emotion(mapped_emotion)
                print(f"🎭 Hand controller emotion changed to: {mapped_emotion}")
                return True
            else:
                print(f"⚠️ Hand controller doesn't support emotion changing")
                return False
                
        except Exception as e:
            print(f"❌ Failed to set emotion {emotion}: {e}")
            return False
    
    def send_reactivity_data(self, reactivity_data: Dict[str, Any]):
        """Send camera reactivity data directly to hand controller."""
        if not self.controller:
            return False
            
        try:
            # Check if this is an action-based command (new pause system)
            action = reactivity_data.get('action')
            
            if action == 'pause':
                # Smooth pause Markov generation for specified duration
                duration = reactivity_data.get('duration', 4.0)
                activity_level = reactivity_data.get('activity_level', 0.0)
                
                if hasattr(self.controller, 'pause_markov_generation'):
                    try:
                        self.controller.pause_markov_generation()
                    except Exception as pause_error:
                        print(f"❌ Pause method error: {pause_error}")
                        import traceback
                        traceback.print_exc()
                        return False
                elif hasattr(self.controller, 'stop_markov_generation'):
                    try:
                        self.controller.stop_markov_generation()
                    except Exception as stop_error:
                        print(f"❌ Stop method error: {stop_error}")
                        import traceback
                        traceback.print_exc()
                        return False
                return True
                
            elif action == 'resume':
                # Smooth resume Markov generation
                activity_level = reactivity_data.get('activity_level', 0.0)
                if hasattr(self.controller, 'resume_markov_generation'):
                    self.controller.resume_markov_generation()
                elif hasattr(self.controller, 'start_markov_generation'):
                    self.controller.start_markov_generation()
                    if getattr(config, 'DEBUG_REACTIVITY_PAUSE', False):
                        print("✅ Hand controller resumed (fallback)")
                return True
            
            # Unknown action type
            if getattr(config, 'DEBUG_REACTIVITY_PAUSE', False):
                print(f"⚠️ Unknown reactivity action: {action}")
            return False
            
        except Exception as e:
            print(f"❌ Failed to send reactivity data: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current hand controller status."""
        if not self.controller:
            return {"available": False, "running": False}
            
        try:
            status = {
                "available": True,
                "running": self.running,
                "current_emotion": getattr(self.controller, 'current_emotion', 'unknown'),
                "dataset_directory": getattr(self.controller, 'dataset_directory', 'unknown')
            }
            
            return status
            
        except Exception as e:
            print(f"❌ Failed to get status: {e}")
            return {"available": False, "error": str(e)}
    
    def change_to_emotion(self, emotion):
        """Change the emotional state of the hand controller - thread-safe and immediate."""
        if not self.controller:
            print("⚠️ Hand controller not available")
            return False
        
        # Check if emotion is already current - avoid unnecessary resets
        if self.current_emotion == emotion:
            debug_print(f"🎭 Hand controller already in {emotion} state - skipping change")
            return True
        
        try:
            if hasattr(self.controller, 'switch_emotional_state'):
                if getattr(config, 'DEBUG_EMOTION_CHANGES', False):
                    debug_print(f"🎭 Switching hand controller emotion to: {emotion}")
                # Note: Clean emotion indicator is handled by hand_control_interface.py
                
                # Call directly - this is thread-safe in Tkinter for simple operations
                self.controller.switch_emotional_state(emotion)
                self.current_emotion = emotion
                return True
            else:
                print("⚠️ Hand controller doesn't support emotion changing")
                return False
        except Exception as e:
            print(f"❌ Error changing emotion to {emotion}: {e}")
            return False
    
    def start_autonomous_mode(self):
        """Start autonomous Markov chain generation."""
        if not self.controller:
            print("⚠️ Hand controller not available")
            return False
        
        try:
            if hasattr(self.controller, 'start_markov_generation'):
                print("🧠 Starting autonomous Markov generation...")
                self.controller.start_markov_generation()
                return True
            else:
                print("⚠️ Hand controller doesn't support autonomous mode")
                return False
        except Exception as e:
            print(f"❌ Error starting autonomous mode: {e}")
            return False

# Global instance for easy access
_hand_controller_instance: Optional[DirectHandController] = None

def get_hand_controller(headless: bool = False) -> Optional[DirectHandController]:
    """Get or create the global hand controller instance."""
    global _hand_controller_instance
    
    if _hand_controller_instance is None:
        _hand_controller_instance = DirectHandController(headless=headless)
        
    return _hand_controller_instance

def start_hand_controller(headless: bool = False) -> bool:
    """Start the hand controller system."""
    controller = get_hand_controller(headless)
    if controller:
        return controller.start()
    return False

def stop_hand_controller():
    """Stop the hand controller system."""
    global _hand_controller_instance
    if _hand_controller_instance:
        _hand_controller_instance.stop()
        _hand_controller_instance = None

def update_hand_controller() -> bool:
    """Update hand controller GUI - call from main thread."""
    controller = get_hand_controller()
    return controller.update() if controller else False

# Simple direct API functions for machine.py
def set_emotion(emotion: str, **kwargs) -> bool:
    """Set hand controller emotion - direct API."""
    controller = get_hand_controller()
    return controller.set_emotion(emotion, **kwargs) if controller else False

def send_reactivity_data(reactivity_data: Dict[str, Any]) -> bool:
    """Send reactivity data - direct API."""
    controller = get_hand_controller()
    return controller.send_reactivity_data(reactivity_data) if controller else False

def get_status() -> Dict[str, Any]:
    """Get status - direct API."""
    controller = get_hand_controller()
    return controller.get_status() if controller else {"available": False}

def change_to_emotion(emotion: str) -> bool:
    """Change hand controller emotion - direct API."""
    controller = get_hand_controller()
    return controller.change_to_emotion(emotion) if controller else False

def start_autonomous_mode() -> bool:
    """Start autonomous Markov generation - direct API."""
    controller = get_hand_controller()
    return controller.start_autonomous_mode() if controller else False
