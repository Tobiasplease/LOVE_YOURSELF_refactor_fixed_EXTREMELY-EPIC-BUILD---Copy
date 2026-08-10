"""
Drawing State Manager
Global state tracking for drawing operations to inform captioning system
"""

import threading
import time
from typing import Optional, Dict, Any

class DrawingState:
    """Global drawing state manager"""
    
    _lock = threading.Lock()
    _is_drawing = False
    # A latched flag disables drawing for the whole session (Aug 10): a
    # TimeoutError on ONE g-code line ("M3 S38 ; PEN UP") raised past
    # end_drawing(), and every drawing check for the next half hour reported
    # "GRBL execution currently in progress" with the machine sitting idle.
    # The raise paths are fixed, but this is the backstop that makes ANY future
    # leak self-healing: no drawing runs for half an hour (observed: ~4 min).
    # Safe against a genuinely long job, since state_manager.is_executing_cnc —
    # which the caller clears reliably — independently gates the same decision.
    _MAX_DRAWING_SECONDS = 1800
    _drawing_start_time: Optional[float] = None
    _drawing_intent: Optional[str] = None
    _drawing_file: Optional[str] = None
    _drawing_description: Optional[str] = None
    
    @classmethod
    def start_drawing(cls, intent: str = None, drawing_file: str = None, description: str = None):
        """Mark drawing as started"""
        with cls._lock:
            cls._is_drawing = True
            cls._drawing_start_time = time.time()
            cls._drawing_intent = intent
            cls._drawing_file = drawing_file
            cls._drawing_description = description
            print(f"[🎨] Drawing started: {description or 'Unknown subject'}")
    
    @classmethod
    def end_drawing(cls):
        """Mark drawing as completed"""
        with cls._lock:
            if cls._is_drawing:
                duration = time.time() - (cls._drawing_start_time or 0)
                print(f"[🎨] Drawing completed after {duration:.1f} seconds")
            
            cls._is_drawing = False
            cls._drawing_start_time = None
            cls._drawing_intent = None
            cls._drawing_file = None
            cls._drawing_description = None
    
    @classmethod
    def is_drawing(cls) -> bool:
        """Check if currently drawing. A flag older than _MAX_DRAWING_SECONDS is
        treated as leaked and cleared — see the note on the constant."""
        with cls._lock:
            if not cls._is_drawing:
                return False
            started = cls._drawing_start_time or 0
            if started and time.time() - started > cls._MAX_DRAWING_SECONDS:
                stale_for = time.time() - started
                cls._is_drawing = False
                cls._drawing_start_time = None
                cls._drawing_intent = None
                cls._drawing_file = None
                cls._drawing_description = None
                print(f"[⚠️] Drawing state was stuck for {stale_for / 60:.0f} min with no completion — clearing it (a drawing run raised past end_drawing)")
                return False
            return True

    # ------------------------------------------------------------------
    # Vision offline (July 30): ComfyUI unplugged means no drawing can be
    # visualised. The machine should KNOW that — a blocked draw attempt sets
    # this; the caption prompt and the reflection context read it, so an
    # evening of not-being-able-to-draw becomes identity-pertinent fact
    # instead of a silent failure.
    # ------------------------------------------------------------------

    _vision_offline_since: Optional[float] = None

    @classmethod
    def mark_vision_offline(cls):
        with cls._lock:
            if cls._vision_offline_since is None:
                cls._vision_offline_since = time.time()

    @classmethod
    def mark_vision_online(cls):
        with cls._lock:
            cls._vision_offline_since = None

    @classmethod
    def vision_offline_hours(cls) -> Optional[float]:
        """Hours since drawing generation became impossible, or None if fine."""
        with cls._lock:
            if cls._vision_offline_since is None:
                return None
            return (time.time() - cls._vision_offline_since) / 3600.0
    
    @classmethod
    def get_drawing_info(cls) -> Dict[str, Any]:
        """Get current drawing information"""
        with cls._lock:
            if not cls._is_drawing:
                return {}
                
            duration = time.time() - (cls._drawing_start_time or 0)
            return {
                "is_drawing": cls._is_drawing,
                "duration": duration,
                "intent": cls._drawing_intent,
                "file": cls._drawing_file,
                "description": cls._drawing_description,
                "start_time": cls._drawing_start_time
            }
    
    # Removed legacy get_drawing_context_for_caption (unused)
