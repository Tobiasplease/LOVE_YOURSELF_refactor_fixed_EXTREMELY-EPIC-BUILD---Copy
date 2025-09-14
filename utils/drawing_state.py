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
        """Check if currently drawing"""
        with cls._lock:
            return cls._is_drawing
    
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
