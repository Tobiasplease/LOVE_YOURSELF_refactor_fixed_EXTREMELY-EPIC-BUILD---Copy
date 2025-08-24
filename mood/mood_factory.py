# mood/mood_factory.py
"""
Mood system factory - creates appropriate mood engine based on configuration.
Provides seamless switching between legacy and experiential systems.
"""

from typing import Union
from config.config import USE_EXPERIENTIAL_MOOD

def create_mood_engine() -> Union['MoodEngine', 'ExperientialMoodEngine']:
    """
    Create the appropriate mood engine based on configuration.
    
    Returns:
        MoodEngine or ExperientialMoodEngine with identical interfaces
    """
    if USE_EXPERIENTIAL_MOOD:
        try:
            from .experiential_mood import ExperientialMoodEngine
            # Using experiential mood system for organic emotional evolution
            return ExperientialMoodEngine()
        except ImportError as e:
            # Failed to import experiential mood system, falling back to legacy
            from .mood import MoodEngine
            return MoodEngine()
    else:
        from .mood import MoodEngine
        # Using legacy numerical mood system
        return MoodEngine()