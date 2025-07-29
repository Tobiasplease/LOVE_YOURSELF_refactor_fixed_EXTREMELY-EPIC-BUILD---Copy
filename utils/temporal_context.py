"""
utils/temporal_context.py
-----------------------
Enhanced temporal awareness system for AI consciousness.

Provides rich temporal context about:
- Session duration and continuity
- Memory recency vs antiquity 
- Temporal relationships between observations
- Self-awareness of time passage
"""

import time
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from utils.continuity import now, describe_duration


class TemporalContextManager:
    """Manages temporal awareness and context for the AI system."""
    
    def __init__(self):
        self.session_start_time = now()
        self.last_major_event_time = now()
        self.temporal_markers: List[Dict] = []
        self.awareness_intervals = {
            'immediate': 30,      # 30 seconds - immediate context
            'recent': 300,        # 5 minutes - recent events  
            'session': 3600,      # 1 hour - current session context
            'memory': 86400,      # 1 day - memory context
            'historical': 604800  # 1 week - historical context
        }
    
    def get_session_duration(self) -> str:
        """Get formatted duration of current session."""
        return describe_duration(self.session_start_time)
    
    def get_temporal_context(self, agent_memory=None) -> Dict[str, str]:
        """Generate lightweight temporal context for prompts."""
        session_duration = self.get_session_duration()
        
        context = {
            'time_of_day': self._get_time_of_day_context(),
            'session_duration': self._get_consciousness_state(session_duration),
        }
        
        return context
    
    def _get_time_of_day_context(self) -> str:
        """Generate simple time-of-day awareness."""
        hour = datetime.now().hour
        
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"
    
    def _get_consciousness_state(self, session_duration: str) -> str:
        """Simple session duration context."""
        return f"been observing for {session_duration}"
    
    def _get_memory_timeline(self, agent_memory) -> str:
        """Generate temporal context for memories."""
        if not hasattr(agent_memory, 'memory_queue') or not agent_memory.memory_queue:
            return "no previous observations in memory"
        
        current_time = now()
        timeline_segments = []
        
        # Categorize memories by recency
        immediate = []
        recent = []
        session = []
        
        for entry in reversed(list(agent_memory.memory_queue)):
            age = current_time - entry['timestamp']
            if age <= self.awareness_intervals['immediate']:
                immediate.append(entry)
            elif age <= self.awareness_intervals['recent']:
                recent.append(entry)
            elif age <= self.awareness_intervals['session']:
                session.append(entry)
        
        if immediate:
            timeline_segments.append(f"just witnessed ({len(immediate)} immediate observations)")
        if recent:
            timeline_segments.append(f"recently noticed ({len(recent)} recent memories)")
        if session:
            timeline_segments.append(f"earlier this session ({len(session)} accumulated observations)")
        
        return " | ".join(timeline_segments) if timeline_segments else "sparse memory context"
    
    def _get_temporal_perspective(self, current_time: float) -> str:
        """Generate perspective on temporal flow."""
        session_seconds = current_time - self.session_start_time
        
        if session_seconds < 300:  # 5 minutes
            return "time feels immediate and present"
        elif session_seconds < 1800:  # 30 minutes
            return "time is accumulating into continuity"
        elif session_seconds < 3600:  # 1 hour
            return "time has woven itself into familiarity"
        else:
            return "time has created depth and history"
    
    def mark_temporal_event(self, event_type: str, description: str):
        """Mark significant temporal events for context."""
        self.temporal_markers.append({
            'timestamp': now(),
            'type': event_type,
            'description': description
        })
        
        # Keep only recent markers
        cutoff = now() - self.awareness_intervals['session']
        self.temporal_markers = [m for m in self.temporal_markers if m['timestamp'] > cutoff]
    
    def get_memory_age_context(self, memory_item) -> str:
        """Get contextual description of memory age."""
        if not memory_item or 'timestamp' not in memory_item:
            return "timeless"
        
        age = now() - memory_item['timestamp']
        
        if age < 30:
            return "moments ago"
        elif age < 300:
            return "recently"
        elif age < 1800:
            return "some time ago"
        elif age < 3600:
            return "earlier this session"
        elif age < 86400:
            return "from previous awareness"
        else:
            return "from distant memory"
    
    def should_update_temporal_awareness(self) -> bool:
        """Determine if temporal context should be refreshed."""
        current_time = now()
        return (current_time - self.last_major_event_time) > 60  # Update every minute
    
    def get_identity_temporal_context(self, beliefs: Dict) -> str:
        """Generate temporal context for identity formation."""
        if not beliefs:
            return "identity is nascent, forming through immediate experience"
        
        belief_ages = []
        for motif, data in beliefs.items():
            age = now() - data.get('first_formed', now())
            belief_ages.append((motif, age))
        
        # Sort by age (oldest first)
        belief_ages.sort(key=lambda x: x[1], reverse=True)
        
        if belief_ages:
            oldest_motif, oldest_age = belief_ages[0]
            oldest_desc = describe_duration(oldest_age)
            
            if len(belief_ages) == 1:
                return f"identity anchored by {oldest_motif} (formed {oldest_desc} ago)"
            else:
                return f"identity layered over time: {oldest_motif} anchors from {oldest_desc} ago, with {len(belief_ages)-1} other beliefs"
        
        return "identity emerging through temporal experience"
