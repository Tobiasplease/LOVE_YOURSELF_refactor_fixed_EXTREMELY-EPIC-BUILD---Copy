"""
state_manager.py
----------------
Handles session persistence for the LOVE_YOURSELF mirror system.
Saves and loads system state between sessions to enable continuity,
memory retention, and identity evolution.
"""

import json
import os
import time
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from utils.continuity import now, describe_duration
from config.config import MOOD_SNAPSHOT_FOLDER


class StateManager:
    def __init__(self, state_file: str = "system_state.json"):
        self.state_file = os.path.join(MOOD_SNAPSHOT_FOLDER, state_file)
        self.lifetime_state_file = os.path.join(MOOD_SNAPSHOT_FOLDER, "lifetime_state.json")
        
    def save_session_state(self, captioner, mood_engine, timekeeper=None) -> bool:
        """Save current session state for next startup."""
        try:
            state = {
                "metadata": {
                    "save_time": now(),
                    "save_datetime": datetime.now().isoformat(),
                    "session_duration": time.time() - captioner.true_session_start,
                    "version": "1.0"
                },
                
                # Captioner/Memory state
                "captioner": {
                    "current_mood": captioner.current_mood,
                    "last_caption": captioner.last_caption,
                    "boredom": captioner.boredom,
                    "novelty_score": captioner.novelty_score,
                    "awakening_done": captioner.awakening_done,
                    
                    # Memory system
                    "motif_counter": dict(captioner.motif_counter),
                    "motif_first_seen": captioner.motif_first_seen,
                    "motif_last_seen": captioner.motif_last_seen,
                    "motif_confidence": captioner.motif_confidence,
                    "motif_confirmed": captioner.motif_confirmed,
                    "current_motifs": list(captioner.current_motifs),
                    
                    # Identity/Beliefs
                    "beliefs": captioner.beliefs,
                    "belief_history": captioner.belief_history,
                    
                    # Recent memory (last 10 entries)
                    "recent_memory": list(captioner.memory_queue)[-10:] if captioner.memory_queue else []
                },
                
                # Mood engine state
                "mood_engine": {
                    "current_mood": mood_engine.current_mood,
                    "last_caption": mood_engine.last_caption,
                    "last_person_detected": mood_engine.last_person_detected
                },
                
                # Timekeeper state (if available)
                "timekeeper": self._get_timekeeper_state(timekeeper) if timekeeper else {}
            }
            
            # Write to temp file first, then rename for atomic operation
            temp_file = self.state_file + ".tmp"
            
            # Remove temp file if it exists (Windows issue)
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
            
            # Remove target file if it exists (Windows issue)
            if os.path.exists(self.state_file):
                os.remove(self.state_file)
            
            # Atomic rename
            os.rename(temp_file, self.state_file)
            
            # Update lifetime stats
            self._update_lifetime_stats(state)
            
            print(f"[💾] Session state saved to {self.state_file}")
            return True
            
        except Exception as e:
            print(f"[❌] Failed to save session state: {e}")
            return False
    
    def load_session_state(self) -> Optional[Dict[str, Any]]:
        """Load previous session state if available."""
        if not os.path.exists(self.state_file):
            print("[🆕] No previous session state found - starting fresh")
            return None
            
        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
                
            # Validate state format
            if not self._validate_state(state):
                print("[⚠️] Invalid state format - starting fresh")
                return None
                
            save_time = state["metadata"]["save_time"]
            time_since_save = describe_duration(save_time)
            
            print(f"[🔄] Loading session state from {time_since_save} ago")
            return state
            
        except Exception as e:
            print(f"[❌] Failed to load session state: {e}")
            return None
    
    def apply_state_to_captioner(self, state: Dict[str, Any], captioner) -> bool:
        """Apply loaded state to captioner instance."""
        try:
            cap_state = state["captioner"]
            
            # Restore mood and state
            captioner.current_mood = cap_state.get("current_mood", 0.5)
            captioner.last_caption = cap_state.get("last_caption", "")
            captioner.boredom = cap_state.get("boredom", 0.0)
            captioner.novelty_score = cap_state.get("novelty_score", 1.0)
            captioner.awakening_done = cap_state.get("awakening_done", False)
            
            # Restore memory system
            from collections import Counter, deque
            captioner.motif_counter = Counter(cap_state.get("motif_counter", {}))
            captioner.motif_first_seen = cap_state.get("motif_first_seen", {})
            captioner.motif_last_seen = cap_state.get("motif_last_seen", {})
            captioner.motif_confidence = cap_state.get("motif_confidence", {})
            captioner.motif_confirmed = cap_state.get("motif_confirmed", {})
            captioner.current_motifs = set(cap_state.get("current_motifs", []))
            
            # Restore identity
            captioner.beliefs = cap_state.get("beliefs", {})
            captioner.belief_history = cap_state.get("belief_history", [])
            
            # Restore recent memory
            recent_memory = cap_state.get("recent_memory", [])
            captioner.memory_queue = deque(recent_memory, maxlen=30)
            
            # Update session start time to maintain continuity
            save_time = state["metadata"]["save_time"]
            captioner.true_session_start = save_time
            
            print(f"[✅] Restored captioner state: {len(captioner.beliefs)} beliefs, {len(captioner.motif_counter)} motifs")
            return True
            
        except Exception as e:
            print(f"[❌] Failed to apply captioner state: {e}")
            return False
    
    def apply_state_to_mood_engine(self, state: Dict[str, Any], mood_engine) -> bool:
        """Apply loaded state to mood engine."""
        try:
            mood_state = state["mood_engine"]
            
            mood_engine.current_mood = mood_state.get("current_mood", 0.5)
            mood_engine.last_caption = mood_state.get("last_caption", "")
            mood_engine.last_person_detected = mood_state.get("last_person_detected", False)
            
            print(f"[✅] Restored mood engine state: mood={mood_engine.current_mood:.2f}")
            return True
            
        except Exception as e:
            print(f"[❌] Failed to apply mood engine state: {e}")
            return False
    
    def get_lifetime_stats(self) -> Dict[str, Any]:
        """Get lifetime statistics across all sessions."""
        if not os.path.exists(self.lifetime_state_file):
            return {"total_sessions": 0, "total_runtime": 0, "first_boot": now()}
            
        try:
            with open(self.lifetime_state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {"total_sessions": 0, "total_runtime": 0, "first_boot": now()}
    
    def _get_timekeeper_state(self, timekeeper) -> Dict[str, Any]:
        """Extract timekeeper state if available."""
        try:
            return {
                "system_start": timekeeper.system_start,
                "last_wake": timekeeper.last_wake,
                "session_starts": timekeeper.session_starts[-5:]  # Last 5 sessions
            }
        except Exception:
            return {}
    
    def _validate_state(self, state: Dict[str, Any]) -> bool:
        """Validate that state has required structure."""
        required_keys = ["metadata", "captioner", "mood_engine"]
        return all(key in state for key in required_keys)
    
    def _update_lifetime_stats(self, current_state: Dict[str, Any]):
        """Update lifetime statistics."""
        try:
            lifetime_stats = self.get_lifetime_stats()
            
            lifetime_stats["total_sessions"] = lifetime_stats.get("total_sessions", 0) + 1
            lifetime_stats["total_runtime"] = lifetime_stats.get("total_runtime", 0) + current_state["metadata"]["session_duration"]
            lifetime_stats["last_session"] = current_state["metadata"]["save_time"]
            
            if "first_boot" not in lifetime_stats:
                lifetime_stats["first_boot"] = current_state["metadata"]["save_time"]
            
            with open(self.lifetime_state_file, 'w', encoding='utf-8') as f:
                json.dump(lifetime_stats, f, indent=2)
                
        except Exception as e:
            print(f"[⚠️] Failed to update lifetime stats: {e}")


# Global instance
state_manager = StateManager()
