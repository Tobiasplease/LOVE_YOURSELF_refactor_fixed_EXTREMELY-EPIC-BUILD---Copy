"""
Compressed drawing memory for thematic continuity.
Stores minimal metadata about recent drawings to inform future drawing decisions.
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from config.config import MOOD_SNAPSHOT_FOLDER


class DrawingMemory:
    """Manages compressed history of recent drawings for thematic continuity."""

    def __init__(self, max_history: int = 5):
        self.max_history = max_history
        self.memory_file = Path(MOOD_SNAPSHOT_FOLDER) / "drawing_memory.json"
        self._history: List[Dict] = []
        self._load_memory()

    def _load_memory(self) -> None:
        """Load existing drawing memory from disk."""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    self._history = data.get('drawings', [])[:self.max_history]
            except Exception as e:
                print(f"[⚠️] Could not load drawing memory: {e}")
                self._history = []

    def _save_memory(self) -> None:
        """Save drawing memory to disk."""
        try:
            self.memory_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.memory_file, 'w') as f:
                json.dump({'drawings': self._history}, f, indent=2)
        except Exception as e:
            print(f"[⚠️] Could not save drawing memory: {e}")

    def add_drawing(
        self,
        prompt: str,
        compressed_summary: str,
        theme_tags: Optional[List[str]] = None,
        emotional_tone: Optional[str] = None,
        narrative_thread: Optional[str] = None
    ) -> None:
        """Add a new drawing to memory with compressed metadata."""
        import time

        entry = {
            'timestamp': time.time(),
            'compressed_summary': compressed_summary[:120],  # Max 120 chars — enough for a real sentence
            'theme_tags': (theme_tags or [])[:3],  # Max 3 tags
            'emotional_tone': (emotional_tone or '')[:30],  # Max 30 chars
            'narrative_thread': (narrative_thread or '')[:50],  # Max 50 chars
        }

        self._history.insert(0, entry)
        self._history = self._history[:self.max_history]
        self._save_memory()

        print(f"[📚] Stored drawing memory: {compressed_summary}")

    def get_recent_drawings_summary(self, max_count: int = 3) -> str:
        """Get a very compressed summary of recent drawings for prompt context."""
        if not self._history:
            return ""

        recent = self._history[:max_count]

        # Build ultra-compact summary
        parts = []
        for entry in recent:
            summary = entry['compressed_summary']
            tone = entry.get('emotional_tone', '')
            if tone:
                parts.append(f"{summary} ({tone})")
            else:
                parts.append(summary)

        if parts:
            return "Recent drawings: " + "; ".join(parts)
        return ""

    def get_thematic_context(self) -> Dict[str, any]:
        """Get thematic patterns from recent drawings."""
        if not self._history:
            return {}

        # Aggregate theme tags
        all_tags = []
        all_tones = []

        for entry in self._history[:3]:
            all_tags.extend(entry.get('theme_tags', []))
            tone = entry.get('emotional_tone', '')
            if tone:
                all_tones.append(tone)

        return {
            'recurring_themes': list(set(all_tags)),
            'recent_tones': all_tones,
            'drawing_count': len(self._history)
        }


# Global singleton
_drawing_memory = None

def get_drawing_memory() -> DrawingMemory:
    """Get the global drawing memory instance."""
    global _drawing_memory
    if _drawing_memory is None:
        _drawing_memory = DrawingMemory()
    return _drawing_memory
