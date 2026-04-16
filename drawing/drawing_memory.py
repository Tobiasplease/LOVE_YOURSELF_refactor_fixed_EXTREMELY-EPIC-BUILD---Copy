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
        self._last_failure = None
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    self._history = data.get('drawings', [])[:self.max_history]
                    self._last_failure = data.get('last_failure', None)
            except Exception as e:
                print(f"[⚠️] Could not load drawing memory: {e}")
                self._history = []

    def _save_memory(self) -> None:
        """Save drawing memory to disk."""
        try:
            self.memory_file.parent.mkdir(parents=True, exist_ok=True)
            data = {'drawings': self._history}
            if self._last_failure:
                data['last_failure'] = self._last_failure
            with open(self.memory_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[⚠️] Could not save drawing memory: {e}")

    def add_drawing(
        self,
        prompt: str,
        compressed_summary: str,
        theme_tags: Optional[List[str]] = None,
        emotional_tone: Optional[str] = None,
        narrative_thread: Optional[str] = None,
        comfy_prompt: Optional[str] = None,
        completed: bool = True,
    ) -> None:
        """Add a new drawing to memory with compressed metadata."""
        import time

        entry = {
            'timestamp': time.time(),
            'compressed_summary': compressed_summary[:120],
            'theme_tags': (theme_tags or [])[:3],
            'emotional_tone': (emotional_tone or '')[:30],
            'narrative_thread': (narrative_thread or '')[:50],
            'comfy_prompt': (comfy_prompt or '')[:200],  # The actual prompt sent to ComfyUI
            'completed': completed,  # Did GRBL actually finish drawing this?
        }

        self._history.insert(0, entry)
        self._history = self._history[:self.max_history]
        self._save_memory()

        print(f"[📚] Stored drawing memory: {compressed_summary}")

    def record_failure(self, reason: str, prompt: Optional[str] = None) -> None:
        """Record a failed drawing attempt — no paper, ComfyUI failure, etc."""
        import time

        self._last_failure = {
            'timestamp': time.time(),
            'reason': reason,
            'prompt': (prompt or '')[:200],
        }
        self._save_memory()
        print(f"[📚] Drawing failed: {reason}")

    def get_last_failure(self) -> Optional[Dict]:
        """Get the most recent drawing failure, if any."""
        return getattr(self, '_last_failure', None)

    def get_recent_drawings_summary(self, max_count: int = 3) -> str:
        """Get a very compressed summary of recent drawings for prompt context."""
        if not self._history:
            return ""

        recent = self._history[:max_count]

        # Build ultra-compact summary — only use real ComfyUI prompts
        parts = []
        for entry in recent:
            desc = entry.get('comfy_prompt', '')
            if not desc:
                continue  # Skip entries without real prompts — tag-clouds cause confabulation
            # Strip the standard ComfyUI preamble to get the actual subject
            for prefix in ["Black ink line drawing on white paper. ",
                           "Black ink line drawing on white paper.",
                           "Black ink drawing on white paper. ",
                           "black ink line drawing on white paper. "]:
                if desc.lower().startswith(prefix.lower()):
                    desc = desc[len(prefix):]
                    break
            # Truncate to a reasonable length
            if len(desc) > 80:
                desc = desc[:80].rsplit(" ", 1)[0] + "..."
            parts.append(desc.strip())

        if parts:
            return "Recent drawings: " + "; ".join(parts)
        return ""

    def get_last_drawing_description(self) -> str:
        """Get a clean description of the most recent drawing, with temporal context.

        Returns something like: "two figures hunched over computers (about 10 minutes ago)"
        or empty string if no drawings exist.
        """
        if not self._history:
            return ""

        import time
        entry = self._history[0]

        # Only use actual ComfyUI prompt — tag-cloud summaries like
        # "Cluttered Room, Bending Man, Introspection" cause the model
        # to confabulate fictional drawing titles and descriptions.
        desc = entry.get('comfy_prompt', '')
        if not desc:
            # No real prompt stored — just report temporal context without misleading tags
            elapsed = time.time() - entry.get('timestamp', time.time())
            if elapsed < 3600:
                mins = int(elapsed / 60)
                return f"something about {mins} minutes ago"
            else:
                hours = int(elapsed / 3600)
                return f"something about {hours} hours ago"

        # Strip ComfyUI preamble
        for prefix in ["Black ink line drawing on white paper. ",
                       "Black ink line drawing on white paper.",
                       "Black ink drawing on white paper. ",
                       "black ink line drawing on white paper. "]:
            if desc.lower().startswith(prefix.lower()):
                desc = desc[len(prefix):]
                break

        # Truncate
        if len(desc) > 70:
            desc = desc[:70].rsplit(" ", 1)[0] + "..."

        # Add temporal context
        elapsed = time.time() - entry.get('timestamp', time.time())
        if elapsed < 120:
            time_str = "just now"
        elif elapsed < 3600:
            mins = int(elapsed / 60)
            time_str = f"about {mins} minutes ago"
        elif elapsed < 7200:
            time_str = "about an hour ago"
        else:
            hours = int(elapsed / 3600)
            time_str = f"about {hours} hours ago"

        return f"{desc.strip()} ({time_str})"

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
