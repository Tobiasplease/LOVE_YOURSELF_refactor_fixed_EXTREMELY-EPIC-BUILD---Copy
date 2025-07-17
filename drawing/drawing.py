from __future__ import annotations

import time
from typing import Optional


class DrawingController:
    """
    Determines whether Lint JR should draw, based on mood, novelty,
    self-evaluation, and elapsed time.
    """

    def __init__(self) -> None:
        self.last_drawing_time: float = 0.0
        self.cooldown: float = 600.0  # seconds between possible drawings
        self.last_prompt: Optional[str] = None

    def ready_to_draw(self) -> bool:
        """Checks whether enough time has passed to consider drawing."""
        return time.time() - self.last_drawing_time > self.cooldown

    def should_draw(
        self,
        mood: float,
        novelty: float,
        boredom: float,
        evaluation: Optional[str] = None,
    ) -> bool:
        """
        Uses a rough heuristic to decide whether to draw.
        Returns True only if all criteria suggest sufficient emotional or symbolic content.
        """
        if not self.ready_to_draw():
            return False

        if novelty > 0.65 or boredom > 0.7:
            return True

        if evaluation:
            lowered = evaluation.lower()
            if any(phrase in lowered for phrase in [
                "i feel stuck", "i need to express", "nothing is changing", "this might be important"
            ]):
                return True

        if mood < 0.3:
            return True

        return False

    def register_drawing(self, prompt: str) -> None:
        """
        Call this after a drawing decision has been made.
        Stores time and prompt.
        """
        self.last_drawing_time = time.time()
        self.last_prompt = prompt
