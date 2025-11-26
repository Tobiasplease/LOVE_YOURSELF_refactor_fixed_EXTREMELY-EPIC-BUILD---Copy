"""Lightweight runtime hooks (no heavy imports).

Other subsystems can register callbacks here without creating import cycles.
"""

from typing import Callable, Optional

# Called by GRBL after G-code execution and homing completion ritual finishes
on_grbl_drawing_complete: Optional[Callable[[], None]] = None

