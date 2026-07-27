"""Lightweight runtime hooks (no heavy imports).

Other subsystems can register callbacks here without creating import cycles.
"""

from typing import Callable, Optional

# Called by GRBL after G-code execution and homing completion ritual finishes
on_grbl_drawing_complete: Optional[Callable[[], None]] = None

# Kinetic bus homing safety (grbl_utils.ensure_homed): _start is called
# before each $H attempt and returns the seconds to WAIT while the left arm
# ramps to its tucked-clear pose (0 = nothing to wait for); _done fires when
# homing completes so the arm blends back into its running dataset.
on_grbl_homing_start: Optional[Callable[[], float]] = None
on_grbl_homing_done: Optional[Callable[[], None]] = None
