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

# Cross-process homing completion: the idle-movements SUBPROCESS homes the
# gantry at its startup, where the parent's hook registrations don't exist.
# ensure_homed touches this file on completion (any process); the kinetic
# bus watches its mtime to release the tucked left arm.
HOMING_SENTINEL = "/tmp/grbl_homing_complete.flag"

# Cross-process arm-clear gate: the parent starts the homing choreography
# and writes the epoch time at which the left arm WILL be clear; the idle
# subprocess spawns immediately (its ~10s of preamble — port find, alarm
# clear, pen-up — moves nothing) and ensure_homed sleeps until that moment
# before sending $H. Choreography and homing prep run in PARALLEL; the
# sweep fires the instant the arm is clear.
ARM_CLEAR_SENTINEL = "/tmp/left_arm_clear_at.flag"

# Gantry arbitration (July 28): the kinetic bus owns the gantry between
# drawings; the drawing pipeline's legacy pause/resume call sites (routed
# through grbl/idle_movement_manager) fire these so the bus releases the
# port before a drawing and re-acquires (and re-homes) after.
on_gantry_pause: Optional[Callable[[], None]] = None
on_gantry_resume: Optional[Callable[[], None]] = None
