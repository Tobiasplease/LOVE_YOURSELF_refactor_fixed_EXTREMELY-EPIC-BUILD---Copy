"""
Egocentric view orientation helper based on servo pan/tilt.

Converts absolute pan/tilt angles into short, organic phrases like
"straight ahead", "slightly left and down", or "far right and up".
"""

from typing import Tuple


def _band(delta: float) -> Tuple[str, str]:
    """Return (magnitude_word, direction_word) for a signed delta."""
    if abs(delta) < 1e-6:
        return ("", "")
    mag = abs(delta)
    if mag < 15:
        mag_word = "slightly"
    elif mag < 30:
        mag_word = "moderately"
    else:
        mag_word = "far"
    return (mag_word, "right" if delta > 0 else "left")


def _band_tilt(delta: float) -> Tuple[str, str]:
    """Return (magnitude_word, direction_word) for tilt (up/down)."""
    if abs(delta) < 1e-6:
        return ("", "")
    mag = abs(delta)
    if mag < 15:
        mag_word = "slightly"
    elif mag < 30:
        mag_word = "moderately"
    else:
        mag_word = "far"
    return (mag_word, "up" if delta > 0 else "down")


def describe_view_orientation(pan: float, tilt: float, *, pan_center: float = 90.0, tilt_center: float = 90.0, tol: float = 8.0) -> str:
    """
    Build an egocentric orientation phrase from pan/tilt.

    Args:
        pan: Current pan angle (deg), increasing to the right
        tilt: Current tilt angle (deg), increasing upward
        pan_center: Neutral pan angle (default 90)
        tilt_center: Neutral tilt angle (default 90)
        tol: Dead-zone tolerance around center

    Returns:
        Short phrase like "straight ahead" or "slightly left and down".
    """
    dpan = pan - pan_center
    dtilt = tilt - tilt_center

    pan_term = None
    tilt_term = None

    if abs(dpan) <= tol and abs(dtilt) <= tol:
        return "straight ahead"

    if abs(dpan) > tol:
        mag, dir_ = _band(dpan)
        pan_term = f"{mag} {dir_}".strip()

    if abs(dtilt) > tol:
        mag, dir_ = _band_tilt(dtilt)
        tilt_term = f"{mag} {dir_}".strip()

    if pan_term and tilt_term:
        return f"{pan_term} and {tilt_term}"
    return pan_term or tilt_term or "straight ahead"


