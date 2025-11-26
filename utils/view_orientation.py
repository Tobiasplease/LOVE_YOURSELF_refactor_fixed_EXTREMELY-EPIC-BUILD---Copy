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


def get_spatial_language_bias(pan: float, tilt: float, *, pan_center: float = 90.0, tilt_center: float = 90.0, tol: float = 8.0) -> list[str]:
    """
    Get spatial language hints based on current view orientation.

    Returns natural spatial phrases that should feel organic when describing
    what the consciousness observes, without explicitly stating the view angle.

    Args:
        pan: Current pan angle (deg), increasing to the right
        tilt: Current tilt angle (deg), increasing upward
        pan_center: Neutral pan angle (default 90)
        tilt_center: Neutral tilt angle (default 90)
        tol: Dead-zone tolerance around center

    Returns:
        List of spatial phrases that feel natural for current orientation
    """
    dpan = pan - pan_center
    dtilt = tilt - tilt_center

    spatial_hints = []

    # Tilt-based spatial language (vertical direction)
    if dtilt < -tol:  # Looking down
        magnitude = abs(dtilt)
        if magnitude > 30:  # Far down
            spatial_hints.extend(["below me", "down here", "on the surface beneath", "at the base"])
        elif magnitude > 15:  # Moderately down
            spatial_hints.extend(["below", "down on", "on the surface", "beneath"])
        else:  # Slightly down
            spatial_hints.extend(["before me", "on the surface", "at hand level"])

    elif dtilt > tol:  # Looking up
        magnitude = abs(dtilt)
        if magnitude > 30:  # Far up
            spatial_hints.extend(["high above", "overhead", "up there", "on the ceiling"])
        elif magnitude > 15:  # Moderately up
            spatial_hints.extend(["above", "overhead", "up there"])
        else:  # Slightly up
            spatial_hints.extend(["above", "higher up"])

    # Pan-based spatial language (horizontal direction)
    if dpan < -tol:  # Looking left
        magnitude = abs(dpan)
        if magnitude > 30:  # Far left
            spatial_hints.extend(["far to the left", "way over there", "on the left side"])
        elif magnitude > 15:  # Moderately left
            spatial_hints.extend(["to the left", "over there", "on that side"])
        else:  # Slightly left
            spatial_hints.extend(["to the left", "over that way"])

    elif dpan > tol:  # Looking right
        magnitude = abs(dpan)
        if magnitude > 30:  # Far right
            spatial_hints.extend(["far to the right", "way over there", "on the right side"])
        elif magnitude > 15:  # Moderately right
            spatial_hints.extend(["to the right", "over there", "on that side"])
        else:  # Slightly right
            spatial_hints.extend(["to the right", "over that way"])

    # Neutral/center case
    if abs(dpan) <= tol and abs(dtilt) <= tol:
        spatial_hints.extend(["in front of me", "straight ahead", "directly before me"])

    # Always include some general spatial terms that work in any direction
    spatial_hints.extend(["nearby", "within view", "in this area"])

    return spatial_hints
