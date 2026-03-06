"""
Gaze intent extraction from captions.
Parses natural language for directional cues to guide camera movement.
Only triggers on intentional directional language, not incidental word matches.
"""

import re
from typing import Optional, Tuple

# Explicit gaze commands (highest priority) - these are unambiguous
EXPLICIT_COMMANDS = {
    "look up": ("up", 1.0),
    "look down": ("down", 1.0),
    "look left": ("left", 1.0),
    "look right": ("right", 1.0),
    "looking up": ("up", 0.9),
    "looking down": ("down", 0.9),
    "looking left": ("left", 0.9),
    "looking right": ("right", 0.9),
    "glance up": ("up", 0.8),
    "glance down": ("down", 0.8),
    "glance left": ("left", 0.8),
    "glance right": ("right", 0.8),
    "gaze up": ("up", 0.9),
    "gaze down": ("down", 0.9),
    "gaze left": ("left", 0.9),
    "gaze right": ("right", 0.9),
    "turn up": ("up", 1.0),
    "turn down": ("down", 1.0),
    "turn left": ("left", 1.0),
    "turn right": ("right", 1.0),
}

# Directional phrases that indicate intentional looking (require word boundaries)
# Format: (regex pattern, direction, intensity)
DIRECTIONAL_PHRASES = [
    # UP - intentional upward references
    (r"\bupward\b", "up", 0.7),
    (r"\babove me\b", "up", 0.8),
    (r"\boverhead\b", "up", 0.7),
    (r"\bthe ceiling\b", "up", 0.8),
    (r"\btoward the ceiling\b", "up", 0.9),
    (r"\bup at\b", "up", 0.7),
    (r"\bup toward\b", "up", 0.8),

    # DOWN - intentional downward references
    (r"\bdownward\b", "down", 0.7),
    (r"\bbelow me\b", "down", 0.8),
    (r"\bbeneath\b", "down", 0.6),
    (r"\bthe floor\b", "down", 0.7),
    (r"\bdown at\b", "down", 0.7),
    (r"\bdown toward\b", "down", 0.8),
    (r"\bmy workspace\b", "down", 0.5),
    (r"\bthe paper\b", "down", 0.5),
    (r"\bdrawing surface\b", "down", 0.6),

    # LEFT - intentional leftward references
    (r"\bto my left\b", "left", 0.8),
    (r"\bon my left\b", "left", 0.8),
    (r"\bleftward\b", "left", 0.7),
    (r"\bto the left\b", "left", 0.6),

    # RIGHT - intentional rightward references
    (r"\bto my right\b", "right", 0.8),
    (r"\bon my right\b", "right", 0.8),
    (r"\brightward\b", "right", 0.7),
    (r"\bto the right\b", "right", 0.6),
]

# False positive patterns - if these match, ignore the directional keyword
FALSE_POSITIVE_PATTERNS = [
    r"wak\w* up",      # "waking up", "woke up"
    r"set up",         # "set up", "setup"
    r"up from",        # "up from sleep"
    r"pick up",        # "pick up"
    r"show up",        # "show up"
    r"up to",          # "up to date"
    r"up on",          # "caught up on"
    r"put down",       # "put down"
    r"sit down",       # "sit down"
    r"write down",     # "write down"
    r"calm down",      # "calm down"
    r"left side",      # describing position, not direction
    r"left of",        # "to the left of" - position
    r"on the left",    # position description (unless explicit "look on the left")
    r"right side",     # describing position
    r"right of",       # position
    r"on the right",   # position description
    r"that's right",   # affirmation
    r"all right",      # acknowledgment
]


def extract_gaze_intent(caption: str) -> Optional[Tuple[str, float]]:
    """
    Extract directional gaze intent from a caption.
    Only matches intentional directional language, not incidental word matches.

    Returns:
        Tuple of (direction, intensity) or None if no directional cue found.
        direction: "up", "down", "left", "right"
        intensity: 0.0 to 1.0, how strongly to nudge in that direction
    """
    if not caption:
        return None

    text = caption.lower()

    # First check for false positives - if any match, be more cautious
    has_false_positive = any(re.search(pattern, text) for pattern in FALSE_POSITIVE_PATTERNS)

    # Check explicit gaze commands first (highest priority, ignore false positives)
    for command, (direction, intensity) in sorted(EXPLICIT_COMMANDS.items(), key=lambda x: -len(x[0])):
        if command in text:
            return (direction, intensity)

    # If we detected a false positive pattern, require stronger evidence
    if has_false_positive:
        # Only match very explicit directional phrases
        for pattern, direction, intensity in DIRECTIONAL_PHRASES:
            if re.search(pattern, text) and intensity >= 0.7:
                return (direction, intensity)
        return None

    # Check directional phrases (regex with word boundaries)
    best_match = None
    best_intensity = 0.0

    for pattern, direction, intensity in DIRECTIONAL_PHRASES:
        if re.search(pattern, text):
            if intensity > best_intensity:
                best_intensity = intensity
                best_match = direction

    if best_match and best_intensity >= 0.5:
        return (best_match, best_intensity)

    return None


def get_nudge_deltas(direction: str, intensity: float, max_pan_delta: float = 20.0, max_tilt_delta: float = 15.0) -> Tuple[float, float]:
    """
    Convert direction and intensity to pan/tilt delta values.

    Reduced from 25/18 to 20/15 for more subtle nudging.

    Returns:
        Tuple of (pan_delta, tilt_delta) in degrees

    Note: Tilt direction calibrated for TILT_MIN=down, TILT_MAX=up servo config.
    Higher tilt values = looking UP.
    """
    pan_delta = 0.0
    tilt_delta = 0.0

    if direction == "up":
        tilt_delta = max_tilt_delta * intensity   # Positive = up (higher servo value)
    elif direction == "down":
        tilt_delta = -max_tilt_delta * intensity  # Negative = down (lower servo value)
    elif direction == "left":
        pan_delta = -max_pan_delta * intensity    # Negative = left
    elif direction == "right":
        pan_delta = max_pan_delta * intensity     # Positive = right

    return (pan_delta, tilt_delta)
