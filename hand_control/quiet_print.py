"""Gated printing for hand controller output.

All informational/status prints in the hand_control package go through
hc_print, which stays silent unless config.DEBUG_HAND_CONTROLLER is True.
Errors and warnings always print. This replaces an earlier global
builtins.print monkeypatch that only covered headless startup.
"""

_ALWAYS_SHOW = ("ERROR", "WARNING", "Failed", "failed", "Exception", "Traceback", "❌", "⚠️")


def hc_print(*args, **kwargs):
    message = args[0] if args and isinstance(args[0], str) else ""
    if any(tag in message for tag in _ALWAYS_SHOW):
        print(*args, **kwargs)
        return
    try:
        from config import config
    except ImportError:
        print(*args, **kwargs)
        return
    if getattr(config, "DEBUG_HAND_CONTROLLER", False):
        print(*args, **kwargs)
