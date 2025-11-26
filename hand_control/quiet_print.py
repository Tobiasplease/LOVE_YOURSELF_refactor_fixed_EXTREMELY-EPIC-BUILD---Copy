"""Global print suppression for hand controller debug output."""


def apply_hand_controller_quiet_mode():
    """Apply global print suppression for hand controller output when DEBUG_HAND_CONTROLLER is False."""
    try:
        from config import config
    except ImportError:
        return  # If no config, don't suppress

    if getattr(config, "DEBUG_HAND_CONTROLLER", False):
        return  # Debug mode enabled, don't suppress

    import builtins

    original_print = builtins.print

    def quiet_print(*args, **kwargs):
        if args and isinstance(args[0], str):
            message = args[0]
            # Suppress any hand controller debug message with emojis
            if any(
                emoji in message
                for emoji in [
                    "🎯",
                    "🔄",
                    "📤",
                    "🎲",
                    "🔗",
                    "SUCCESS",
                    "🔒",
                    "⏳",
                    "🎨",
                    "🏋️",
                    "🛑",
                    "📁",
                    "⏱️",
                    "🌅",
                    "🌊",
                    "🤖",
                    "🔧",
                    "🎪",
                    "⚡",
                    "🔍",
                    "💡",
                    "📍",
                    "⭐",
                    "💾",
                    "📐",
                    "😊",
                    "🔌",
                    "ERROR",
                    "🧹",
                    "🎭",
                    "📊",
                    "💖",
                ]
            ):
                return
            # Also suppress specific text patterns
            if any(pattern in message for pattern in ["Using fonts", "[INFO]", "No background image", "Direct wave-based"]):
                return

        # Call original print for non-suppressed messages
        original_print(*args, **kwargs)

    # Replace global print
    builtins.print = quiet_print
    return original_print


def restore_original_print(original_print):
    """Restore the original print function."""
    if original_print:
        import builtins

        builtins.print = original_print
