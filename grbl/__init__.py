"""
GRBL Control Module
Functions for GRBL communication, control, and SVG processing
"""

from .grbl_utils import (
    ensure_homed,
    execute_gcode_file,
    find_grbl_port,
    initialize_grbl_for_drawing,
    pen_control,
    process_svg_to_grbl,
    send_cmd,
    setup_basic_grbl,
    wait_until_idle,
)


def svg_to_grbl(svg_input, output_gcode=None, execute_grbl=True, scale_to=None):
    """
    Convert SVG to G-code and optionally execute on GRBL hardware
    Legacy wrapper for compatibility with image_monitor

    Defaults to DRAWING_SCALE_TARGET from config (65x35mm landscape, matching warp quad)

    Args:
        svg_input: Path to input SVG file
        output_gcode: Path for output G-code file (optional)
        auto_run: Legacy parameter for compatibility (ignored)
        execute_grbl: Whether to execute the G-code on GRBL hardware
        scale_to: Scale dimensions (default: "50x50mm" for safety)

    Returns:
        str: Path to generated G-code file if successful, None if failed
    """
    if scale_to is None:
        try:
            from config.config import DRAWING_SCALE_TARGET

            scale_to = DRAWING_SCALE_TARGET
        except (ImportError, AttributeError):
            scale_to = "65x35mm"
    return process_svg_to_grbl(svg_input, output_gcode, execute_grbl, scale_to=scale_to)


__all__ = [
    "find_grbl_port",
    "send_cmd",
    "wait_until_idle",
    "ensure_homed",
    "setup_basic_grbl",
    "pen_control",
    "execute_gcode_file",
    "initialize_grbl_for_drawing",
    "process_svg_to_grbl",
    "svg_to_grbl",
]
