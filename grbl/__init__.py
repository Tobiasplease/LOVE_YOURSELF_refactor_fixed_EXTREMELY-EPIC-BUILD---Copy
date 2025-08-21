"""
GRBL Control Module
Functions for GRBL communication, control, and SVG processing
"""

from .grbl_utils import (
    find_grbl_port,
    send_cmd,
    wait_until_idle,
    ensure_homed,
    setup_basic_grbl,
    pen_control,
    execute_gcode_file,
    initialize_grbl_for_drawing,
    process_svg_to_grbl,
)


def svg_to_grbl(svg_input, output_gcode=None, execute_grbl=True):
    """
    Convert SVG to G-code and optionally execute on GRBL hardware
    Legacy wrapper for compatibility with image_monitor

    Args:
        svg_input: Path to input SVG file
        output_gcode: Path for output G-code file (optional)
        auto_run: Legacy parameter for compatibility (ignored)
        execute_grbl: Whether to execute the G-code on GRBL hardware

    Returns:
        str: Path to generated G-code file if successful, None if failed
    """
    return process_svg_to_grbl(svg_input, output_gcode, execute_grbl)


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
