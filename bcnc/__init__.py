"""
bCNC utilities and converters package
Provides SVG to G-code conversion and related functionality
"""

import os
import sys

# Add current directory to path for internal imports
current_dir = os.path.dirname(__file__)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from .svg_centerliner import raster_to_centerline_svg
from .svg_to_gcode import svg_to_gcode

__all__ = ["raster_to_centerline_svg", "svg_to_gcode"]
