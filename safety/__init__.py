"""
Safety Systems Module

Provides safety systems for preventing damage and ensuring safe operation
of the drawing machine system.

Current safety systems:
- Paper Detection: Prevents drawing on bare surfaces
"""

from .paper_detection import PaperCheckResult, check_paper_before_drawing, get_paper_detection_status, paper_detector

__all__ = ["paper_detector", "check_paper_before_drawing", "get_paper_detection_status", "PaperCheckResult"]
