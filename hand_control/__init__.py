"""
Hand Control Module
==================

Direct integration of hand controller into the main codebase.
Replaces the complex bridge system with direct function calls.
"""

from .hand_control_interface import HandControlInterface

__all__ = ['HandControlInterface']
