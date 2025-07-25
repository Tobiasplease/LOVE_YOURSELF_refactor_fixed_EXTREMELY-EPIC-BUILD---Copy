"""
Run management utilities for organizing logs and images by run ID.
"""

import os
from typing import Optional
from .event_logger import get_current_run_id


def get_run_image_folder(base_folder: str, run_id: Optional[str] = None) -> str:
    """
    Get the run-specific image folder path.

    Args:
        base_folder: Base folder for mood snapshots
        run_id: Optional run ID. If not provided, uses current run ID.

    Returns:
        Path to the run-specific image folder
    """
    if run_id is None:
        run_id = get_current_run_id()

    run_folder = os.path.join(base_folder, f"{run_id}-images")
    os.makedirs(run_folder, exist_ok=True)
    return run_folder


def get_run_image_path(base_folder: str, filename: str, run_id: Optional[str] = None) -> str:
    """
    Get the full path for an image file in the run-specific folder.

    Args:
        base_folder: Base folder for mood snapshots
        filename: Name of the image file
        run_id: Optional run ID. If not provided, uses current run ID.

    Returns:
        Full path to the image file in the run-specific folder
    """
    run_folder = get_run_image_folder(base_folder, run_id)
    return os.path.join(run_folder, filename)
