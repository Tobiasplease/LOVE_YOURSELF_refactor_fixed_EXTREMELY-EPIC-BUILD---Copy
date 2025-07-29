"""
Simple resource manager implementation to handle ComfyUI resource coordination.
This is a minimal implementation to replace the missing resource_manager module.
"""
import time
from enum import Enum
from contextlib import contextmanager
from typing import Optional


class ResourcePriority(Enum):
    """Priority levels for resource requests."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ResourceManager:
    """Simple resource manager for coordinating system resources."""
    
    def __init__(self):
        self._comfyui_active = False
        self._current_operation = None
    
    @contextmanager
    def request_resource(self, priority: ResourcePriority = ResourcePriority.MEDIUM, 
                        operation_name: Optional[str] = None, timeout: float = 30.0):
        """
        Context manager for requesting system resources.
        
        Args:
            priority: Priority level for the request
            operation_name: Name of the operation requesting resources
            timeout: Maximum time to wait for resources
        """
        start_time = time.time()
        
        try:
            # Simple implementation - just track the operation
            self._current_operation = operation_name
            yield
        finally:
            self._current_operation = None
    
    def set_comfyui_active(self, active: bool, operation: Optional[str] = None):
        """
        Set ComfyUI active status to coordinate with other system components.
        
        Args:
            active: Whether ComfyUI is active
            operation: Name of the operation setting the status
        """
        self._comfyui_active = active
        if active and operation:
            self._current_operation = operation
    
    def is_comfyui_active(self) -> bool:
        """Check if ComfyUI is currently active."""
        return self._comfyui_active
    
    def get_current_operation(self) -> Optional[str]:
        """Get the name of the currently active operation."""
        return self._current_operation


# Global resource manager instance
resource_manager = ResourceManager()