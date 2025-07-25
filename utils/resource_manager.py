"""
Resource Manager for coordinating GPU/AI resource access between Ollama and ComfyUI services.

This module provides a singleton ResourceManager that prevents resource contention by:
- Queuing Ollama calls when ComfyUI is generating images
- Using priority-based resource allocation
- Integrating with ImageMonitor to detect ComfyUI completion
"""

import threading
import time
import queue
from enum import Enum
from typing import Optional, Callable, Any
from dataclasses import dataclass
from contextlib import contextmanager

from event_logging.event_logger import log_json_entry, LogType
from config.config import MOOD_SNAPSHOT_FOLDER


class ResourcePriority(Enum):
    """Priority levels for resource requests."""
    HIGH = 1    # ComfyUI image generation (blocks all Ollama)
    MEDIUM = 2  # Drawing prompt generation (can queue regular captions)
    LOW = 3     # Regular captioning and reflection (can be queued/delayed)


@dataclass
class ResourceRequest:
    """Represents a request for GPU/AI resources."""
    priority: ResourcePriority
    operation_name: str
    callback: Callable[[], Any]
    timeout: float = 30.0
    created_at: float = None
    request_id: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if self.request_id is None:
            self.request_id = str(time.time())
    
    def __lt__(self, other):
        """Support priority queue ordering by priority value, then by creation time."""
        if not isinstance(other, ResourceRequest):
            return NotImplemented
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.created_at < other.created_at


class ResourceManager:
    """
    Singleton resource manager for coordinating GPU/AI resource access.
    
    Prevents resource contention between Ollama and ComfyUI by:
    - Blocking Ollama calls when ComfyUI is active
    - Queuing lower priority requests
    - Using ImageMonitor callbacks to release resources
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
            
        self._initialized = True
        self._resource_lock = threading.RLock()
        self._request_queue = queue.PriorityQueue()
        self._comfyui_active = False
        self._current_high_priority = None
        self._queue_worker_thread = None
        self._running = False
        self._completion_callbacks = []
        
        # Configuration
        self.enabled = True
        self.max_queue_size = 50
        self.queue_timeout = 60.0
        
        self._start_queue_worker()
        
    def configure(self, enabled: bool = True, max_queue_size: int = 50, queue_timeout: float = 60.0):
        """Configure resource manager settings."""
        with self._resource_lock:
            self.enabled = enabled
            self.max_queue_size = max_queue_size
            self.queue_timeout = queue_timeout
            
    def _start_queue_worker(self):
        """Start the background thread that processes queued requests."""
        if self._queue_worker_thread is None or not self._queue_worker_thread.is_alive():
            self._running = True
            self._queue_worker_thread = threading.Thread(target=self._queue_worker, daemon=True)
            self._queue_worker_thread.start()
            
    def _queue_worker(self):
        """Background worker that processes queued resource requests."""
        while self._running:
            try:
                # Get next request with timeout
                try:
                    request = self._request_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                    
                # Check if request has timed out
                if time.time() - request.created_at > request.timeout:
                    self._log_resource_event("request_timeout", {
                        "operation": request.operation_name,
                        "priority": request.priority.name,
                        "wait_time": time.time() - request.created_at
                    })
                    continue
                
                # Wait for resources to become available
                while self._should_block_request(request.priority):
                    time.sleep(0.1)
                    if time.time() - request.created_at > request.timeout:
                        break
                        
                if time.time() - request.created_at > request.timeout:
                    continue
                
                # Execute the request
                try:
                    with self._resource_lock:
                        if request.priority == ResourcePriority.HIGH:
                            self._current_high_priority = request.operation_name
                            
                    self._log_resource_event("request_start", {
                        "operation": request.operation_name,
                        "priority": request.priority.name,
                        "wait_time": time.time() - request.created_at
                    })
                    
                    # Execute the callback
                    result = request.callback()
                    
                    self._log_resource_event("request_complete", {
                        "operation": request.operation_name,
                        "priority": request.priority.name
                    })
                    
                finally:
                    with self._resource_lock:
                        if request.priority == ResourcePriority.HIGH:
                            self._current_high_priority = None
                            
            except Exception as e:
                self._log_resource_event("queue_worker_error", {"error": str(e)})
                time.sleep(1.0)
                
    def _should_block_request(self, priority: ResourcePriority) -> bool:
        """Check if a request should be blocked due to resource contention."""
        with self._resource_lock:
            # HIGH priority requests are never blocked
            if priority == ResourcePriority.HIGH:
                return False
                
            # Block if ComfyUI is active or high priority operation is running
            if self._comfyui_active or self._current_high_priority:
                return True
                
            return False
    
    @contextmanager
    def request_resource(self, priority: ResourcePriority, operation_name: str, timeout: float = 30.0):
        """
        Context manager for requesting GPU/AI resources.
        
        Args:
            priority: Priority level of the request
            operation_name: Human-readable name for logging
            timeout: Maximum time to wait for resources
            
        Yields:
            None when resources are available
            
        Raises:
            TimeoutError: If resources aren't available within timeout
        """
        if not self.enabled:
            yield
            return
            
        start_time = time.time()
        
        # For HIGH priority requests, block immediately if needed
        if priority == ResourcePriority.HIGH:
            while self._should_block_request(priority):
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Timeout waiting for resources for {operation_name}")
                time.sleep(0.1)
                
            with self._resource_lock:
                self._current_high_priority = operation_name
                
            try:
                self._log_resource_event("resource_acquired", {
                    "operation": operation_name,
                    "priority": priority.name,
                    "wait_time": time.time() - start_time
                })
                yield
            finally:
                with self._resource_lock:
                    self._current_high_priority = None
                    
        else:
            # For MEDIUM/LOW priority, use immediate check or queue
            if not self._should_block_request(priority):
                # Resources available immediately
                self._log_resource_event("resource_acquired", {
                    "operation": operation_name,
                    "priority": priority.name,
                    "wait_time": 0
                })
                yield
            else:
                # Queue the request
                if self._request_queue.qsize() >= self.max_queue_size:
                    raise queue.Full(f"Resource queue is full, cannot queue {operation_name}")
                    
                # Create a synchronization event
                completion_event = threading.Event()
                result_container = {"result": None, "exception": None}
                
                def callback():
                    try:
                        completion_event.set()
                        return None
                    except Exception as e:
                        result_container["exception"] = e
                        completion_event.set()
                        raise
                
                request = ResourceRequest(
                    priority=priority,
                    operation_name=operation_name,
                    callback=callback,
                    timeout=timeout
                )
                
                # Queue the request
                self._request_queue.put(request)
                
                self._log_resource_event("request_queued", {
                    "operation": operation_name,
                    "priority": priority.name,
                    "queue_size": self._request_queue.qsize()
                })
                
                # Wait for completion
                if completion_event.wait(timeout):
                    if result_container["exception"]:
                        raise result_container["exception"]
                    yield
                else:
                    raise TimeoutError(f"Timeout waiting for queued {operation_name}")
    
    def set_comfyui_active(self, active: bool, operation_name: str = "comfyui_generation"):
        """
        Set ComfyUI active state to block Ollama calls.
        
        Args:
            active: True when ComfyUI starts generating, False when complete
            operation_name: Name of the ComfyUI operation for logging
        """
        with self._resource_lock:
            if active and not self._comfyui_active:
                self._comfyui_active = True
                self._log_resource_event("comfyui_start", {"operation": operation_name})
            elif not active and self._comfyui_active:
                self._comfyui_active = False
                self._log_resource_event("comfyui_complete", {"operation": operation_name})
                
                # Notify completion callbacks
                for callback in self._completion_callbacks:
                    try:
                        callback()
                    except Exception as e:
                        self._log_resource_event("callback_error", {"error": str(e)})
    
    def add_completion_callback(self, callback: Callable[[], None]):
        """Add a callback to be called when ComfyUI completes."""
        self._completion_callbacks.append(callback)
    
    def is_comfyui_active(self) -> bool:
        """Check if ComfyUI is currently active."""
        with self._resource_lock:
            return self._comfyui_active
    
    def get_queue_size(self) -> int:
        """Get current size of the resource request queue."""
        return self._request_queue.qsize()
    
    def get_status(self) -> dict:
        """Get current resource manager status for debugging."""
        with self._resource_lock:
            return {
                "enabled": self.enabled,
                "comfyui_active": self._comfyui_active,
                "current_high_priority": self._current_high_priority,
                "queue_size": self._request_queue.qsize(),
                "running": self._running
            }
    
    def _log_resource_event(self, event_type: str, data: dict):
        """Log resource management events."""
        log_data = {
            "event": f"resource_{event_type}",
            "timestamp": time.time(),
            **data
        }
        
        log_json_entry(
            LogType.SYSTEM,
            log_data,
            MOOD_SNAPSHOT_FOLDER,
            auto_print=True,
            print_message=f"🔧 Resource {event_type}: {data}"
        )
    
    def shutdown(self):
        """Clean shutdown of the resource manager."""
        self._running = False
        if self._queue_worker_thread and self._queue_worker_thread.is_alive():
            self._queue_worker_thread.join(timeout=5.0)


# Global singleton instance
resource_manager = ResourceManager()