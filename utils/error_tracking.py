"""
Comprehensive error tracking and silent failure detection for the LOVE_YOURSELF system.
"""

import functools
import threading
import time
import traceback
from typing import Any, Callable, Dict

from event_logging.event_logger import log_json_entry
from event_logging.log_type import LogType


class SilentFailureTracker:
    """Tracks components that stop working silently."""

    def __init__(self):
        self.component_heartbeats: Dict[str, float] = {}
        self.component_errors: Dict[str, int] = {}
        self.expected_intervals: Dict[str, float] = {
            "captioner": 30.0,  # Should caption every 30s
            "mood_engine": 5.0,  # Should update mood every 5s
            "drawing_controller": 60.0,  # Should check drawing every 60s
            "hand_controller": 1.0,  # Should send data every 1s
            "object_detection": 1.0,  # Should detect every 1s
        }

        # Start monitoring thread
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_components, daemon=True)
        self._monitor_thread.start()

    def heartbeat(self, component_name: str):
        """Record that a component is still alive."""
        self.component_heartbeats[component_name] = time.time()

    def log_component_error(self, component_name: str, error: Exception, context: str = ""):
        """Log an error from a specific component."""
        self.component_errors[component_name] = self.component_errors.get(component_name, 0) + 1

        error_data = {
            "component": component_name,
            "error_type": type(error).__name__,
            "message": str(error),
            "context": context,
            "total_errors": self.component_errors[component_name],
            "traceback": traceback.format_exc(),
        }

        log_json_entry(LogType.ERROR, error_data, print_message=f"[❌] {component_name} error: {str(error)}")

    def _monitor_components(self):
        """Monitor components for silent failures."""
        while self._monitoring:
            current_time = time.time()

            for component, expected_interval in self.expected_intervals.items():
                last_heartbeat = self.component_heartbeats.get(component, 0)
                time_since_heartbeat = current_time - last_heartbeat

                # Component has gone silent
                if time_since_heartbeat > expected_interval * 3:  # 3x the expected interval
                    if last_heartbeat > 0:  # Only alert if we've seen it before
                        silence_data = {
                            "component": component,
                            "expected_interval_s": expected_interval,
                            "time_since_last_heartbeat_s": time_since_heartbeat,
                            "last_heartbeat_timestamp": last_heartbeat,
                        }

                        log_json_entry(
                            LogType.ERROR,
                            {"message": f"Component '{component}' has gone silent", "component": component, "silent_failure_data": silence_data},
                            print_message=f"[⚠️] Component '{component}' has gone silent",
                        )

                        # Reset to avoid spam
                        self.component_heartbeats[component] = current_time

            time.sleep(10)  # Check every 10 seconds

    def shutdown(self):
        """Shutdown the monitoring thread."""
        self._monitoring = False


# Global tracker instance
_failure_tracker = SilentFailureTracker()


def get_failure_tracker() -> SilentFailureTracker:
    """Get the global failure tracker."""
    return _failure_tracker


def track_component_health(component_name: str):
    """Decorator to track component health and catch silent failures."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                # Record heartbeat before execution
                _failure_tracker.heartbeat(component_name)

                result = func(*args, **kwargs)

                # Record successful completion
                return result

            except Exception as e:
                # Log the error with full context
                context = f"Function: {func.__name__}, Args: {len(args)}, Kwargs: {list(kwargs.keys())}"
                _failure_tracker.log_component_error(component_name, e, context)

                # Re-raise so calling code can handle appropriately
                raise

        return wrapper

    return decorator


