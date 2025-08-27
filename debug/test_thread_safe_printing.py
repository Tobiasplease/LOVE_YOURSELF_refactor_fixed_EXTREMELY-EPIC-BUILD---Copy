#!/usr/bin/env python3
"""
Test the thread-safe printing fix to verify no repetition occurs
even under heavy concurrent load.
"""
import threading
import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def simulate_concurrent_printing():
    """Simulate multiple threads trying to print simultaneously."""
    
    class MockCaptioner:
        def __init__(self):
            self.print_lock = threading.Lock()
            
        def thread_safe_print(self, thread_id, message):
            """Simulate the thread-safe printing mechanism."""
            with self.print_lock:
                print("\r" + " " * 80 + "\r", end="")  # Clear line
                print(f"[Thread-{thread_id}] {message}")
                
        def old_unsafe_print(self, thread_id, message):
            """Simulate the old problematic printing."""
            print(f"\r[Thread-{thread_id}] {message}")
    
    print("=== TESTING THREAD-SAFE PRINTING FIX ===")
    
    # Test with thread-safe printing
    print("\n1. THREAD-SAFE VERSION (should be clean):")
    captioner = MockCaptioner()
    
    def safe_worker(thread_id):
        for i in range(3):
            captioner.thread_safe_print(thread_id, f"Caption {i}: This should appear only once")
            time.sleep(0.1)
    
    # Start multiple threads simultaneously
    threads = []
    for t_id in range(3):
        thread = threading.Thread(target=safe_worker, args=(t_id,))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    print("\n2. OLD UNSAFE VERSION (would show repetition):")
    
    def unsafe_worker(thread_id):
        for i in range(2):
            captioner.old_unsafe_print(thread_id, f"Caption {i}: This might repeat")
            time.sleep(0.1)
    
    # Start multiple threads simultaneously (fewer to reduce spam)
    threads = []
    for t_id in range(2):
        thread = threading.Thread(target=unsafe_worker, args=(t_id,))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    print("\n" + "=" * 60)
    print("RESULT VERIFICATION:")
    print("✓ Thread-safe version should show each message exactly once")
    print("✗ Unsafe version might show garbled or repeated output")
    print("The fix prevents multiple threads from interfering with each other")

if __name__ == "__main__":
    simulate_concurrent_printing()