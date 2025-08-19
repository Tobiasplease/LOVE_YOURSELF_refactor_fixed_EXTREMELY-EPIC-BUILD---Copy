#!/usr/bin/env python3
"""Test session time awareness."""

import sys
import time
from captioner.memory import MemoryMixin

# Mock captioner for testing
class MockCaptioner(MemoryMixin):
    def __init__(self, session_gap=None):
        super().__init__()
        self._captioner_ref = self
        self.last_session_gap = session_gap

def test_temporal_lines():
    print("Testing temporal context generation...")
    
    # Test fresh start (no gap)
    print("\n=== Fresh Start (No Previous Session) ===")
    captioner = MockCaptioner()
    lines = captioner.temporal_prompt_lines()
    print(f"Temporal lines: {lines}")
    
    # Test with short gap (5 minutes)
    print("\n=== After 5 Minutes Sleep ===")
    captioner = MockCaptioner(session_gap=300)  # 5 minutes
    lines = captioner.temporal_prompt_lines()
    print(f"Temporal lines: {lines}")
    
    # Test with longer gap (2 hours)
    print("\n=== After 2 Hours Sleep ===")
    captioner = MockCaptioner(session_gap=7200)  # 2 hours
    lines = captioner.temporal_prompt_lines()
    print(f"Temporal lines: {lines}")
    
    # Test with very long gap (1 day)
    print("\n=== After 1 Day Sleep ===")
    captioner = MockCaptioner(session_gap=86400)  # 1 day
    lines = captioner.temporal_prompt_lines()
    print(f"Temporal lines: {lines}")
    
    # Wait a bit and test session time tracking
    print("\n=== Waiting 3 seconds for session time test ===")
    time.sleep(3)
    lines = captioner.temporal_prompt_lines()
    print(f"Temporal lines after 3 seconds: {lines}")

if __name__ == "__main__":
    test_temporal_lines()
