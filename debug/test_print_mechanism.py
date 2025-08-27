#!/usr/bin/env python3
"""
Direct test of the print mechanism that was causing repetition.
This simulates the exact animation/print logic to verify the fix.
"""
import time
import threading
from datetime import datetime

def test_old_problematic_method():
    """Test the old problematic print method (before fix)."""
    print("\n=== TESTING OLD METHOD (would cause repetition) ===")
    
    # Simulate loading animation
    loading_stop = threading.Event()
    
    def loading_animation():
        frames = [" ", ".", "..", "..."]
        idx = 0
        while not loading_stop.is_set():
            print(f"\r{frames[idx % 4]}", end="", flush=True)
            idx += 1
            time.sleep(0.1)
    
    loading_thread = threading.Thread(target=loading_animation, daemon=True)
    loading_thread.start()
    
    # Let animation run briefly
    time.sleep(0.5)
    
    # Stop animation and print caption (OLD WAY)
    loading_stop.set()
    loading_thread.join(timeout=0.5)
    
    # OLD PROBLEMATIC METHOD: Just \r with caption
    timestamp = datetime.now().strftime("%H:%M:%S")
    caption = "This is a test caption that might get repeated"
    formatted_caption = f"[{timestamp}] {caption}"
    
    print(f"\r{formatted_caption}")  # OLD WAY - might not clear properly
    print()
    
    print("^^^ Old method completed")

def test_new_fixed_method():
    """Test the new fixed print method."""
    print("\n=== TESTING NEW METHOD (should prevent repetition) ===")
    
    # Simulate loading animation
    loading_stop = threading.Event()
    
    def loading_animation():
        frames = [" ", ".", "..", "..."]
        idx = 0
        while not loading_stop.is_set():
            print(f"\r{frames[idx % 4]}", end="", flush=True)
            idx += 1
            time.sleep(0.1)
    
    loading_thread = threading.Thread(target=loading_animation, daemon=True)
    loading_thread.start()
    
    # Let animation run briefly
    time.sleep(0.5)
    
    # Stop animation and print caption (NEW WAY)
    loading_stop.set()
    loading_thread.join(timeout=0.5)
    
    # NEW FIXED METHOD: Properly clear line before printing
    timestamp = datetime.now().strftime("%H:%M:%S")
    caption = "This is a test caption that should appear only once"
    formatted_caption = f"[{timestamp}] {caption}"
    
    print("\r" + " " * 80 + "\r", end="")  # Clear any animation remnants
    print(formatted_caption)  # Print without \r to avoid buffer issues
    
    print("^^^ New method completed")

def test_multiple_rapid_prints():
    """Test multiple rapid prints to simulate the repetition scenario."""
    print("\n=== TESTING MULTIPLE RAPID PRINTS ===")
    
    for i in range(3):
        print(f"\nRapid print test {i+1}:")
        
        # Simulate the fixed print mechanism
        loading_stop = threading.Event()
        
        def quick_animation():
            for _ in range(3):
                if loading_stop.is_set():
                    break
                print("\r...", end="", flush=True)
                time.sleep(0.05)
        
        anim_thread = threading.Thread(target=quick_animation, daemon=True)
        anim_thread.start()
        
        time.sleep(0.2)
        loading_stop.set()
        anim_thread.join(timeout=0.1)
        
        # Use the fixed print method
        timestamp = datetime.now().strftime("%H:%M:%S")
        caption = f"Caption #{i+1} - should appear exactly once"
        formatted_caption = f"[{timestamp}] {caption}"
        
        print("\r" + " " * 80 + "\r", end="")  # Clear line
        print(formatted_caption)  # Print cleanly
        
        time.sleep(0.1)

if __name__ == "__main__":
    print("Testing caption print mechanism fix...")
    print("Look for any duplicate or repeated output:")
    
    test_old_problematic_method()
    test_new_fixed_method()
    test_multiple_rapid_prints()
    
    print("\n" + "=" * 60)
    print("VERIFICATION:")
    print("- Each caption should appear exactly ONCE")
    print("- No garbled or repeated text")
    print("- Clean line clearing between animation and caption")
    print("=" * 60)