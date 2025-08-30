#!/usr/bin/env python3
"""
Lightbulb Controller Crash Fix Verification Test
===============================================
Test the robust lightbulb controller under stress conditions
that previously caused system crashes.
"""

import sys
import os
import time
import threading
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from servo_control.lightbulb_controller_robust import ThreadSafeLightbulbWrapper, RobustLightbulbController


def stress_test_lightbulb(port: str, duration: int = 30):
    """Stress test the lightbulb controller with concurrent access."""
    print(f"🔥 Starting {duration}s stress test on {port}")
    
    controller = ThreadSafeLightbulbWrapper(port, debug=True)
    stop_event = threading.Event()
    errors = []
    
    def brightness_spammer():
        """Simulate main thread brightness updates."""
        while not stop_event.is_set():
            try:
                brightness = random.randint(0, 255)
                controller.set_frame_diff_brightness(brightness)
                time.sleep(random.uniform(0.01, 0.1))
            except Exception as e:
                errors.append(f"Brightness thread error: {e}")
    
    def flash_spammer():
        """Simulate mood thread caption flashes."""
        while not stop_event.is_set():
            try:
                controller.caption_flash()
                time.sleep(random.uniform(0.5, 2.0))
            except Exception as e:
                errors.append(f"Flash thread error: {e}")
    
    def connection_disruptor():
        """Simulate USB disconnections and port issues."""
        while not stop_event.is_set():
            try:
                if hasattr(controller.controller, 'is_connected'):
                    controller.controller.is_connected = False
                time.sleep(random.uniform(3.0, 8.0))
            except Exception as e:
                errors.append(f"Disruptor error: {e}")
    
    threads = [
        threading.Thread(target=brightness_spammer, daemon=True, name="BrightnessSpam"),
        threading.Thread(target=flash_spammer, daemon=True, name="FlashSpam"),
        threading.Thread(target=connection_disruptor, daemon=True, name="ConnectionDisrupt")
    ]
    
    for thread in threads:
        thread.start()
    
    try:
        time.sleep(duration)
        print("⏰ Test duration completed")
    except KeyboardInterrupt:
        print("⚠️ Test interrupted by user")
    finally:
        stop_event.set()
        for thread in threads:
            thread.join(timeout=1.0)
    
    controller.close()
    
    if errors:
        print(f"❌ Test completed with {len(errors)} errors:")
        for error in errors[-5:]:
            print(f"   {error}")
    else:
        print("✅ Test completed successfully with no errors!")
    
    return len(errors) == 0


def test_disconnection_recovery():
    """Test that the controller handles port disconnection gracefully."""
    print("🔌 Testing disconnection recovery...")
    
    fake_port = "/dev/nonexistent"
    controller = ThreadSafeLightbulbWrapper(fake_port, debug=True)
    
    try:
        controller.set_frame_diff_brightness(128)
        controller.caption_flash()
        controller.caption_flash()
        controller.set_frame_diff_brightness(200)
        print("✅ Disconnection test passed - no crashes!")
        return True
    except Exception as e:
        print(f"❌ Disconnection test failed: {e}")
        return False
    finally:
        controller.close()


def compare_old_vs_new_controller():
    """Compare crash safety of old vs new controller."""
    print("📊 Comparing old vs new controller safety...")
    
    fake_port = "/dev/nonexistent"
    
    print("Testing old SimpleLightbulbController...")
    try:
        from servo_control.lightbulb_controller_simple import SimpleLightbulbController
        old_controller = SimpleLightbulbController(fake_port, debug=False)
        old_controller.set_frame_diff_brightness(128)
        old_controller.caption_flash()
        print("❓ Old controller didn't crash (unexpected)")
        old_safe = True
    except Exception as e:
        print(f"❌ Old controller crashed: {e}")
        old_safe = False
    
    print("Testing new ThreadSafeLightbulbWrapper...")
    try:
        new_controller = ThreadSafeLightbulbWrapper(fake_port, debug=False)
        new_controller.set_frame_diff_brightness(128)
        new_controller.caption_flash()
        new_controller.close()
        print("✅ New controller handled gracefully")
        new_safe = True
    except Exception as e:
        print(f"❌ New controller crashed: {e}")
        new_safe = False
    
    if new_safe and not old_safe:
        print("🎉 New controller is more robust!")
    elif new_safe and old_safe:
        print("🤔 Both controllers handled the test")
    else:
        print("⚠️ Issues remain")
    
    return new_safe


def main():
    """Run comprehensive lightbulb controller crash fix verification."""
    print("🔬 LIGHTBULB CONTROLLER CRASH FIX VERIFICATION")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    if test_disconnection_recovery():
        tests_passed += 1
    
    if compare_old_vs_new_controller():
        tests_passed += 1
    
    available_ports = [f"/dev/ttyUSB{i}" for i in range(3) if os.path.exists(f"/dev/ttyUSB{i}")]
    if available_ports:
        print(f"Found USB ports: {available_ports}")
        test_port = available_ports[0]
        if stress_test_lightbulb(test_port, duration=10):
            tests_passed += 1
    else:
        print("⚠️ No USB ports available for stress test - simulating...")
        if stress_test_lightbulb("/dev/nonexistent", duration=5):
            tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"📈 RESULTS: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 LIGHTBULB CRASH FIX VERIFIED SUCCESSFULLY!")
        print("✅ The system should no longer crash from lightbulb issues")
        return True
    else:
        print("⚠️ Some tests failed - issues may remain")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)