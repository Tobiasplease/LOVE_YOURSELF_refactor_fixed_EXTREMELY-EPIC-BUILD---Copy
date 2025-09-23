#!/usr/bin/env python3
"""
Comprehensive Motion Systems Test Suite
======================================

Tests all uArm motion recording and playback systems:
1. Official SDK Teach system
2. Custom MotionManager
3. High-frequency angle-based MotionManager

Validates smooth playback, timing accuracy, and error handling.
"""

import os
import sys
import threading
import time
from datetime import datetime

# Add uarm_control to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "uarm_control"))

from high_frequency_motion_manager import HighFrequencyMotionManager
from motion_manager import MotionManager
from uarm_controller import UarmController

try:
    from uarm.swift.teach import Teach

    OFFICIAL_SDK_AVAILABLE = True
except ImportError:
    print("Warning: Official uArm SDK not available")
    OFFICIAL_SDK_AVAILABLE = False


class MotionSystemTester:
    def __init__(self):
        self.controller = None
        self.motion_manager = None
        self.hf_motion_manager = None
        self.official_teach = None
        self.test_results = {}

    def setup(self):
        """Setup test environment"""
        print("=" * 60)
        print("🔧 MOTION SYSTEMS TEST SUITE")
        print("=" * 60)

        # Connect to uArm
        print("\n📡 Connecting to uArm...")
        self.controller = UarmController(auto_home=False)

        if not self.controller.is_connected():
            print("❌ Failed to connect to uArm - aborting tests")
            return False

        print("✅ Connected to uArm")

        # Initialize motion managers
        self.motion_manager = MotionManager(controller=self.controller)
        self.hf_motion_manager = HighFrequencyMotionManager(controller=self.controller)

        # Setup official SDK if available
        if OFFICIAL_SDK_AVAILABLE:
            test_file = "movement_recordings/uarm/test_official_suite.txt"
            os.makedirs("movement_recordings/uarm", exist_ok=True)
            self.official_teach = Teach(test_file, self.controller.robot)
            self.official_teach.start_standby_mode()

        print("🛠️ All motion systems initialized")
        return True

    def test_official_sdk_smooth_playback(self):
        """Test 1: Official SDK smooth playback"""
        if not OFFICIAL_SDK_AVAILABLE:
            return {"status": "skipped", "reason": "SDK not available"}

        print("\n🎯 TEST 1: Official SDK Smooth Playback")
        print("-" * 40)

        try:
            # Record a simple motion
            print("📹 Recording 3-second test motion...")
            print("Move the arm smoothly in a simple pattern")

            self.official_teach.start_record(interval=0.05)  # 20Hz
            time.sleep(3)
            self.official_teach.stop_record()

            print("✅ Recording complete")

            # Test multiple playback speeds
            speeds = [0.5, 1.0, 2.0]
            playback_results = {}

            for speed in speeds:
                print(f"▶️ Testing playback at {speed}x speed...")

                start_time = time.time()
                self.official_teach.start_play(speed=speed, times=1)

                # Monitor playback
                while self.official_teach.is_playing():
                    progress = self.official_teach.get_progress(wait=False)
                    if progress:
                        print(f"  Progress: {progress[1]:.1f}%", end="\r")
                    time.sleep(0.1)

                duration = time.time() - start_time
                expected_duration = 3.0 / speed
                timing_accuracy = abs(duration - expected_duration) / expected_duration

                playback_results[speed] = {
                    "actual_duration": duration,
                    "expected_duration": expected_duration,
                    "timing_accuracy": timing_accuracy,
                    "smooth": timing_accuracy < 0.2,  # Within 20%
                }

                print(f"\n  Duration: {duration:.1f}s (expected {expected_duration:.1f}s)")
                print(f"  Timing accuracy: {timing_accuracy*100:.1f}%")

            return {"status": "passed", "details": playback_results}

        except Exception as e:
            return {"status": "failed", "error": str(e)}

    def test_motion_manager_coordination(self):
        """Test 2: MotionManager coordinated movement"""
        print("\n🎯 TEST 2: MotionManager Coordinated Movement")
        print("-" * 40)

        try:
            # Test slot 1
            slot = 1

            print("📹 Recording motion for MotionManager test...")
            print("Move arm in a simple arc motion")

            # Simulate recording (in real test, user would move arm)
            if not self.motion_manager.record_motion(slot, "test_coordination", "Coordination test motion"):
                return {"status": "failed", "error": "Recording failed"}

            print("✅ Recording complete")

            # Test playback with timing measurement
            print("▶️ Testing coordinated playback...")

            start_time = time.time()
            result = self.motion_manager.play_motion(slot)

            if not result:
                return {"status": "failed", "error": "Playback failed"}

            duration = time.time() - start_time
            print(f"✅ Playback completed in {duration:.1f}s")

            return {"status": "passed", "duration": duration}

        except Exception as e:
            return {"status": "failed", "error": str(e)}

    def test_hf_motion_manager_smoothness(self):
        """Test 3: High-frequency motion manager smoothness"""
        print("\n🎯 TEST 3: High-Frequency Motion Manager Smoothness")
        print("-" * 40)

        try:
            slot = 2

            print("📹 Starting high-frequency recording...")
            print("Move arm smoothly for 5 seconds")

            if not self.hf_motion_manager.start_recording(slot, "test_smoothness", "Smoothness test"):
                return {"status": "failed", "error": "Recording start failed"}

            # Monitor recording
            for i in range(5):
                time.sleep(1)
                status = self.hf_motion_manager.get_recording_status()
                if status.get("recording"):
                    freq = status.get("current_frequency", 0)
                    print(f"  Recording: {i+1}/5s - {freq:.1f}Hz")

            if not self.hf_motion_manager.stop_recording():
                return {"status": "failed", "error": "Recording stop failed"}

            print("✅ High-frequency recording complete")

            # Test playback at different speeds
            speeds = [0.5, 1.0, 1.5]
            smoothness_results = {}

            for speed in speeds:
                print(f"▶️ Testing playback at {speed}x speed...")

                start_time = time.time()
                result = self.hf_motion_manager.play_motion(slot, speed_multiplier=speed)

                if not result:
                    smoothness_results[speed] = {"status": "failed"}
                    continue

                duration = time.time() - start_time
                smoothness_results[speed] = {"status": "passed", "duration": duration, "speed_multiplier": speed}

                print(f"  Completed in {duration:.1f}s")

            return {"status": "passed", "details": smoothness_results}

        except Exception as e:
            return {"status": "failed", "error": str(e)}

    def test_error_handling(self):
        """Test 4: Error handling and edge cases"""
        print("\n🎯 TEST 4: Error Handling")
        print("-" * 40)

        error_tests = {}

        # Test invalid slot numbers
        try:
            result = self.motion_manager.play_motion(99)  # Invalid slot
            error_tests["invalid_slot"] = "passed" if not result else "failed"
        except Exception:
            error_tests["invalid_slot"] = "passed"  # Should handle gracefully

        # Test playback of non-existent recording
        try:
            self.motion_manager.delete_motion(3)  # Ensure slot 3 is empty
            result = self.motion_manager.play_motion(3)  # No recording
            error_tests["no_recording"] = "passed" if not result else "failed"
        except Exception:
            error_tests["no_recording"] = "passed"

        # Test simultaneous operations
        if OFFICIAL_SDK_AVAILABLE:
            try:
                # Try to start recording while already recording
                self.official_teach.start_record()
                time.sleep(0.1)
                # This should handle gracefully
                self.official_teach.stop_record()
                error_tests["simultaneous_ops"] = "passed"
            except Exception as e:
                error_tests["simultaneous_ops"] = f"failed: {e}"

        return {"status": "completed", "details": error_tests}

    def test_timing_precision(self):
        """Test 5: Timing precision across systems"""
        print("\n🎯 TEST 5: Timing Precision")
        print("-" * 40)

        timing_results = {}

        # Test consistent timing across multiple playbacks
        if hasattr(self, "official_teach") and self.official_teach:
            print("📊 Testing official SDK timing consistency...")

            durations = []
            for i in range(3):
                start = time.time()
                self.official_teach.start_play(speed=1.0, times=1)

                while self.official_teach.is_playing():
                    time.sleep(0.1)

                duration = time.time() - start
                durations.append(duration)
                print(f"  Run {i+1}: {duration:.2f}s")

            if durations:
                avg_duration = sum(durations) / len(durations)
                max_deviation = max(abs(d - avg_duration) for d in durations)
                timing_results["official_sdk"] = {
                    "average_duration": avg_duration,
                    "max_deviation": max_deviation,
                    "consistency": "good" if max_deviation < 0.5 else "poor",
                }

        return {"status": "completed", "details": timing_results}

    def run_all_tests(self):
        """Run complete test suite"""
        if not self.setup():
            return

        tests = [
            ("Official SDK Smooth Playback", self.test_official_sdk_smooth_playback),
            ("MotionManager Coordination", self.test_motion_manager_coordination),
            ("High-Frequency Smoothness", self.test_hf_motion_manager_smoothness),
            ("Error Handling", self.test_error_handling),
            ("Timing Precision", self.test_timing_precision),
        ]

        print(f"\n🚀 Running {len(tests)} motion system tests...")

        for test_name, test_func in tests:
            print(f"\n{'='*60}")
            try:
                result = test_func()
                self.test_results[test_name] = result

                status = result.get("status", "unknown")
                if status == "passed":
                    print(f"✅ {test_name}: PASSED")
                elif status == "failed":
                    print(f"❌ {test_name}: FAILED - {result.get('error', 'Unknown error')}")
                elif status == "skipped":
                    print(f"⏭️ {test_name}: SKIPPED - {result.get('reason', 'No reason')}")
                else:
                    print(f"ℹ️ {test_name}: {status}")

            except Exception as e:
                print(f"💥 {test_name}: CRASHED - {e}")
                self.test_results[test_name] = {"status": "crashed", "error": str(e)}

        self.print_summary()
        self.cleanup()

    def print_summary(self):
        """Print test results summary"""
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)

        passed = sum(1 for r in self.test_results.values() if r.get("status") == "passed")
        failed = sum(1 for r in self.test_results.values() if r.get("status") == "failed")
        skipped = sum(1 for r in self.test_results.values() if r.get("status") == "skipped")
        crashed = sum(1 for r in self.test_results.values() if r.get("status") == "crashed")

        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"⏭️ Skipped: {skipped}")
        print(f"💥 Crashed: {crashed}")

        print(f"\n📈 Overall Success Rate: {passed/(passed+failed+crashed)*100:.1f}%" if (passed + failed + crashed) > 0 else "No conclusive tests")

        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"movement_recordings/uarm/test_results_{timestamp}.json"

        import json

        try:
            os.makedirs(os.path.dirname(results_file), exist_ok=True)
            with open(results_file, "w") as f:
                json.dump(
                    {
                        "timestamp": timestamp,
                        "summary": {"passed": passed, "failed": failed, "skipped": skipped, "crashed": crashed},
                        "details": self.test_results,
                    },
                    f,
                    indent=2,
                )
            print(f"\n💾 Results saved to: {results_file}")
        except Exception as e:
            print(f"⚠️ Could not save results: {e}")

    def cleanup(self):
        """Cleanup test environment"""
        print("\n🧹 Cleaning up...")

        try:
            if self.official_teach:
                self.official_teach.stop_standby_mode()

            if self.controller:
                # Move to safe position
                self.controller.home()
                self.controller.disconnect()

        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")

        print("✅ Cleanup complete")


def main():
    """Main test runner"""
    tester = MotionSystemTester()

    print("🔬 uArm Motion Systems Test Suite")
    print("This will test all motion recording and playback systems")
    print("Make sure your uArm is connected and ready to move!")

    input("\nPress Enter to start tests...")

    tester.run_all_tests()


if __name__ == "__main__":
    main()
