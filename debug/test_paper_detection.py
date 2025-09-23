#!/usr/bin/env python3
"""
Paper Detection Testing and Calibration Tool

Provides manual testing and calibration capabilities for the paper detection safety system.
Use this to set up reference images, test detection accuracy, and debug the system.
"""

import sys
import os
import time
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safety.paper_detection import paper_detector, get_paper_detection_status, check_paper_before_drawing, capture_paper_present_reference, capture_paper_absent_reference
from config.config import ENABLE_PAPER_DETECTION, PAPER_CHECK_METHOD, PAPER_DETECTION_CONFIDENCE_THRESHOLD


def init_hardware():
    """Initialize camera and servo hardware for testing."""
    try:
        # Try to import and initialize hardware
        from camera.camera_opencv import Camera
        from servo_control.servo_control import ServoController
        from captioner.captioner import Captioner

        print("🔧 Initializing hardware...")

        camera = Camera()
        servos = ServoController()
        captioner = Captioner()

        print("✅ Hardware initialized successfully")
        return camera, servos, captioner

    except Exception as e:
        print(f"❌ Hardware initialization failed: {e}")
        print("Make sure camera and servos are connected and accessible")
        return None, None, None


def test_capture_reference():
    """Test capturing a reference image for paper detection."""
    print("\n📄 CAPTURE REFERENCE IMAGE TEST")
    print("=" * 50)

    camera, servos, captioner = init_hardware()
    if not camera or not servos:
        return False

    print("📋 Instructions:")
    print("1. Place paper properly in the drawing area")
    print("2. Ensure paper is flat, clean, and ready for drawing")
    print("3. Press Enter when ready to capture reference image")
    input("Press Enter to continue...")

    success = capture_paper_reference(camera, servos)

    if success:
        print("✅ Reference image captured successfully!")
        print(f"📁 Saved to: {paper_detector.reference_image_path}")
    else:
        print("❌ Failed to capture reference image")

    return success


def test_paper_detection():
    """Test paper detection with current setup."""
    print("\n🔍 PAPER DETECTION TEST")
    print("=" * 50)

    camera, servos, captioner = init_hardware()
    if not camera or not servos:
        return False

    print("📋 Instructions:")
    print("1. You can test with paper present or absent")
    print("2. Try different paper positions, lighting conditions")
    print("3. Press Enter when ready to test detection")
    input("Press Enter to continue...")

    print("🔍 Running paper detection...")
    result = paper_detector.check_paper_present(camera, servos, captioner)

    print(f"\n📊 DETECTION RESULT:")
    print(f"   Paper Present: {'✅ YES' if result.paper_present else '❌ NO'}")
    print(f"   Confidence: {result.confidence:.2f}")
    print(f"   Method: {result.method_used}")
    print(f"   Check Image: {result.check_image_path}")

    if result.error_message:
        print(f"   Error: {result.error_message}")

    print(f"\n🤖 LLM Response:")
    print(f"   {result.llm_response}")

    return result.paper_present


def test_multiple_detections():
    """Run multiple detection tests for accuracy analysis."""
    print("\n📊 MULTIPLE DETECTION TEST")
    print("=" * 50)

    camera, servos, captioner = init_hardware()
    if not camera or not servos:
        return

    try:
        num_tests = int(input("How many tests to run (default 5): ") or "5")
    except ValueError:
        num_tests = 5

    print(f"🔍 Running {num_tests} detection tests...")
    print("📋 You can move paper, change lighting, etc. between tests")

    results = []

    for i in range(num_tests):
        print(f"\n--- Test {i+1}/{num_tests} ---")
        input("Press Enter for next test...")

        result = paper_detector.check_paper_present(camera, servos, captioner)
        results.append(result)

        print(f"Result: {'✅' if result.paper_present else '❌'} (confidence: {result.confidence:.2f})")

    # Analysis
    print(f"\n📈 TEST ANALYSIS:")
    paper_detected = sum(1 for r in results if r.paper_present)
    avg_confidence = sum(r.confidence for r in results) / len(results)

    print(f"   Total Tests: {num_tests}")
    print(f"   Paper Detected: {paper_detected}")
    print(f"   Detection Rate: {paper_detected/num_tests*100:.1f}%")
    print(f"   Average Confidence: {avg_confidence:.2f}")
    print(f"   Confidence Threshold: {PAPER_DETECTION_CONFIDENCE_THRESHOLD}")


def test_safety_integration():
    """Test the safety integration with drawing system."""
    print("\n🛡️ SAFETY INTEGRATION TEST")
    print("=" * 50)

    camera, servos, captioner = init_hardware()
    if not camera or not servos:
        return

    print("📋 This tests the same function used by the drawing system")
    print("⚠️ Make sure paper detection is enabled in config")

    if not ENABLE_PAPER_DETECTION:
        print("❌ Paper detection is DISABLED in config")
        print("   Set ENABLE_PAPER_DETECTION = True to test")
        return

    input("Press Enter to test safety check...")

    print("🛡️ Running safety check...")
    safe_to_draw = check_paper_before_drawing(camera, servos, captioner)

    if safe_to_draw:
        print("✅ SAFE TO DRAW - Paper detected")
        print("   Drawing would proceed normally")
    else:
        print("🛑 DRAWING BLOCKED - No paper detected")
        print("   Drawing execution would be aborted for safety")


def show_system_status():
    """Show current paper detection system status."""
    print("\n📋 PAPER DETECTION SYSTEM STATUS")
    print("=" * 50)

    status = get_paper_detection_status()

    print(f"🔧 Configuration:")
    print(f"   Enabled: {'✅' if status['enabled'] else '❌'}")
    print(f"   Method: {status['method']}")
    print(f"   Confidence Threshold: {status['confidence_threshold']}")
    print(f"   Reference Image: {'✅' if status['reference_image_exists'] else '❌'}")
    print(f"   Reference Path: {status['reference_image_path']}")

    print(f"\n📊 Detection History:")
    print(f"   Recent Checks: {status['recent_checks']}")

    if status['last_check']:
        last = status['last_check']
        print(f"   Last Result: {'✅' if last['paper_present'] else '❌'}")
        print(f"   Last Confidence: {last['confidence']:.2f}")
        print(f"   Last Method: {last['method_used']}")


def interactive_menu():
    """Interactive menu for testing paper detection."""
    while True:
        print("\n🔧 PAPER DETECTION TEST MENU")
        print("=" * 40)
        print("1. Show system status")
        print("2. Capture reference image")
        print("3. Test paper detection (single)")
        print("4. Test paper detection (multiple)")
        print("5. Test safety integration")
        print("6. Exit")

        try:
            choice = input("\nSelect option (1-6): ").strip()

            if choice == "1":
                show_system_status()
            elif choice == "2":
                test_capture_reference()
            elif choice == "3":
                test_paper_detection()
            elif choice == "4":
                test_multiple_detections()
            elif choice == "5":
                test_safety_integration()
            elif choice == "6":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please select 1-6.")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main entry point for paper detection testing."""
    parser = argparse.ArgumentParser(description="Paper Detection Testing Tool")
    parser.add_argument("--status", action="store_true", help="Show system status and exit")
    parser.add_argument("--capture", action="store_true", help="Capture reference image and exit")
    parser.add_argument("--test", action="store_true", help="Run single detection test and exit")
    parser.add_argument("--multiple", type=int, metavar="N", help="Run N detection tests and exit")

    args = parser.parse_args()

    print("📄 Paper Detection Testing Tool")
    print("=" * 50)

    if args.status:
        show_system_status()
    elif args.capture:
        test_capture_reference()
    elif args.test:
        test_paper_detection()
    elif args.multiple:
        # Set up for multiple tests
        camera, servos, captioner = init_hardware()
        if camera and servos:
            results = []
            for i in range(args.multiple):
                print(f"Test {i+1}/{args.multiple}...")
                result = paper_detector.check_paper_present(camera, servos, captioner)
                results.append(result)
                print(f"   {'✅' if result.paper_present else '❌'} (confidence: {result.confidence:.2f})")
                time.sleep(1)  # Brief pause between tests

            # Summary
            detected = sum(1 for r in results if r.paper_present)
            avg_conf = sum(r.confidence for r in results) / len(results)
            print(f"\nSummary: {detected}/{args.multiple} detected, avg confidence: {avg_conf:.2f}")
    else:
        interactive_menu()


if __name__ == "__main__":
    main()