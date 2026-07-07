#!/usr/bin/env python3
"""
debug/test_llama_server.py
--------------------------
Test script for the llama-server migration. Tests three modes:

1. Health check — is the server running?
2. Single-image query — parity with current Ollama behavior
3. Multi-frame video query — new temporal perception capability

Usage:
    # Test with a live camera frame:
    python debug/test_llama_server.py

    # Test with a specific image:
    python debug/test_llama_server.py --image /path/to/image.jpg

    # Test video mode with camera:
    python debug/test_llama_server.py --video --seconds 5
"""

import argparse
import sys
import time

sys.path.insert(0, ".")


def test_health():
    """Test 1: Is llama-server responding?"""
    from utils.llama_server import is_server_running, LLAMA_SERVER_URL
    print(f"\n--- Test 1: Health check ({LLAMA_SERVER_URL}) ---")
    if is_server_running():
        print("OK - llama-server is running")
        return True
    else:
        print("FAIL - llama-server not responding")
        print("Start it with:")
        print("  ./llama.cpp/build/bin/llama-server \\")
        print("    -m <model.gguf> --mmproj <mmproj.gguf> \\")
        print("    --jinja --ctx-size 65536 --port 8080")
        return False


def test_single_image(image_path: str = None):
    """Test 2: Single-image caption (parity with Ollama)."""
    from utils.llama_server import query_llama_server
    print("\n--- Test 2: Single-image caption ---")

    if image_path:
        print(f"Using image: {image_path}")
    else:
        import cv2
        print("Capturing frame from camera...")
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("FAIL - could not capture camera frame")
            return False
        image_path = "/tmp/test_llama_frame.jpg"
        cv2.imwrite(image_path, frame)
        print(f"Saved test frame to {image_path}")

    system_prompt = (
        "You are a drawing machine attached to a table. "
        "You see through a camera that moves. "
        "One sentence, first person, present tense."
    )

    start = time.time()
    result = query_llama_server(
        prompt="What do you see?",
        image=image_path,
        system_prompt=system_prompt,
        options={"temperature": 0.8, "num_predict": 60},
        timeout=30,
    )
    elapsed = time.time() - start

    if result.startswith("[WARNING]"):
        print(f"FAIL - {result}")
        return False

    print(f"Response ({elapsed:.1f}s): {result}")
    return True


def test_video(seconds: float = 5.0):
    """Test 3: Multi-frame video caption (new capability)."""
    from captioner.frame_buffer import FrameBuffer
    from utils.llama_server import query_llama_server_video
    import cv2

    print(f"\n--- Test 3: Video caption ({seconds}s capture) ---")

    buf = FrameBuffer(target_fps=2.0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("FAIL - could not open camera")
        return False

    print(f"Buffering {seconds}s of video at 2fps...")
    start = time.time()
    while time.time() - start < seconds:
        ret, frame = cap.read()
        if ret:
            buf.push(frame)
        time.sleep(0.03)  # ~30fps capture rate, buffer handles throttling
    cap.release()

    frames = buf.get_recent(seconds=seconds, max_frames=10)
    print(f"Collected {len(frames)} frames ({buf.seconds_buffered:.1f}s buffered)")

    if len(frames) < 2:
        print("FAIL - not enough frames captured")
        return False

    system_prompt = (
        "You are a drawing machine attached to a table. "
        "You see through a camera. These frames span the last few seconds. "
        "Describe what changed. One sentence, first person."
    )

    start = time.time()
    result = query_llama_server_video(
        prompt="What happened in these frames?",
        frames=frames,
        fps=2.0,
        system_prompt=system_prompt,
        timeout=60,
    )
    elapsed = time.time() - start

    if result.startswith("[WARNING]"):
        print(f"FAIL - {result}")
        return False

    print(f"Response ({elapsed:.1f}s): {result}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test llama-server integration")
    parser.add_argument("--image", type=str, help="Path to test image")
    parser.add_argument("--video", action="store_true", help="Test multi-frame video mode")
    parser.add_argument("--seconds", type=float, default=5.0, help="Seconds of video to capture")
    args = parser.parse_args()

    results = {}

    # Always test health first
    results["health"] = test_health()
    if not results["health"]:
        print("\nServer not running. Fix that first.")
        sys.exit(1)

    # Single image test
    results["single_image"] = test_single_image(args.image)

    # Video test (if requested)
    if args.video:
        results["video"] = test_video(args.seconds)

    # Summary
    print("\n--- Results ---")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
