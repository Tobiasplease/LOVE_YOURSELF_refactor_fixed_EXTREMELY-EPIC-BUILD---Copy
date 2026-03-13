#!/usr/bin/env python3
"""Detect actual camera capabilities and optimal settings."""

import cv2
import sys

def test_camera_capabilities():
    print("Opening camera...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        sys.exit(1)

    print("\n=== CURRENT SETTINGS ===")
    properties = {
        'Width': cv2.CAP_PROP_FRAME_WIDTH,
        'Height': cv2.CAP_PROP_FRAME_HEIGHT,
        'FPS': cv2.CAP_PROP_FPS,
        'Brightness': cv2.CAP_PROP_BRIGHTNESS,
        'Contrast': cv2.CAP_PROP_CONTRAST,
        'Saturation': cv2.CAP_PROP_SATURATION,
        'Hue': cv2.CAP_PROP_HUE,
        'Gain': cv2.CAP_PROP_GAIN,
        'Exposure': cv2.CAP_PROP_EXPOSURE,
        'Sharpness': cv2.CAP_PROP_SHARPNESS,
        'Auto Exposure': cv2.CAP_PROP_AUTO_EXPOSURE,
        'Auto Focus': cv2.CAP_PROP_AUTOFOCUS,
    }

    for name, prop in properties.items():
        value = cap.get(prop)
        print(f"{name:20s}: {value}")

    print("\n=== TESTING RESOLUTIONS ===")
    test_resolutions = [
        (2560, 1440, "2K"),
        (1920, 1080, "1080p"),
        (1280, 720, "720p"),
        (800, 600, "SVGA"),
        (640, 480, "VGA"),
        (320, 240, "QVGA"),
    ]

    for width, height, name in test_resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if actual_w == width and actual_h == height:
            print(f"✓ {name:8s} ({width}x{height}): SUPPORTED")
        else:
            print(f"✗ {name:8s} ({width}x{height}): Got {actual_w}x{actual_h}")

    print("\n=== TESTING ADJUSTABLE PARAMETERS ===")

    # Reset to default resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Test brightness
    original_brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 75)
    new_brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
    brightness_works = abs(new_brightness - 75) < 5
    cap.set(cv2.CAP_PROP_BRIGHTNESS, original_brightness)
    print(f"Brightness:  {'✓ Adjustable' if brightness_works else '✗ Fixed/Not supported'}")

    # Test contrast
    original_contrast = cap.get(cv2.CAP_PROP_CONTRAST)
    cap.set(cv2.CAP_PROP_CONTRAST, 75)
    new_contrast = cap.get(cv2.CAP_PROP_CONTRAST)
    contrast_works = abs(new_contrast - 75) < 5
    cap.set(cv2.CAP_PROP_CONTRAST, original_contrast)
    print(f"Contrast:    {'✓ Adjustable' if contrast_works else '✗ Fixed/Not supported'}")

    # Test saturation
    original_saturation = cap.get(cv2.CAP_PROP_SATURATION)
    cap.set(cv2.CAP_PROP_SATURATION, 75)
    new_saturation = cap.get(cv2.CAP_PROP_SATURATION)
    saturation_works = abs(new_saturation - 75) < 5
    cap.set(cv2.CAP_PROP_SATURATION, original_saturation)
    print(f"Saturation:  {'✓ Adjustable' if saturation_works else '✗ Fixed/Not supported'}")

    # Test sharpness
    original_sharpness = cap.get(cv2.CAP_PROP_SHARPNESS)
    cap.set(cv2.CAP_PROP_SHARPNESS, 75)
    new_sharpness = cap.get(cv2.CAP_PROP_SHARPNESS)
    sharpness_works = abs(new_sharpness - 75) < 5
    cap.set(cv2.CAP_PROP_SHARPNESS, original_sharpness)
    print(f"Sharpness:   {'✓ Adjustable' if sharpness_works else '✗ Fixed/Not supported'}")

    # Test exposure
    original_exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
    cap.set(cv2.CAP_PROP_EXPOSURE, -5)
    new_exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
    exposure_works = abs(new_exposure - (-5)) < 1
    cap.set(cv2.CAP_PROP_EXPOSURE, original_exposure)
    print(f"Exposure:    {'✓ Adjustable' if exposure_works else '✗ Fixed/Auto only'}")

    print("\n=== RECOMMENDATIONS ===")

    # Find best supported resolution
    best_res = None
    for width, height, name in test_resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_w == width and actual_h == height:
            best_res = (width, height, name)
            break

    if best_res:
        print(f"1. Set resolution to: {best_res[2]} ({best_res[0]}x{best_res[1]})")
        print(f"   CAMERA_WIDTH = {best_res[0]}")
        print(f"   CAMERA_HEIGHT = {best_res[1]}")

    if brightness_works:
        print(f"2. Brightness is adjustable (currently {original_brightness})")
        print(f"   Try: CAMERA_BRIGHTNESS = 128 (or adjust 0-255)")

    if contrast_works:
        print(f"3. Contrast is adjustable (currently {original_contrast})")
        print(f"   Try: CAMERA_CONTRAST = 128 (or adjust 0-255)")

    if sharpness_works:
        print(f"4. Sharpness is adjustable (currently {original_sharpness})")
        print(f"   Try: CAMERA_SHARPNESS = 150 (higher for more sharpness)")

    if not exposure_works:
        print(f"5. Exposure appears to be auto-only")
        print(f"   Keep: CAMERA_EXPOSURE = -1")

    cap.release()
    print("\nDone!")

if __name__ == "__main__":
    test_camera_capabilities()
