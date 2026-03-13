#!/usr/bin/env python3
import sys
import cv2
sys.path.insert(0, '/home/impostor/LOVE_YOURSELF_refactor_fixed_EXTREMELY-EPIC-BUILD---Copy')

from safety.paper_detection import PaperDetector

image_path = sys.argv[1] if len(sys.argv) > 1 else "event_log/df99f284-images/paper_check_1759511573.jpg"

detector = PaperDetector()

# Create cropped version
cropped_path = detector._crop_detection_area(image_path)
print(f"Cropped image: {cropped_path}")

# Show crop dimensions
img = cv2.imread(cropped_path)
if img is not None:
    print(f"Crop size: {img.shape[1]}x{img.shape[0]} pixels")

# Test brightness detection
result = detector._check_brightness_based(cropped_path)

print(f"\nResult:")
print(f"  Paper present: {result.paper_present}")
print(f"  Confidence: {result.confidence:.2f}")
print(f"  Method: {result.method_used}")
print(f"  Details: {result.llm_response}")
