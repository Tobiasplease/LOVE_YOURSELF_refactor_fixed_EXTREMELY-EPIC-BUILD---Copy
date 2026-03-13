#!/usr/bin/env python3
import cv2
import numpy as np

img = cv2.imread("event_log/df99f284-images/paper_check_1759511573_detection_area.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(f"Image shape: {gray.shape}")
print(f"Mean brightness: {gray.mean():.1f}")
print(f"Min brightness: {gray.min()}")
print(f"Max brightness: {gray.max()}")
print(f"Median brightness: {np.median(gray):.1f}")

for threshold in [100, 120, 150, 180, 200]:
    white_pct = ((gray > threshold).sum() / gray.size) * 100
    print(f"Pixels > {threshold}: {white_pct:.1f}%")
