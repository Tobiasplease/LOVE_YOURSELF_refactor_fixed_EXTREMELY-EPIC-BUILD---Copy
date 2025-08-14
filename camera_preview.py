#!/usr/bin/env python3
"""
Simple camera preview with reactivity visualization
"""
import cv2
import numpy as np
from reactivity.camera_reactive import CameraReactivityEngine

def run_camera_preview():
    """Run a camera preview showing reactivity data"""
    print("🎥 Starting camera preview with reactivity visualization...")
    print("Press 'q' to quit")
    
    # Initialize camera and reactivity engine
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    reactivity_engine = CameraReactivityEngine()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame for reactivity
        metrics = reactivity_engine.process_frame(frame)
        
        # Create overlay with reactivity info
        overlay = frame.copy()
        height, width = frame.shape[:2]
        
        # Draw reactivity metrics
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.rectangle(overlay, (10, 10), (400, 120), (255, 255, 255), 2)
        
        # Activity bars
        activity_bar_length = int(metrics['activity_level'] * 300)
        speed_bar_length = int(metrics['speed_multiplier'] * 100)
        chaos_bar_length = int(metrics['chaos_multiplier'] * 100)
        
        cv2.rectangle(overlay, (20, 25), (20 + activity_bar_length, 35), (0, 255, 0), -1)
        cv2.rectangle(overlay, (20, 45), (20 + speed_bar_length, 55), (255, 0, 0), -1)
        cv2.rectangle(overlay, (20, 65), (20 + chaos_bar_length, 75), (0, 0, 255), -1)
        
        # Text
        cv2.putText(overlay, f"Activity: {metrics['activity_level']:.3f}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(overlay, f"Speed: {metrics['speed_multiplier']:.2f}x", (150, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(overlay, f"Chaos: {metrics['chaos_multiplier']:.2f}x", (250, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Pause indicator
        if metrics.get('paused', False):
            cv2.rectangle(overlay, (width-100, 10), (width-10, 50), (0, 0, 255), -1)
            cv2.putText(overlay, "PAUSED", (width-95, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show the frame
        cv2.imshow('Camera Reactivity Preview', overlay)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("📹 Camera preview closed")

if __name__ == "__main__":
    run_camera_preview()
