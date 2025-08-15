#!/usr/bin/env python3
"""
Simple camera preview with reactivity visualization
"""
import cv2
import numpy as np
from reactivity.camera_reactive import CameraReactivityEngine

def run_camera_preview():
    """Run a camera preview showing reactivity data"""
    print("🎥 Starting camera preview with simplified pause visualization...")
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
        
        # Create overlay with single activity bar showing pause proximity
        overlay = frame.copy()
        height, width = frame.shape[:2]
        
        # Main activity bar - shows how close we are to triggering a pause (positioned at bottom center)
        bar_width = 400
        bar_height = 30
        bar_x = (width - bar_width) // 2  # Center horizontally
        bar_y = height - 80  # Move to bottom with margin
        
        # Background (black)
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (0, 0, 0), -1)
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
        
        # Activity level (blue bar)
        activity_width = int(metrics['progress_to_pause'] * bar_width / 100)
        activity_color = (255, 0, 0) if metrics['progress_to_pause'] >= 100 else (255, 255, 0) if metrics['progress_to_pause'] >= 80 else (0, 255, 0)
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + activity_width, bar_y + bar_height), activity_color, -1)
        
        # Pause threshold line (red)
        threshold_x = bar_x + bar_width  # Always at 100% since we show progress to pause
        cv2.line(overlay, (threshold_x, bar_y), (threshold_x, bar_y + bar_height), (0, 0, 255), 3)
        
        # Text overlay - position above the bar at bottom
        activity_text = f"Activity: {metrics['activity_level']:.3f} | Progress to Pause: {metrics['progress_to_pause']:.1f}%"
        cv2.putText(overlay, activity_text, (bar_x, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Pause indicator
        if metrics.get('is_paused', False):
            cv2.rectangle(overlay, (width-150, 10), (width-10, 60), (0, 0, 255), -1)
            cv2.putText(overlay, "PAUSED", (width-140, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(overlay, f"{metrics['pause_remaining']:.1f}s", (width-140, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Cooldown indicator
        elif metrics.get('cooldown_remaining', 0) > 0:
            cv2.rectangle(overlay, (width-150, 10), (width-10, 40), (100, 100, 100), -1)
            cv2.putText(overlay, f"Cooldown: {metrics['cooldown_remaining']:.1f}s", (width-145, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
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
