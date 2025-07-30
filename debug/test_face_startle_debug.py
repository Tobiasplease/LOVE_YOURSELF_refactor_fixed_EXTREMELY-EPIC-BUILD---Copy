#!/usr/bin/env python3
"""
Debug script to test face detection startle reaction
"""

import cv2
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perception.face_detection import FaceDetector
from servo_control.hand_expression import HandExpressionController

def main():
    print("FACE DETECTION STARTLE DEBUG")
    print("="*50)
    print("This will show debug info about face detection and startle reactions")
    print("Position your face in front of the camera to test")
    print("Press 'q' to quit")
    print()
    
    # Initialize components
    face_detector = FaceDetector()
    
    try:
        hand_controller = HandExpressionController()
        print("✅ Hand controller connected")
    except Exception as e:
        print(f"❌ Hand controller failed: {e}")
        print("Continuing with face detection only...")
        hand_controller = None
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    print("✅ Camera opened")
    print("\nStarting face detection debug...")
    print("Watch for startle reactions when a face is first detected!")
    print()
    
    face_detected_last_frame = False
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect faces
        faces = face_detector.detect_faces(frame)
        face_detected_this_frame = len(faces) > 0
        
        # Check for face detection transition (no face -> face detected)
        newly_detected = face_detected_this_frame and not face_detected_last_frame
        
        # Debug output every 30 frames (about once per second)
        if frame_count % 30 == 0:
            status = "FACE PRESENT" if face_detected_this_frame else "NO FACE"
            print(f"Frame {frame_count}: {status} | Faces: {len(faces)}")
        
        # If face is newly detected, trigger startle
        if newly_detected:
            print(f"🎯 NEWLY DETECTED FACE! Triggering startle reaction...")
            if hand_controller:
                try:
                    hand_controller.trigger_startle()
                    print("✅ Startle command sent to hand")
                except Exception as e:
                    print(f"❌ Startle failed: {e}")
            else:
                print("⚠️  No hand controller - would trigger startle here")
        
        # Update face detection state
        face_detected_last_frame = face_detected_this_frame
        
        # Show video with face rectangles
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "FACE DETECTED", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add status text to frame
        status_text = f"Faces: {len(faces)} | Frame: {frame_count}"
        cv2.putText(frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if newly_detected:
            cv2.putText(frame, "STARTLE TRIGGERED!", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        cv2.imshow('Face Detection Debug', frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    if hand_controller:
        hand_controller.close()
    
    print("\nDebug session complete!")

if __name__ == "__main__":
    main()
