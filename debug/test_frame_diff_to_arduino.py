import cv2
import serial
import time

# CONFIG
SERIAL_PORT = 'COM4'  # Update to your working port
BAUDRATE = 9600
CAMERA_INDEX = 0

# Initialize serial
ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
print(f"Opened serial port {SERIAL_PORT}")

# Initialize camera
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

ret, prev_frame = cap.read()
if not ret:
    print("Could not read from camera.")
    exit(1)
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Frame diff
    diff = cv2.absdiff(prev_gray, gray)
    diff_score = diff.mean()  # Range: 0-255
    prev_gray = gray.copy()
    # Map 0-50 frame diff to 0-255 PWM
    max_diff = 50.0
    diff_score = max(0, min(max_diff, diff_score))
    # Smoothing (exponential moving average)
    if 'smoothed_pwm' not in locals():
        smoothed_pwm = 0
    pwm_value = int((diff_score / max_diff) * 255)
    alpha = 0.2  # Smoothing factor
    smoothed_pwm = int(alpha * pwm_value + (1 - alpha) * smoothed_pwm)
    # Threshold
    if diff_score < 1:
        smoothed_pwm = 0
    print(f"Frame diff: {diff_score:.1f} -> PWM: {smoothed_pwm}")
    try:
        ser.write(f"{smoothed_pwm}\n".encode())
    except Exception as e:
        print(f"Serial write error: {e}")
    time.sleep(0.05)  # 20 FPS
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
ser.close()
print("Test finished.")
