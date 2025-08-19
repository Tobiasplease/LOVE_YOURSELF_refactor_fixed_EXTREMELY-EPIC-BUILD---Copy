import serial

class LightbulbController:
    def flicker(self, duration=0.2, brightness=255):
        """Flash bulb to full brightness for a short duration."""
        self.set_pwm(brightness)
        import time
        time.sleep(duration)
        self.set_pwm(0)

    def ease(self, duration=0.5, brightness=255, steps=20):
        """Ease bulb to brightness and back over duration."""
        import time
        interval = duration / (2 * steps)
        # Fade in
        for i in range(steps):
            val = int((i + 1) * brightness / steps)
            self.set_pwm(val)
            time.sleep(interval)
        # Fade out
        for i in range(steps):
            val = int((steps - i - 1) * brightness / steps)
            self.set_pwm(val)
            time.sleep(interval)
    def __init__(self, port, baudrate=9600):
        self.ser = serial.Serial(port, baudrate, timeout=1)

    def set_pwm(self, value):
        pwm = max(0, min(255, int(value)))
        try:
            self.ser.write(f"{pwm}\n".encode())
        except Exception as e:
            print(f"[LightbulbController] Serial write error: {e}")
