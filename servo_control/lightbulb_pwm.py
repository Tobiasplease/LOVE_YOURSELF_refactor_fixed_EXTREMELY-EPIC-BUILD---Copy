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
        self.fluctuating = False

    def set_base_brightness(self, brightness):
        """Set base brightness from frame difference (Arduino handles fluctuation)."""
        brightness = max(18, min(255, int(brightness)))
        try:
            command = f"BASE:{brightness}\n"
            self.ser.write(command.encode())
        except Exception as e:
            print(f"[LightbulbController] Base brightness error: {e}")

    def update_mood(self, speed, randomness):
        """Update mood-based fluctuation parameters."""
        speed = max(0.1, min(2.0, float(speed)))
        randomness = max(0.0, min(1.0, float(randomness)))
        try:
            command = f"MOOD:{speed}:{randomness}\n"
            self.ser.write(command.encode())
        except Exception as e:
            print(f"[LightbulbController] Mood update error: {e}")

    def caption_boost(self, duration=600):
        """Trigger caption brightness boost for specified duration in ms."""
        duration = max(100, min(2000, int(duration)))
        try:
            command = f"BOOST:{duration}\n"
            self.ser.write(command.encode())
        except Exception as e:
            print(f"[LightbulbController] Caption boost error: {e}")

    def set_pwm(self, value):
        """Legacy method - just sets base brightness now."""
        self.set_base_brightness(value)
