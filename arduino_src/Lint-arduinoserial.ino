
#include <Servo.h>

Servo panServo;
Servo tiltServo;
Servo lungServo;

bool panAttached = false;
bool tiltAttached = false;
bool lungAttached = false;

String lungMode = "hold";
unsigned long lastLungUpdate = 0;
int lungPos = 90;
int lungDir = 1;

int currentPan = 90;
int targetPan = 90;

int currentTilt = 90;
int targetTilt = 90;

int currentLung = 90;

void setup() {
  Serial.begin(9600);
  Serial.println("DEVICE_ID:SERVO_CONTROLLER");
  Serial.println("READY");
}

void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    Serial.println("Received: " + line);

    if (line.startsWith("PAN:")) {
      // Expanded natural head movement: ±45° from center (45-135°)
      targetPan = constrain(line.substring(4).toInt(), 45, 135);
      if (!panAttached) {
        panServo.attach(9);
        panAttached = true;
      }
    }

    else if (line.startsWith("TILT:")) {
      // Expanded natural head movement: ±40° from center (50-130°)
      targetTilt = constrain(line.substring(5).toInt(), 50, 130);
      if (!tiltAttached) {
        tiltServo.attach(10);
        tiltAttached = true;
      }
    }

    else if (line.startsWith("LUNG:")) {
      String value = line.substring(5);
      if (value == "hold") {
        lungMode = value;  // Stop autonomous breathing, hold position
      } else if (value == "slow") {
        lungMode = value;  // Enable autonomous breathing (fallback mode)
      } else {
        int angle = value.toInt();
        if (!lungAttached) {
          lungServo.attach(6);
          lungAttached = true;
        }
        lungServo.write(constrain(angle, 0, 180));
        lungMode = "python";  // Set to python-controlled mode
      }
    }
  }

  updatePanTilt();
  updateLung();
  delay(10);  // run loop ~100Hz for smoother movement
}

void updatePanTilt() {
  // Direct control - Python already handles smoothing perfectly
  // This eliminates double-smoothing and choppy movement
  
  if (panAttached && currentPan != targetPan) {
    currentPan = targetPan;
    panServo.write(currentPan);
  }

  if (tiltAttached && currentTilt != targetTilt) {
    currentTilt = targetTilt;
    tiltServo.write(currentTilt);
  }
}

void updateLung() {
  // Only run autonomous breathing when explicitly in "slow" mode (fallback)
  // Python-controlled mode ("python") and hold mode ("hold") don't use this
  if (lungMode != "slow" || !lungAttached) return;

  unsigned long now = millis();
  int interval = 30;

  if (now - lastLungUpdate > interval) {
    lungPos += lungDir;
    if (lungPos > 120 || lungPos < 60) {
      lungDir *= -1;
    }
    lungServo.write(lungPos);
    lastLungUpdate = now;
  }
}
