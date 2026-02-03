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
      int angle = constrain(line.substring(4).toInt(), 0, 180);
      if (!panAttached) {
        panServo.attach(9);
        panAttached = true;
      }
      // DIRECT control - no smoothing since Python already does it
      panServo.write(angle);
    }

    else if (line.startsWith("TILT:")) {
      // Constrain TILT to safe vertical range (45-150 degrees) - expanded for lowered mount
      int angle = constrain(line.substring(5).toInt(), 45, 150);
      if (!tiltAttached) {
        tiltServo.attach(10);
        tiltAttached = true;
      }
      // DIRECT control - no smoothing since Python already does it
      tiltServo.write(angle);
    }

    else if (line.startsWith("LUNG:")) {
      String value = line.substring(5);
      if (value == "hold" || value == "slow") {
        lungMode = value;
      } else {
        int angle = value.toInt();
        if (!lungAttached) {
          lungServo.attach(6);
          lungAttached = true;
        }
        lungServo.write(constrain(angle, 0, 180));
        lungMode = "hold";
      }
    }
  }

  updateLung();
  delay(5);  // Very fast loop for responsive control
}

void updateLung() {
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