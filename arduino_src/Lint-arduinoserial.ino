
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
  Serial.println("READY");
}

void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    Serial.println("Received: " + line);

    if (line.startsWith("PAN:")) {
      targetPan = constrain(line.substring(4).toInt(), 0, 180);
      if (!panAttached) {
        panServo.attach(9);
        panAttached = true;
      }
    }

    else if (line.startsWith("TILT:")) {
      targetTilt = constrain(line.substring(5).toInt(), 0, 180);
      if (!tiltAttached) {
        tiltServo.attach(10);
        tiltAttached = true;
      }
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

  updatePanTilt();
  updateLung();
  delay(30);  // run loop ~30Hz
}

void updatePanTilt() {
  if (panAttached) {
    if (abs(currentPan - targetPan) > 2) {
      currentPan += (targetPan > currentPan) ? 2 : -2;
      panServo.write(currentPan);
    }
  }

  if (tiltAttached) {
    if (abs(currentTilt - targetTilt) > 2) {
      currentTilt += (targetTilt > currentTilt) ? 2 : -2;
      tiltServo.write(currentTilt);
    }
  }
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
