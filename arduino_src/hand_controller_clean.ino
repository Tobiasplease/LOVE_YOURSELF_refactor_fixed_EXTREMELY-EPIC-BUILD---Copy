#include <Servo.h>

const int NUM_SERVOS = 6;  // 4 hand servos + 2 left arm servos
const int pins[NUM_SERVOS] = {8, 9, 10, 11, 4, 5};  // Hand servos on 8-11, left arm on 4-5
const bool isMirrored[NUM_SERVOS] = {false, false, true, true, false, false};

Servo servos[NUM_SERVOS];
int currentAngles[NUM_SERVOS];
int targetAngles[NUM_SERVOS];
unsigned long lastUpdate[NUM_SERVOS];
int speeds[NUM_SERVOS];

String inputBuffer = "";

void setup() {
  Serial.begin(9600);
  Serial.println("DEVICE_ID:HAND_CONTROLLER");
  Serial.println("Hand Controller Ready - Direct Servo Control");

  // Initialize all servos
  for (int i = 0; i < NUM_SERVOS; i++) {
    currentAngles[i] = 90;
    targetAngles[i] = 90;
    lastUpdate[i] = 0;
    speeds[i] = 15; // Default speed for hand servos (15ms)
  }

  // Set faster speeds for left arm servos (will be controlled by Python)
  speeds[4] = 5;  // Left arm servo 1
  speeds[5] = 5;  // Left arm servo 2

  Serial.println("Ready for commands (0-180 degree range)");
}

void loop() {
  unsigned long now = millis();

  // Handle serial commands
  handleSerialInput();

  // Update all servos toward targets
  updateServoPositions(now);
}

void handleSerialInput() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      if (inputBuffer.length() > 0) {
        processCommand(inputBuffer);
        inputBuffer = "";
      }
    } else {
      inputBuffer += c;
    }
  }
}

void processCommand(String command) {
  command.trim();

  if (command.startsWith("HAND,")) {
    // Parse hand command: "HAND,pos0,pos1,pos2,pos3"
    int positions[4];
    int startIndex = 5; // Skip "HAND,"

    for (int i = 0; i < 4; i++) {
      int commaIndex = command.indexOf(',', startIndex);
      if (commaIndex == -1 && i == 3) {
        positions[i] = command.substring(startIndex).toInt();
      } else if (commaIndex != -1) {
        positions[i] = command.substring(startIndex, commaIndex).toInt();
        startIndex = commaIndex + 1;
      } else {
        Serial.println("Invalid HAND command format");
        return;
      }
    }

    // Set target angles for hand servos (0-3) and attach if needed
    for (int i = 0; i < 4; i++) {
      targetAngles[i] = constrain(positions[i], 0, 180);

      // Attach servo if not already attached
      if (!servos[i].attached()) {
        servos[i].attach(pins[i]);
        Serial.print("Attached hand servo ");
        Serial.print(i);
        Serial.print(" on pin ");
        Serial.println(pins[i]);
      }
    }
  }
  else if (command.startsWith("SERVO,")) {
    // Parse servo command: "SERVO,pin,angle"
    parseServoCommand(command);
  }
  else if (command.startsWith("MOOD,")) {
    // Parse mood command - kept for compatibility
    Serial.print("Mood received: ");
    Serial.println(command.substring(5));
  }
}

void parseServoCommand(String command) {
  // Parse: "SERVO,pin,angle"
  int firstComma = command.indexOf(',');
  int secondComma = command.indexOf(',', firstComma + 1);

  if (firstComma == -1 || secondComma == -1) {
    Serial.println("Invalid SERVO command format");
    return;
  }

  int pin = command.substring(firstComma + 1, secondComma).toInt();
  int angle = command.substring(secondComma + 1).toInt();

  // Find servo index for this pin
  int servoIndex = -1;
  for (int i = 0; i < NUM_SERVOS; i++) {
    if (pins[i] == pin) {
      servoIndex = i;
      break;
    }
  }

  if (servoIndex == -1) {
    Serial.print("Invalid pin: ");
    Serial.println(pin);
    return;
  }

  // Attach servo if not already attached
  if (!servos[servoIndex].attached()) {
    servos[servoIndex].attach(pin);
  }

  // Set target angle
  targetAngles[servoIndex] = constrain(angle, 0, 180);
  Serial.print("Servo ");
  Serial.print(servoIndex);
  Serial.print(" (pin ");
  Serial.print(pin);
  Serial.print(") -> ");
  Serial.println(angle);
}

void updateServoPositions(unsigned long now) {
  // Update all servos toward targets
  for (int i = 0; i < NUM_SERVOS; i++) {
    if (now - lastUpdate[i] >= speeds[i]) {
      // Move toward target with smooth steps
      int distance = abs(targetAngles[i] - currentAngles[i]);
      int stepSize = 1;

      if (distance > 0) {
        if (currentAngles[i] < targetAngles[i]) {
          currentAngles[i] = min(currentAngles[i] + stepSize, targetAngles[i]);
        } else {
          currentAngles[i] = max(currentAngles[i] - stepSize, targetAngles[i]);
        }

        // Only write to servo if it's attached
        if (servos[i].attached()) {
          writeMapped(i, currentAngles[i]);
        }
      }

      lastUpdate[i] = now;
    }
  }
}

void writeMapped(int i, int angle) {
  int mappedAngle = isMirrored[i] ? (180 - angle) : angle;
  servos[i].write(mappedAngle);
}