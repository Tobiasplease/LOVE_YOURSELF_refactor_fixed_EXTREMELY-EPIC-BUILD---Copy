#include <Servo.h>

const int NUM_SERVOS = 6;
const int pins[NUM_SERVOS] = {8, 9, 10, 11, 4, 5};
const bool isMirrored[NUM_SERVOS] = {false, false, true, true, false, false};

Servo servos[NUM_SERVOS];
int currentAngles[NUM_SERVOS];
int targetAngles[NUM_SERVOS];
unsigned long lastUpdate[NUM_SERVOS];
int speeds[NUM_SERVOS];

// SIMPLE left arm movement - NO PAUSES, NO COMPLEX LOGIC
const int LEFT_ARM_START = 4;
const int LEFT_ARM_COUNT = 2;
const int LEFT_ARM_CENTER = 90;
const int LEFT_ARM_MAX_RANGE = 30;  // ±15 degrees from center
const int LEFT_ARM_MIN = LEFT_ARM_CENTER - (LEFT_ARM_MAX_RANGE / 2);  // 75
const int LEFT_ARM_MAX = LEFT_ARM_CENTER + (LEFT_ARM_MAX_RANGE / 2);  // 105

bool leftArmMovementEnabled = false;
bool leftArmServosAttached = false;

// SIMPLE movement variables
int leftArmCurrentPos[LEFT_ARM_COUNT];
int leftArmTargetPos[LEFT_ARM_COUNT];
int leftArmDirection[LEFT_ARM_COUNT];
unsigned long leftArmLastMove[LEFT_ARM_COUNT];
const int LEFT_ARM_MOVE_INTERVAL = 200;  // Move every 200ms (slow)
const int LEFT_ARM_STEP_SIZE = 1;        // 1 degree per step

unsigned long lastStatusPrint = 0;

String inputBuffer = "";

void setup() {
  Serial.begin(9600);
  Serial.println("DEVICE_ID:HAND_CONTROLLER");
  Serial.println("Hand Controller Ready - DEBUG VERSION");

  // Initialize finger servos (attach immediately)
  for (int i = 0; i < 4; i++) {
    servos[i].attach(pins[i]);
    currentAngles[i] = 90;
    targetAngles[i] = 90;
    lastUpdate[i] = 0;
    speeds[i] = 15;
    writeMapped(i, currentAngles[i]);
  }

  // Initialize left arm data (DON'T attach)
  for (int i = LEFT_ARM_START; i < NUM_SERVOS; i++) {
    currentAngles[i] = LEFT_ARM_CENTER;
    targetAngles[i] = LEFT_ARM_CENTER;
    lastUpdate[i] = 0;
    speeds[i] = LEFT_ARM_MOVE_INTERVAL;
  }

  // Simple left arm init - NO PAUSES
  for (int i = 0; i < LEFT_ARM_COUNT; i++) {
    leftArmCurrentPos[i] = LEFT_ARM_CENTER;
    leftArmTargetPos[i] = LEFT_ARM_CENTER;
    leftArmDirection[i] = (i == 0) ? 1 : -1;  // Opposite directions
    leftArmLastMove[i] = millis();
  }

  delay(1000);
  Serial.println("Ready - Left arm DETACHED until enabled");
  Serial.println("Send LEFT_ARM_ENABLE to start movement");
}

void loop() {
  unsigned long now = millis();

  handleSerialInput();
  updateFingerServos(now);
  updateLeftArmSimple(now);

  // Debug status every 5 seconds
  if (now - lastStatusPrint > 5000) {
    printStatus();
    lastStatusPrint = now;
  }
}

void printStatus() {
  Serial.print("STATUS: enabled=");
  Serial.print(leftArmMovementEnabled ? "YES" : "NO");
  Serial.print(", attached=");
  Serial.print(leftArmServosAttached ? "YES" : "NO");
  Serial.print(", positions=[");
  for (int i = 0; i < LEFT_ARM_COUNT; i++) {
    Serial.print(leftArmCurrentPos[i]);
    if (i < LEFT_ARM_COUNT - 1) Serial.print(",");
  }
  Serial.println("]");
}

void handleSerialInput() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      processCommand(inputBuffer);
      inputBuffer = "";
    } else {
      inputBuffer += c;
    }
  }
}

void processCommand(String command) {
  command.trim();
  Serial.print("RECEIVED: ");
  Serial.println(command);

  if (command.startsWith("HAND,")) {
    // Parse hand positions
    int positions[4];
    int startIndex = 5;

    for (int i = 0; i < 4; i++) {
      int commaIndex = command.indexOf(',', startIndex);
      if (commaIndex == -1 && i == 3) {
        positions[i] = command.substring(startIndex).toInt();
      } else if (commaIndex != -1) {
        positions[i] = command.substring(startIndex, commaIndex).toInt();
        startIndex = commaIndex + 1;
      } else {
        Serial.println("ERROR: Invalid HAND command format");
        return;
      }
    }

    for (int i = 0; i < 4; i++) {
      targetAngles[i] = constrain(positions[i], 0, 180);
    }
    Serial.println("Hand positions updated");
  }
  else if (command.startsWith("LEFT_ARM_ENABLE")) {
    Serial.println("ENABLING left arm movement...");
    attachLeftArmServos();
    leftArmMovementEnabled = true;
    // Set initial targets
    for (int i = 0; i < LEFT_ARM_COUNT; i++) {
      leftArmTargetPos[i] = LEFT_ARM_CENTER + (leftArmDirection[i] * 10);
    }
    Serial.println("Left arm ENABLED and ATTACHED");
  }
  else if (command.startsWith("LEFT_ARM_DISABLE")) {
    Serial.println("DISABLING left arm movement...");
    leftArmMovementEnabled = false;
    detachLeftArmServos();
    Serial.println("Left arm DISABLED and DETACHED");
  }
  else {
    Serial.print("Unknown command: ");
    Serial.println(command);
  }
}

void attachLeftArmServos() {
  if (!leftArmServosAttached) {
    Serial.println("Attaching left arm servos...");
    for (int i = LEFT_ARM_START; i < NUM_SERVOS; i++) {
      servos[i].attach(pins[i]);
      writeMapped(i, currentAngles[i]);
    }
    leftArmServosAttached = true;
    Serial.println("Left arm servos ATTACHED");
  }
}

void detachLeftArmServos() {
  if (leftArmServosAttached) {
    Serial.println("Detaching left arm servos...");
    for (int i = LEFT_ARM_START; i < NUM_SERVOS; i++) {
      servos[i].detach();
    }
    leftArmServosAttached = false;
    Serial.println("Left arm servos DETACHED");
  }
}

void updateFingerServos(unsigned long now) {
  // Update finger servos (0-3) toward targets
  for (int i = 0; i < 4; i++) {
    if (now - lastUpdate[i] >= speeds[i]) {
      int distance = abs(targetAngles[i] - currentAngles[i]);
      int stepSize = 1;

      if (distance > 8) {
        stepSize = 3;
      } else if (distance > 4) {
        stepSize = 2;
      }

      if (currentAngles[i] < targetAngles[i]) {
        currentAngles[i] = min(currentAngles[i] + stepSize, targetAngles[i]);
      } else if (currentAngles[i] > targetAngles[i]) {
        currentAngles[i] = max(currentAngles[i] - stepSize, targetAngles[i]);
      }

      writeMapped(i, currentAngles[i]);
      lastUpdate[i] = now;
    }
  }
}

void updateLeftArmSimple(unsigned long now) {
  if (!leftArmMovementEnabled || !leftArmServosAttached) {
    return;
  }

  // SIMPLE movement: move toward target, pick new target when reached
  for (int i = 0; i < LEFT_ARM_COUNT; i++) {
    int servoIndex = LEFT_ARM_START + i;

    if (now - leftArmLastMove[i] >= LEFT_ARM_MOVE_INTERVAL) {

      // Move toward target
      if (leftArmCurrentPos[i] < leftArmTargetPos[i]) {
        leftArmCurrentPos[i] += LEFT_ARM_STEP_SIZE;
      } else if (leftArmCurrentPos[i] > leftArmTargetPos[i]) {
        leftArmCurrentPos[i] -= LEFT_ARM_STEP_SIZE;
      }

      // If reached target, pick new target
      if (abs(leftArmCurrentPos[i] - leftArmTargetPos[i]) <= LEFT_ARM_STEP_SIZE) {
        // Change direction and pick new target
        leftArmDirection[i] = -leftArmDirection[i];

        if (leftArmDirection[i] > 0) {
          leftArmTargetPos[i] = random(LEFT_ARM_CENTER + 5, LEFT_ARM_MAX);
        } else {
          leftArmTargetPos[i] = random(LEFT_ARM_MIN, LEFT_ARM_CENTER - 5);
        }

        Serial.print("Servo ");
        Serial.print(i);
        Serial.print(" new target: ");
        Serial.print(leftArmTargetPos[i]);
        Serial.print(" (current: ");
        Serial.print(leftArmCurrentPos[i]);
        Serial.println(")");
      }

      // Update servo position
      leftArmCurrentPos[i] = constrain(leftArmCurrentPos[i], LEFT_ARM_MIN, LEFT_ARM_MAX);
      currentAngles[servoIndex] = leftArmCurrentPos[i];
      writeMapped(servoIndex, currentAngles[servoIndex]);

      leftArmLastMove[i] = now;
    }
  }
}

void writeMapped(int i, int angle) {
  int mappedAngle = isMirrored[i] ? (180 - angle) : angle;
  servos[i].write(mappedAngle);
}