#include <Servo.h>

const int NUM_SERVOS = 6;  // Expanded to include 2 left arm servos
const int pins[NUM_SERVOS] = {8, 9, 10, 11, 4, 5};  // Added pins 4, 5 for left arm (D4, D5)
const bool isMirrored[NUM_SERVOS] = {false, false, true, true, false, false};  // Left arm servos not inverted

Servo servos[NUM_SERVOS];
int currentAngles[NUM_SERVOS];
int targetAngles[NUM_SERVOS];
unsigned long lastUpdate[NUM_SERVOS];
int speeds[NUM_SERVOS];

// Left arm autonomous movement system - IMPROVED VERSION
const int LEFT_ARM_START = 4;  // First left arm servo index
const int LEFT_ARM_COUNT = 2;  // Number of left arm servos
const int LEFT_ARM_CENTER = 90;  // Center position
const int LEFT_ARM_MAX_RANGE = 40;  // INCREASED: 40 degrees total range (was 15)
const int LEFT_ARM_MIN = LEFT_ARM_CENTER - (LEFT_ARM_MAX_RANGE / 2);  // 70 degrees
const int LEFT_ARM_MAX = LEFT_ARM_CENTER + (LEFT_ARM_MAX_RANGE / 2);  // 110 degrees

// Simplified movement state with pause system
bool leftArmMovementEnabled = false;  // Control flag - starts disabled
bool leftArmServosAttached = false;   // NEW: Track if left arm servos are attached
unsigned long leftArmLastMove[LEFT_ARM_COUNT];
float leftArmPosition[LEFT_ARM_COUNT];  // Current floating point position
float leftArmDirection[LEFT_ARM_COUNT]; // Movement direction (-1 to +1)
float leftArmSpeed[LEFT_ARM_COUNT];     // Individual movement speeds

// Simple pause system
bool leftArmInPause[LEFT_ARM_COUNT];
unsigned long leftArmPauseStart[LEFT_ARM_COUNT];
unsigned long leftArmPauseDuration[LEFT_ARM_COUNT];

// Dynamic movement system - simplified dynamics from Python
struct MovementDynamics {
  float speed;           // 0-1: Overall movement speed factor
  float transition_rate; // 0-1: How often to change direction/behavior
  float variability;     // 0-1: Randomness in movement parameters
  float pause_tendency;  // 0-1: Likelihood to pause (now handled by Python)
  float entropy;         // 0-1: Complexity/chaos in movement
} dynamics = {0.3, 0.2, 0.1, 0.5, 0.2}; // Default moderate values

// SLOWED DOWN: Much slower timing for smooth movement
const int LEFT_ARM_BASE_SPEED = 150;  // SLOWER: 150ms interval (30% of previous speed)

String inputBuffer = "";

void setup() {
  Serial.begin(9600);
  Serial.println("DEVICE_ID:HAND_CONTROLLER");
  Serial.println("Hand Controller Ready - Improved Left Arm Control");

  // Initialize hand servos only (0-3)
  for (int i = 0; i < 4; i++) {
    servos[i].attach(pins[i]);
    currentAngles[i] = 90;  // Start at middle position
    targetAngles[i] = 90;
    lastUpdate[i] = 0;
    speeds[i] = 15;  // Default speed
    writeMapped(i, currentAngles[i]);
  }

  // Initialize left arm servo data but don't attach yet
  for (int i = LEFT_ARM_START; i < NUM_SERVOS; i++) {
    currentAngles[i] = LEFT_ARM_CENTER;
    targetAngles[i] = LEFT_ARM_CENTER;
    lastUpdate[i] = 0;
    speeds[i] = LEFT_ARM_BASE_SPEED;
  }

  // Initialize left arm movement system
  initializeLeftArm();

  delay(1000);
  Serial.println("Ready for commands (0-180 degree range)");
  Serial.println("Left arm movement DISABLED - waiting for Python enable command");
}

void loop() {
  unsigned long now = millis();

  // Handle serial commands
  handleSerialInput();

  // Update servo positions toward targets (hand servos)
  updateServoPositions(now);

  // Update left arm autonomous movement (improved)
  updateLeftArmMovement(now);
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

  if (command.startsWith("HAND,")) {
    // Parse hand positions: "HAND,pos1,pos2,pos3,pos4"
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
        Serial.println("ERROR: Invalid HAND command format");
        return;
      }
    }

    // Set hand servo targets (servos 0-3)
    for (int i = 0; i < 4; i++) {
      targetAngles[i] = constrain(positions[i], 0, 180);
    }
  }
  else if (command.startsWith("LEFT_ARM_ENABLE")) {
    attachLeftArmServos();
    leftArmMovementEnabled = true;
    Serial.println("Left arm movement ENABLED and servos ATTACHED");
  }
  else if (command.startsWith("LEFT_ARM_DISABLE")) {
    leftArmMovementEnabled = false;
    detachLeftArmServos();
    Serial.println("Left arm movement DISABLED and servos DETACHED");
  }
  else if (command.startsWith("DYNAMICS,")) {
    // Parse dynamics: "DYNAMICS,speed,transition_rate,variability,pause_tendency,entropy"
    parseDynamics(command);
  }
  else if (command.startsWith("MOOD,")) {
    // Parse mood command - not used in simplified version but kept for compatibility
    Serial.print("Mood received: ");
    Serial.println(command.substring(5));
  }
}

void attachLeftArmServos() {
  if (!leftArmServosAttached) {
    for (int i = LEFT_ARM_START; i < NUM_SERVOS; i++) {
      servos[i].attach(pins[i]);
      writeMapped(i, currentAngles[i]);  // Set to current position
    }
    leftArmServosAttached = true;
    Serial.println("Left arm servos ATTACHED");
  }
}

void detachLeftArmServos() {
  if (leftArmServosAttached) {
    for (int i = LEFT_ARM_START; i < NUM_SERVOS; i++) {
      servos[i].detach();
    }
    leftArmServosAttached = false;
    Serial.println("Left arm servos DETACHED");
  }
}

void parseDynamics(String command) {
  // Parse: "DYNAMICS,speed,transition_rate,variability,pause_tendency,entropy"
  int startIndex = 9; // Skip "DYNAMICS,"
  float values[5];

  for (int i = 0; i < 5; i++) {
    int commaIndex = command.indexOf(',', startIndex);
    if (commaIndex == -1 && i == 4) {
      values[i] = command.substring(startIndex).toFloat();
    } else if (commaIndex != -1) {
      values[i] = command.substring(startIndex, commaIndex).toFloat();
      startIndex = commaIndex + 1;
    } else {
      Serial.println("ERROR: Invalid DYNAMICS command format");
      return;
    }
  }

  dynamics.speed = constrain(values[0], 0.0, 1.0);
  dynamics.transition_rate = constrain(values[1], 0.0, 1.0);
  dynamics.variability = constrain(values[2], 0.0, 1.0);
  dynamics.pause_tendency = constrain(values[3], 0.0, 1.0);
  dynamics.entropy = constrain(values[4], 0.0, 1.0);

  Serial.print("Dynamics updated: speed=");
  Serial.print(dynamics.speed, 3);
  Serial.print(", transition=");
  Serial.print(dynamics.transition_rate, 3);
  Serial.print(", variability=");
  Serial.print(dynamics.variability, 3);
  Serial.print(", pause=");
  Serial.print(dynamics.pause_tendency, 3);
  Serial.print(", entropy=");
  Serial.println(dynamics.entropy, 3);
}

void updateServoPositions(unsigned long now) {
  // Update hand servos (0-3) toward targets
  for (int i = 0; i < 4; i++) {
    if (now - lastUpdate[i] >= speeds[i]) {
      // Move toward target with smooth steps
      int distance = abs(targetAngles[i] - currentAngles[i]);
      int stepSize = 1;

      if (distance > 8) {
        stepSize = 3;
      } else if (distance > 4) {
        stepSize = 2;
      }

      // Move toward target
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

void writeMapped(int i, int angle) {
  int mappedAngle = isMirrored[i] ? (180 - angle) : angle;
  servos[i].write(mappedAngle);
}

void updateLeftArmMovement(unsigned long now) {
  // Only move if enabled AND servos are attached
  if (!leftArmMovementEnabled || !leftArmServosAttached) {
    return;
  }

  // IMPROVED: Simpler, faster movement for left arm servos
  for (int i = 0; i < LEFT_ARM_COUNT; i++) {
    int servoIndex = LEFT_ARM_START + i;

    // Check if in pause
    if (leftArmInPause[i]) {
      // Check if pause is over
      if (now - leftArmPauseStart[i] >= leftArmPauseDuration[i]) {
        leftArmInPause[i] = false;
        leftArmLastMove[i] = now;
        Serial.print("Left arm servo ");
        Serial.print(i);
        Serial.println(" resuming movement after pause");
      } else {
        continue; // Still pausing, skip movement
      }
    }

    // SLOWED DOWN: Much slower timing calculation
    float speedMultiplier = 0.5 + dynamics.speed * 1.5; // 0.5x to 2.0x speed (reduced range)
    int dynamicInterval = LEFT_ARM_BASE_SPEED / speedMultiplier;
    dynamicInterval = constrain(dynamicInterval, 100, 500); // 100ms to 500ms (much slower)

    if (now - leftArmLastMove[i] >= dynamicInterval) {
      // SLOWED DOWN: Much smaller movement steps
      float baseStepSize = 0.3 + dynamics.speed * 0.7; // 0.3 to 1.0 degree steps (much smaller)
      float currentStepSize = baseStepSize * (0.8 + random(0, 40) / 100.0); // ±20% variation

      // Move the floating point position
      leftArmPosition[i] += leftArmDirection[i] * currentStepSize;

      // Check boundaries and reverse direction
      if (leftArmPosition[i] >= LEFT_ARM_MAX) {
        leftArmPosition[i] = LEFT_ARM_MAX;
        leftArmDirection[i] = -leftArmDirection[i];
        // Vary speed on boundary hit
        leftArmSpeed[i] = 0.5 + (random(0, 50) / 100.0); // 0.5 to 1.0
      } else if (leftArmPosition[i] <= LEFT_ARM_MIN) {
        leftArmPosition[i] = LEFT_ARM_MIN;
        leftArmDirection[i] = -leftArmDirection[i];
        // Vary speed on boundary hit
        leftArmSpeed[i] = 0.5 + (random(0, 50) / 100.0); // 0.5 to 1.0
      }

      // Random direction changes (simplified)
      if (random(0, 100) < (dynamics.transition_rate * 10)) { // 0-10% chance per update
        leftArmDirection[i] = -leftArmDirection[i];
      }

      // Convert to integer and apply to servo
      int newAngle = (int)(leftArmPosition[i] + 0.5);
      newAngle = constrain(newAngle, LEFT_ARM_MIN, LEFT_ARM_MAX);

      if (newAngle != currentAngles[servoIndex]) {
        currentAngles[servoIndex] = newAngle;
        writeMapped(servoIndex, currentAngles[servoIndex]);
      }

      leftArmLastMove[i] = now;

      // IMPROVED: Much simpler pause system
      if (random(0, 1000) < (dynamics.pause_tendency * 5)) { // 0-0.5% chance per movement
        leftArmInPause[i] = true;
        leftArmPauseStart[i] = now;

        // Shorter, simpler pauses
        leftArmPauseDuration[i] = 3000 + random(0, 7000); // 3-10 second pauses (was too long)

        Serial.print("Left arm servo ");
        Serial.print(i);
        Serial.print(" pausing for ");
        Serial.print(leftArmPauseDuration[i] / 1000);
        Serial.println(" seconds");
      }
    }
  }
}

void initializeLeftArm() {
  // Initialize left arm servos for continuous movement
  for (int i = 0; i < LEFT_ARM_COUNT; i++) {
    int servoIndex = LEFT_ARM_START + i;

    // Start at center position
    int startPosition = LEFT_ARM_CENTER;
    currentAngles[servoIndex] = startPosition;
    leftArmPosition[i] = (float)startPosition;

    // Set random initial directions
    leftArmDirection[i] = (i % 2 == 0) ? 1.0 : -1.0;

    // IMPROVED: Simpler initial speeds
    leftArmSpeed[i] = 0.5 + random(0, 50) / 100.0; // 0.5 to 1.0

    // Initialize pause system - start with short initial pause
    leftArmInPause[i] = true;
    leftArmPauseStart[i] = millis();
    leftArmPauseDuration[i] = random(1000, 3000); // SHORTER: 1-3 second initial pause (was 2-5)

    leftArmLastMove[i] = millis();
    speeds[servoIndex] = LEFT_ARM_BASE_SPEED;

    Serial.print("Left arm servo ");
    Serial.print(i);
    Serial.print(" initialized at ");
    Serial.print(startPosition);
    Serial.print(" degrees, direction: ");
    Serial.print(leftArmDirection[i] > 0 ? "up" : "down");
    Serial.print(", speed: ");
    Serial.print(leftArmSpeed[i], 2);
    Serial.print(", initial pause: ");
    Serial.print(leftArmPauseDuration[i] / 1000);
    Serial.println("s");
  }
}