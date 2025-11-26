#include <Servo.h>

const int NUM_SERVOS = 6;  // Expanded to include 2 left arm servos
const int pins[NUM_SERVOS] = {8, 9, 10, 11, 4, 5};  // Added pins 4, 5 for left arm (D4, D5)
const bool isMirrored[NUM_SERVOS] = {false, false, true, true, false, false};  // Left arm servos not inverted

Servo servos[NUM_SERVOS];
int currentAngles[NUM_SERVOS];
int targetAngles[NUM_SERVOS];
unsigned long lastUpdate[NUM_SERVOS];
int speeds[NUM_SERVOS];

// Left arm autonomous movement system (servos 4 and 5) - STRICTLY LIMITED RANGE
const int LEFT_ARM_START = 4;  // First left arm servo index
const int LEFT_ARM_COUNT = 2;  // Number of left arm servos
const int LEFT_ARM_CENTER = 90;  // Center position
const int LEFT_ARM_MAX_RANGE = 15;  // 15 degrees total range with easing for smooth movement
const int LEFT_ARM_MIN = LEFT_ARM_CENTER - (LEFT_ARM_MAX_RANGE / 2);  // 82 degrees
const int LEFT_ARM_MAX = LEFT_ARM_CENTER + (LEFT_ARM_MAX_RANGE / 2);  // 97 degrees

// Autonomous movement timing - CONTINUOUS BREATHING-LIKE MOVEMENT
unsigned long leftArmLastMove[LEFT_ARM_COUNT];
unsigned long leftArmPauseDuration[LEFT_ARM_COUNT];
bool leftArmInPause[LEFT_ARM_COUNT];
String currentMood = "calm_observant";
bool leftArmMovementEnabled = false;  // NEW: Control flag - starts disabled

// Continuous movement state
float leftArmPosition[LEFT_ARM_COUNT];  // Current floating point position for smooth movement
float leftArmDirection[LEFT_ARM_COUNT]; // Movement direction (-1 to +1)
float leftArmSpeed[LEFT_ARM_COUNT];     // Individual movement speeds

// Configurable timing parameters (all in milliseconds)
// Movement speed - consistent for mechanical safety
const int LEFT_ARM_SAFE_SPEED = 50;  // Smooth fluid movement timing

// Pause durations (time between movements - longer pauses for more contemplative feel)
const unsigned long PAUSE_ENERGIZED_MIN = 15000;  const unsigned long PAUSE_ENERGIZED_MAX = 30000;   // 15-30s
const unsigned long PAUSE_ALERT_MIN = 25000;      const unsigned long PAUSE_ALERT_MAX = 45000;      // 25-45s
const unsigned long PAUSE_CALM_MIN = 45000;       const unsigned long PAUSE_CALM_MAX = 90000;       // 45-90s
const unsigned long PAUSE_QUIET_MIN = 60000;      const unsigned long PAUSE_QUIET_MAX = 120000;     // 60-120s (1-2m)
const unsigned long PAUSE_WITHDRAWN_MIN = 90000;  const unsigned long PAUSE_WITHDRAWN_MAX = 180000;  // 90-180s (1.5-3m)

// EXPANDED SERVO RANGE - now matches Python interface (0-180)
const int minAngle = 0;   // Increased from 40 to 0
const int maxAngle = 180; // Increased from 130 to 180

String inputBuffer = "";

void setup() {
  Serial.begin(9600);
  Serial.println("DEVICE_ID:HAND_CONTROLLER");
  Serial.println("Hand Controller Ready - Pure Consciousness Mode (EXPANDED RANGE)");

  // Initialize servos
  for (int i = 0; i < NUM_SERVOS; i++) {
    servos[i].attach(pins[i]);
    currentAngles[i] = 90;  // Start at middle position (was 70)
    targetAngles[i] = 90;
    lastUpdate[i] = 0;
    speeds[i] = 15;  // Default speed
    writeMapped(i, currentAngles[i]);
  }

  // Initialize left arm autonomous movement system (but don't start moving yet)
  initializeLeftArm();

  delay(1000);
  Serial.println("Ready for consciousness commands (0-180 degree range)");
  Serial.println("Left arm movement DISABLED - waiting for Python enable command");
}

void loop() {
  unsigned long now = millis();

  // Handle serial commands for hand servos
  handleSerialInput();

  // Update servo positions toward targets set by consciousness (hand servos)
  updateServoPositions(now);

  // Update autonomous left arm movement (handles its own servo positioning)
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
  if (command.startsWith("HAND,")) {
    // Parse consciousness command: "HAND,f0,f1,f2,f3"
    int commaIndex = command.indexOf(',');
    String fingerData = command.substring(commaIndex + 1);

    int fingerPositions[4];
    int index = 0;
    int lastComma = -1;

    for (int i = 0; i < 4 && index < 4; i++) {
      int nextComma = fingerData.indexOf(',', lastComma + 1);
      if (nextComma == -1 && i == 3) {
        // Last finger
        fingerPositions[index] = fingerData.substring(lastComma + 1).toInt();
      } else if (nextComma != -1) {
        fingerPositions[index] = fingerData.substring(lastComma + 1, nextComma).toInt();
      }
      lastComma = nextComma;
      index++;
    }

    // Set targets for pure consciousness control - EXCLUDE left arm servos
    for (int i = 0; i < NUM_SERVOS - LEFT_ARM_COUNT; i++) {  // Only control hand servos, not left arm
      int newTarget = constrain(fingerPositions[i], minAngle, maxAngle);
      targetAngles[i] = newTarget;

      // Adaptive servo speeds based on movement distance
      int distance = abs(newTarget - currentAngles[i]);

      // Check if this looks like a startle reaction (large simultaneous movement)
      bool isStartleMovement = false;
      if (distance > 25) {  // Increased threshold for larger range (was 15)
        int totalMovement = 0;
        for (int j = 0; j < NUM_SERVOS - LEFT_ARM_COUNT; j++) {  // Only check hand servos
          totalMovement += abs(fingerPositions[j] - currentAngles[j]);
        }
        // If total movement across all fingers is large, it's likely a startle
        if (totalMovement > 100) {  // Increased threshold for larger range (was 60)
          isStartleMovement = true;
        }
      }

      if (isStartleMovement) {
        speeds[i] = 3;   // VERY fast for startle reactions - immediate response!
      } else if (distance > 30) {  // Adjusted for larger range (was 20)
        speeds[i] = 8;   // Fast for large movements
      } else if (distance > 10) {  // Adjusted for larger range (was 5)
        speeds[i] = 12;  // Medium speed for medium movements
      } else {
        speeds[i] = 20;  // Slower for fine adjustments - prevents servo noise
      }
    }

    // Optional: Echo command for debugging
    Serial.print("Consciousness: ");
    for (int i = 0; i < 4; i++) {
      Serial.print(fingerPositions[i]);
      if (i < 3) Serial.print(",");
    }
    Serial.println();
  }
  else if (command.startsWith("MOOD,")) {
    // Parse mood command: "MOOD,calm_observant"
    int commaIndex = command.indexOf(',');
    String moodState = command.substring(commaIndex + 1);
    moodState.trim();

    // Update current mood for left arm movement parameters
    currentMood = moodState;
    Serial.print("Mood updated to: ");
    Serial.println(currentMood);

    // Speed stays consistent for mechanical safety - only pause behavior changes
  }
  else if (command.startsWith("HEARTBEAT")) {
    // Acknowledge heartbeat but don't do anything special
    // This is purely for Python's peace of mind
    Serial.println("Heartbeat acknowledged");
  }
  else if (command.startsWith("LEFT_ARM_ENABLE")) {
    // Enable left arm autonomous movement
    leftArmMovementEnabled = true;
    Serial.println("Left arm movement ENABLED");
  }
  else if (command.startsWith("LEFT_ARM_DISABLE")) {
    // Disable left arm autonomous movement
    leftArmMovementEnabled = false;
    Serial.println("Left arm movement DISABLED");
  }
}

void updateServoPositions(unsigned long now) {
  // Simple, clean servo movement toward targets - EXCLUDE left arm servos (they have their own update function)
  for (int i = 0; i < NUM_SERVOS - LEFT_ARM_COUNT; i++) {
    if (now - lastUpdate[i] >= speeds[i]) {
      // Variable step size based on speed for responsiveness
      int stepSize = 1;  // Default smooth movement
      if (speeds[i] <= 3) {
        stepSize = 12;  // INSTANT: 12 degrees per step for startle reactions (increased from 8)
      } else if (speeds[i] <= 10) {
        stepSize = 4;   // Fast: 4 degrees per step (increased from 3)
      }

      // Move toward target with variable step size
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

void updateLeftArmServoPositions(unsigned long now) {
  // Dedicated smooth eased movement for left arm servos only
  for (int i = 0; i < LEFT_ARM_COUNT; i++) {
    int servoIndex = LEFT_ARM_START + i;

    if (now - lastUpdate[servoIndex] >= speeds[servoIndex]) {
      // Smooth movement with variable step size based on distance to target
      int distance = abs(targetAngles[servoIndex] - currentAngles[servoIndex]);
      int stepSize = 1;

      // Use larger steps for smoother movement
      if (distance > 8) {
        stepSize = 3;  // Larger steps when far from target
      } else if (distance > 4) {
        stepSize = 2;  // Medium steps when medium distance
      } else {
        stepSize = 1;  // Small steps when close to target
      }

      // Apply the movement step
      if (currentAngles[servoIndex] < targetAngles[servoIndex]) {
        currentAngles[servoIndex] = min(currentAngles[servoIndex] + stepSize, targetAngles[servoIndex]);
        writeMapped(servoIndex, currentAngles[servoIndex]);
      } else if (currentAngles[servoIndex] > targetAngles[servoIndex]) {
        currentAngles[servoIndex] = max(currentAngles[servoIndex] - stepSize, targetAngles[servoIndex]);
        writeMapped(servoIndex, currentAngles[servoIndex]);
      }

      lastUpdate[servoIndex] = now;
    }
  }
}

void writeMapped(int i, int angle) {
  int mappedAngle = isMirrored[i] ? (180 - angle) : angle;
  servos[i].write(mappedAngle);
}

void updateLeftArmMovement(unsigned long now) {
  // Only move if enabled by Python script
  if (!leftArmMovementEnabled) {
    return;
  }

  // Continuous breathing-like movement for left arm servos
  for (int i = 0; i < LEFT_ARM_COUNT; i++) {
    int servoIndex = LEFT_ARM_START + i;

    if (leftArmInPause[i]) {
      // Check if pause is over
      if (now - leftArmLastMove[i] >= leftArmPauseDuration[i]) {
        leftArmInPause[i] = false;
        leftArmLastMove[i] = now;
        Serial.print("Left arm servo ");
        Serial.print(i);
        Serial.println(" resuming continuous movement");
      }
    } else {
      // Continuous movement every update cycle
      if (now - leftArmLastMove[i] >= LEFT_ARM_SAFE_SPEED) {

        // Move the floating point position
        leftArmPosition[i] += leftArmDirection[i] * leftArmSpeed[i];

        // Check boundaries and reverse direction if needed
        if (leftArmPosition[i] >= LEFT_ARM_MAX) {
          leftArmPosition[i] = LEFT_ARM_MAX;
          leftArmDirection[i] = -leftArmDirection[i]; // Reverse direction
          // Slight speed variation for organic feel
          leftArmSpeed[i] = random(10, 30) / 100.0; // 0.1 to 0.3 degrees per step
        } else if (leftArmPosition[i] <= LEFT_ARM_MIN) {
          leftArmPosition[i] = LEFT_ARM_MIN;
          leftArmDirection[i] = -leftArmDirection[i]; // Reverse direction
          leftArmSpeed[i] = random(10, 30) / 100.0; // 0.1 to 0.3 degrees per step
        }

        // Convert to integer and apply to servo
        int newAngle = (int)(leftArmPosition[i] + 0.5); // Round to nearest integer
        newAngle = constrain(newAngle, LEFT_ARM_MIN, LEFT_ARM_MAX); // Safety constraint

        if (newAngle != currentAngles[servoIndex]) {
          currentAngles[servoIndex] = newAngle;
          writeMapped(servoIndex, currentAngles[servoIndex]);
        }

        leftArmLastMove[i] = now;

        // Random chance to pause (much less frequent for continuous feel)
        int pauseChance = getPauseChance() / 10; // Divide by 10 for much less frequent pauses
        if (random(0, 10000) < pauseChance) { // Out of 10000 instead of 1000
          leftArmInPause[i] = true;
          leftArmPauseDuration[i] = getMoodPauseDuration();
          Serial.print("Left arm servo ");
          Serial.print(i);
          Serial.print(" pausing for ");
          Serial.print(leftArmPauseDuration[i] / 1000);
          Serial.println(" seconds");
        }
      }
    }
  }
}

int getLeftArmSafeSpeed() {
  // Return consistent safe speed for mechanical protection
  return LEFT_ARM_SAFE_SPEED;
}

int getPauseChance() {
  // Return pause probability (per 1000) - reduced for more subtle movement
  if (currentMood == "energized_engaged") return 2;    // 0.2% chance per cycle - very rarely pauses
  if (currentMood == "alert_curious") return 5;        // 0.5% chance per cycle
  if (currentMood == "calm_observant") return 10;      // 1% chance per cycle
  if (currentMood == "quiet_detached") return 20;      // 2% chance per cycle
  if (currentMood == "withdrawn_distant") return 40;   // 4% chance per cycle - more frequent pauses
  return 10; // Default to calm_observant
}

unsigned long getMoodPauseDuration() {
  // Return pause duration based on current mood (uses configurable constants)
  if (currentMood == "energized_engaged") return random(PAUSE_ENERGIZED_MIN, PAUSE_ENERGIZED_MAX);
  if (currentMood == "alert_curious") return random(PAUSE_ALERT_MIN, PAUSE_ALERT_MAX);
  if (currentMood == "calm_observant") return random(PAUSE_CALM_MIN, PAUSE_CALM_MAX);
  if (currentMood == "quiet_detached") return random(PAUSE_QUIET_MIN, PAUSE_QUIET_MAX);
  if (currentMood == "withdrawn_distant") return random(PAUSE_WITHDRAWN_MIN, PAUSE_WITHDRAWN_MAX);
  return random(PAUSE_CALM_MIN, PAUSE_CALM_MAX); // Default to calm_observant
}

void initializeLeftArm() {
  // Initialize left arm servos for continuous breathing-like movement
  for (int i = 0; i < LEFT_ARM_COUNT; i++) {
    int servoIndex = LEFT_ARM_START + i;

    // Start at random positions within safe range
    int startPosition = random(LEFT_ARM_MIN, LEFT_ARM_MAX + 1);
    currentAngles[servoIndex] = startPosition;
    leftArmPosition[i] = (float)startPosition; // Initialize floating point position

    // Set random initial directions (one up, one down for variety)
    leftArmDirection[i] = (i % 2 == 0) ? 1.0 : -1.0; // Alternate directions

    // Set random initial speeds
    leftArmSpeed[i] = random(15, 25) / 100.0; // 0.15 to 0.25 degrees per step

    // Start with initial pause to prevent startup jolt
    leftArmInPause[i] = true;
    leftArmLastMove[i] = millis();
    leftArmPauseDuration[i] = random(3000, 6000); // Initial pause 3-6s
    speeds[servoIndex] = getLeftArmSafeSpeed();

    Serial.print("Left arm servo ");
    Serial.print(i);
    Serial.print(" initialized at ");
    Serial.print(startPosition);
    Serial.print(" degrees, direction: ");
    Serial.print(leftArmDirection[i] > 0 ? "up" : "down");
    Serial.print(", speed: ");
    Serial.println(leftArmSpeed[i]);
  }
}