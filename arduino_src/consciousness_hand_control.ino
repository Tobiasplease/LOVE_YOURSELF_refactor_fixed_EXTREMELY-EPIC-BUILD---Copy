/*
 * LOVE_YOURSELF Consciousness-Driven Hand Control
 * Based on your existing tapping hand sketch
 * 
 * Supports both autonomous tapping behavior and consciousness-driven gestures
 * Maintains your mirrored servo setup and proven movement mechanics
 * 
 * Serial Commands:
 * - PAN:angle       (existing - mirror pan)
 * - TILT:angle      (existing - mirror tilt) 
 * - LUNG:angle      (existing - breathing servo)
 * - FINGER0:angle   (new - index finger, pin 8)
 * - FINGER1:angle   (new - middle finger, pin 9)
 * - FINGER2:angle   (new - ring finger, pin 10, mirrored)
 * - FINGER3:angle   (new - pinky finger, pin 11, mirrored)
 * - TAP:ON/OFF      (new - enable/disable autonomous tapping)
 */

#include <Servo.h>

// === EXISTING MIRROR SERVOS ===
Servo panServo;
Servo tiltServo;
Servo lungServo;

// === HAND SERVOS ===
const int NUM_FINGERS = 4;
const int fingerPins[NUM_FINGERS] = {8, 9, 10, 11};
const bool fingerMirrored[NUM_FINGERS] = {false, false, true, true};

Servo fingerServos[NUM_FINGERS];
int currentFingerAngles[NUM_FINGERS];
int targetFingerAngles[NUM_FINGERS];
unsigned long lastFingerUpdate[NUM_FINGERS];
int fingerSpeeds[NUM_FINGERS];

// === HAND BEHAVIOR SETTINGS ===
const int minFingerAngle = 40;
const int maxFingerAngle = 130;
const int minFingerSpeed = 20;
const int maxFingerSpeed = 50;

// === TAPPING BEHAVIOR ===
const int tapFingerIndex = 3; // Pinky finger (pin 11)
int tapCurled = 130;          // 30° when mirrored (curled)
int tapOutstretched = 10;     // 170° when mirrored (outstretched)
int tapStepFast = 3;
int tapStepEase = 3;

bool tappingEnabled = true;
unsigned long tapPauseAtTop = random(250, 400);
unsigned long tapTimer = 0;
bool tappingDown = true;
bool tapIsPaused = false;
unsigned long tapUpdateTimer = 0;
const unsigned long tapUpdateInterval = 10;

// === CONSCIOUSNESS CONTROL ===
bool consciousnessMode = false;  // When true, consciousness overrides autonomous behavior
unsigned long lastConsciousnessCommand = 0;
const unsigned long consciousnessTimeout = 5000; // Return to autonomous after 5s of no commands

void setup() {
  Serial.begin(9600);
  
  // === INITIALIZE MIRROR SERVOS ===
  panServo.attach(2);
  tiltServo.attach(3);  
  lungServo.attach(4);
  
  panServo.write(90);   // Center position
  tiltServo.write(90);  // Center position
  lungServo.write(90);  // Center position
  
  // === INITIALIZE HAND SERVOS ===
  for (int i = 0; i < NUM_FINGERS; i++) {
    fingerServos[i].attach(fingerPins[i]);
    currentFingerAngles[i] = random(minFingerAngle, maxFingerAngle);
    targetFingerAngles[i] = currentFingerAngles[i];
    writeFingerMapped(i, currentFingerAngles[i]);
    lastFingerUpdate[i] = millis();
    fingerSpeeds[i] = random(minFingerSpeed, maxFingerSpeed);
  }
  
  randomSeed(analogRead(0));
  Serial.println("LOVE_YOURSELF Hand Controller Ready");
}

void loop() {
  unsigned long now = millis();
  
  // === HANDLE SERIAL COMMANDS ===
  handleSerialCommands();
  
  // === CHECK CONSCIOUSNESS TIMEOUT ===
  if (consciousnessMode && (now - lastConsciousnessCommand > consciousnessTimeout)) {
    consciousnessMode = false;
    Serial.println("Returning to autonomous mode");
  }
  
  // === HAND BEHAVIOR ===
  if (!consciousnessMode) {
    // Autonomous behavior - your original organic movements
    updateAutonomousHandBehavior(now);
  }
  // If in consciousness mode, finger positions are controlled by serial commands
}

void handleSerialCommands() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    if (command.startsWith("PAN:")) {
      int angle = command.substring(4).toInt();
      angle = constrain(angle, 0, 180);
      panServo.write(angle);
      
    } else if (command.startsWith("TILT:")) {
      int angle = command.substring(5).toInt();
      angle = constrain(angle, 0, 180);
      tiltServo.write(angle);
      
    } else if (command.startsWith("LUNG:")) {
      int angle = command.substring(5).toInt();
      angle = constrain(angle, 0, 180);
      lungServo.write(angle);
      
    } else if (command.startsWith("FINGER")) {
      // Parse FINGER0:angle, FINGER1:angle, etc.
      int colonPos = command.indexOf(':');
      if (colonPos > 0) {
        int fingerIndex = command.substring(6, colonPos).toInt(); // Extract number after "FINGER"
        int angle = command.substring(colonPos + 1).toInt();
        
        if (fingerIndex >= 0 && fingerIndex < NUM_FINGERS) {
          angle = constrain(angle, minFingerAngle, maxFingerAngle);
          setFingerTarget(fingerIndex, angle);
          consciousnessMode = true;
          lastConsciousnessCommand = millis();
        }
      }
      
    } else if (command.startsWith("TAP:")) {
      String mode = command.substring(4);
      if (mode == "ON") {
        tappingEnabled = true;
        Serial.println("Tapping enabled");
      } else if (mode == "OFF") {
        tappingEnabled = false;
        Serial.println("Tapping disabled");
      }
    }
  }
}

void setFingerTarget(int fingerIndex, int angle) {
  targetFingerAngles[fingerIndex] = angle;
  fingerSpeeds[fingerIndex] = 30; // Smooth movement for consciousness control
}

void updateAutonomousHandBehavior(unsigned long now) {
  // === BACKGROUND FINGER MOTION (your original code) ===
  for (int i = 0; i < NUM_FINGERS; i++) {
    if (i == tapFingerIndex && tappingEnabled) continue; // Skip tap finger
    
    if (currentFingerAngles[i] == targetFingerAngles[i]) {
      targetFingerAngles[i] = random(minFingerAngle, maxFingerAngle);
      fingerSpeeds[i] = random(minFingerSpeed, maxFingerSpeed);
    }

    if (now - lastFingerUpdate[i] >= fingerSpeeds[i]) {
      if (currentFingerAngles[i] < targetFingerAngles[i]) currentFingerAngles[i]++;
      else if (currentFingerAngles[i] > targetFingerAngles[i]) currentFingerAngles[i]--;
      writeFingerMapped(i, currentFingerAngles[i]);
      lastFingerUpdate[i] = now;
    }
  }

  // === TAPPING BEHAVIOR (your original code) ===
  if (tappingEnabled && now - tapUpdateTimer >= tapUpdateInterval) {
    tapUpdateTimer = now;

    if (!tapIsPaused) {
      int target = tappingDown ? tapCurled : tapOutstretched;
      int step;

      if (!tappingDown) {
        // Easing toward outstretched
        if (currentFingerAngles[tapFingerIndex] < tapOutstretched + 20) step = tapStepEase;
        else step = tapStepFast;
      } else {
        // Snappy tap down
        step = tapStepFast;
      }

      // Move toward target
      if (currentFingerAngles[tapFingerIndex] < target) currentFingerAngles[tapFingerIndex] += step;
      else if (currentFingerAngles[tapFingerIndex] > target) currentFingerAngles[tapFingerIndex] -= step;

      // Clamp
      if ((tappingDown && currentFingerAngles[tapFingerIndex] > target) ||
          (!tappingDown && currentFingerAngles[tapFingerIndex] < target)) {
        currentFingerAngles[tapFingerIndex] = target;
      }

      writeFingerMapped(tapFingerIndex, currentFingerAngles[tapFingerIndex]);

      if (currentFingerAngles[tapFingerIndex] == target) {
        tapTimer = now;
        tapIsPaused = true;
      }

    } else {
      if (!tappingDown) {
        // Pause at top (outstretched)
        if (now - tapTimer >= tapPauseAtTop) {
          tappingDown = true;
          tapIsPaused = false;
          tapPauseAtTop = random(250, 400); // refresh
        }
      } else {
        // No pause at bottom
        tappingDown = false;
        tapIsPaused = false;
      }
    }
  }
  
  // Update finger positions in consciousness mode (smooth movement)
  if (consciousnessMode) {
    for (int i = 0; i < NUM_FINGERS; i++) {
      if (now - lastFingerUpdate[i] >= fingerSpeeds[i]) {
        if (currentFingerAngles[i] < targetFingerAngles[i]) currentFingerAngles[i]++;
        else if (currentFingerAngles[i] > targetFingerAngles[i]) currentFingerAngles[i]--;
        writeFingerMapped(i, currentFingerAngles[i]);
        lastFingerUpdate[i] = now;
      }
    }
  }
}

void writeFingerMapped(int fingerIndex, int angle) {
  int mappedAngle = fingerMirrored[fingerIndex] ? (180 - angle) : angle;
  fingerServos[fingerIndex].write(mappedAngle);
}
