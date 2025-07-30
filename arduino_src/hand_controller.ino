/*
 * Hand Controller for AI Consciousness System
 * 
 * Receives commands from Python via serial (COM11) and controls 4 finger servos
 * Based on your existing tapping behavior but extended for consciousness-driven expressions
 * 
 * Command Format: "HAND,finger0,finger1,finger2,finger3\n"
 * Example: "HAND,70,80,90,60\n" - sets each finger to specified angle
 * 
 * Maintains your existing mirrored servo setup and ranges
 */

#include <Servo.h>

const int NUM_SERVOS = 4;
const int pins[NUM_SERVOS] = {8, 9, 10, 11};
const bool isMirrored[NUM_SERVOS] = {false, false, true, true};

Servo servos[NUM_SERVOS];
int currentAngles[NUM_SERVOS];
int targetAngles[NUM_SERVOS];
unsigned long lastUpdate[NUM_SERVOS];
int speeds[NUM_SERVOS];

const int minAngle = 40;
const int maxAngle = 130;
const int minSpeed = 15;  // Slower for more organic movement
const int maxSpeed = 35;

// Consciousness control state
bool consciousnessMode = false;
unsigned long lastConsciousnessCommand = 0;
const unsigned long consciousnessTimeout = 5000; // 5 seconds

// Your original tapping logic (for fallback/idle behavior)
const int tapIndex = 3; // pin 11 = finger D
int tapCurled = 130;      
int tapOutstretched = 10; 
int tapStepFast = 2;      // Slightly slower for more organic feel
int tapStepEase = 2;

unsigned long tapPauseAtTop = random(300, 600);  // Longer pauses
unsigned long tapTimer = 0;
bool tappingDown = true;
bool isPaused = false;
unsigned long tapUpdateTimer = 0;
const unsigned long tapUpdateInterval = 15;  // Slightly slower update

String inputBuffer = "";

void setup() {
  Serial.begin(9600);
  Serial.println("Hand Controller Ready - Waiting for consciousness...");
  
  for (int i = 0; i < NUM_SERVOS; i++) {
    servos[i].attach(pins[i]);
    currentAngles[i] = random(minAngle + 10, maxAngle - 10);  // Start in comfortable range
    targetAngles[i] = currentAngles[i];
    writeMapped(i, currentAngles[i]);
    lastUpdate[i] = millis();
    speeds[i] = random(minSpeed, maxSpeed);
  }
  randomSeed(analogRead(0));
}

void loop() {
  unsigned long now = millis();
  
  // Check for serial commands from consciousness system
  handleSerialInput();
  
  // Check if consciousness system is still active
  if (consciousnessMode && (now - lastConsciousnessCommand > consciousnessTimeout)) {
    consciousnessMode = false;
    Serial.println("Consciousness timeout - resuming autonomous behavior");
  }
  
  // If consciousness is not controlling, run autonomous behavior
  if (!consciousnessMode) {
    runAutonomousBehavior(now);
  }
  
  // Always update servo positions smoothly
  updateServoPositions(now);
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
    consciousnessMode = true;
    lastConsciousnessCommand = millis();
    
    // Extract finger positions
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
    
    // Set targets and speeds for smooth movement
    for (int i = 0; i < NUM_SERVOS; i++) {
      int newTarget = constrain(fingerPositions[i], minAngle, maxAngle);
      targetAngles[i] = newTarget;
      
      // Vary speed based on distance to travel (closer = slower, more organic)
      int distance = abs(newTarget - currentAngles[i]);
      speeds[i] = map(distance, 0, 90, minSpeed, maxSpeed);
    }
    
    Serial.print("Consciousness command: ");
    for (int i = 0; i < 4; i++) {
      Serial.print(fingerPositions[i]);
      if (i < 3) Serial.print(",");
    }
    Serial.println();
  }
}

void runAutonomousBehavior(unsigned long now) {
  // Your original background finger motion
  for (int i = 0; i < NUM_SERVOS; i++) {
    if (i == tapIndex) continue;
    if (currentAngles[i] == targetAngles[i]) {
      targetAngles[i] = random(minAngle, maxAngle);
      speeds[i] = random(minSpeed, maxSpeed);
    }
  }

  // Your original tap logic (slightly modified for more organic feel)
  if (now - tapUpdateTimer >= tapUpdateInterval) {
    tapUpdateTimer = now;

    if (!isPaused) {
      int target = tappingDown ? tapCurled : tapOutstretched;
      int step;

      if (!tappingDown) {
        // Easing toward outstretched
        if (currentAngles[tapIndex] < tapOutstretched + 25) step = tapStepEase;
        else step = tapStepFast;
      } else {
        // Snappy tap down
        step = tapStepFast;
      }

      // Move toward target
      if (currentAngles[tapIndex] < target) currentAngles[tapIndex] += step;
      else if (currentAngles[tapIndex] > target) currentAngles[tapIndex] -= step;

      // Clamp
      if ((tappingDown && currentAngles[tapIndex] > target) ||
          (!tappingDown && currentAngles[tapIndex] < target)) {
        currentAngles[tapIndex] = target;
      }

      targetAngles[tapIndex] = currentAngles[tapIndex];  // Keep in sync

      if (currentAngles[tapIndex] == target) {
        tapTimer = now;
        isPaused = true;
      }

    } else {
      if (!tappingDown) {
        // Pause at top (outstretched)
        if (now - tapTimer >= tapPauseAtTop) {
          tappingDown = true;
          isPaused = false;
          tapPauseAtTop = random(300, 800); // More variation in pauses
        }
      } else {
        // Minimal pause at bottom
        if (now - tapTimer >= 50) {  // Very brief pause
          tappingDown = false;
          isPaused = false;
        }
      }
    }
  }
}

void updateServoPositions(unsigned long now) {
  // Smooth movement toward targets for all servos
  for (int i = 0; i < NUM_SERVOS; i++) {
    if (consciousnessMode && i == tapIndex) continue;  // Let consciousness control tap finger
    
    if (now - lastUpdate[i] >= speeds[i]) {
      if (currentAngles[i] < targetAngles[i]) currentAngles[i]++;
      else if (currentAngles[i] > targetAngles[i]) currentAngles[i]--;
      
      writeMapped(i, currentAngles[i]);
      lastUpdate[i] = now;
    }
  }
}

void writeMapped(int i, int angle) {
  int mappedAngle = isMirrored[i] ? (180 - angle) : angle;
  servos[i].write(mappedAngle);
}
