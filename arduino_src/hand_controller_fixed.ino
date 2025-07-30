/*
 * Hand Controller for AI Consciousness System - FIXED VERSION
 * 
 * Receives commands from Python via serial (COM3) and controls 4 finger servos
 * Fixed: Consciousness commands now directly control ALL fingers including the tap finger
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
const int minSpeed = 15;
const int maxSpeed = 35;

// Consciousness control state
bool consciousnessMode = false;
unsigned long lastConsciousnessCommand = 0;
const unsigned long consciousnessTimeout = 5000; // 5 seconds

// Your original tapping logic (for fallback/idle behavior)
const int tapIndex = 3; // pin 11 = finger D
int tapCurled = 130;      
int tapOutstretched = 10; 
int tapStepFast = 2;
int tapStepEase = 2;

unsigned long tapPauseAtTop = random(300, 600);
unsigned long tapTimer = 0;
bool tappingDown = true;
bool isPaused = false;
unsigned long tapUpdateTimer = 0;
const unsigned long tapUpdateInterval = 15;

String inputBuffer = "";

void setup() {
  Serial.begin(9600);
  Serial.println("Hand Controller Ready - Consciousness Fixed");
  
  // Initialize servos
  for (int i = 0; i < NUM_SERVOS; i++) {
    servos[i].attach(pins[i]);
    currentAngles[i] = 70;  // Start at neutral
    targetAngles[i] = 70;
    lastUpdate[i] = 0;
    speeds[i] = random(minSpeed, maxSpeed);
    writeMapped(i, currentAngles[i]);
  }
  
  delay(1000);
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
  
  // Update servo positions (now properly handles consciousness mode)
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
    
    // FIXED: Set targets for consciousness physics system
    for (int i = 0; i < NUM_SERVOS; i++) {
      int newTarget = constrain(fingerPositions[i], minAngle, maxAngle);
      targetAngles[i] = newTarget;
      
      // Adaptive servo speeds based on movement distance
      int distance = abs(newTarget - currentAngles[i]);
      
      // Check if this looks like a startle reaction (large simultaneous movement)
      bool isStartleMovement = false;
      if (distance > 15) {  // Large movement on this finger
        int totalMovement = 0;
        for (int j = 0; j < NUM_SERVOS; j++) {
          totalMovement += abs(fingerPositions[j] - currentAngles[j]);
        }
        // If total movement across all fingers is large, it's likely a startle
        if (totalMovement > 60) {
          isStartleMovement = true;
        }
      }
      
      if (isStartleMovement) {
        speeds[i] = 1;   // ULTRA VIOLENT: 1ms updates + 8 degree steps = extreme speed!
        Serial.print("VIOLENT STARTLE detected! Finger ");
        Serial.print(i);
        Serial.print(" speed=1ms, target=");
        Serial.println(targetAngles[i]);
      } else if (distance > 20) {
        speeds[i] = 8;   // Fast for large movements
      } else if (distance > 5) {
        speeds[i] = 12;  // Medium speed for medium movements  
      } else {
        speeds[i] = 20;  // Slower for fine adjustments - prevents "kikiki" sound
      }
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

  // Handle tapping finger logic
  if (now - tapUpdateTimer >= tapUpdateInterval) {
    tapUpdateTimer = now;
    
    if (!isPaused) {
      int target = tappingDown ? tapCurled : tapOutstretched;
      int step = tapStepFast;
      
      if (!tappingDown) {
        if (currentAngles[tapIndex] < tapOutstretched + 25) step = tapStepEase;
      }

      // Move toward target
      if (currentAngles[tapIndex] < target) currentAngles[tapIndex] += step;
      else if (currentAngles[tapIndex] > target) currentAngles[tapIndex] -= step;

      // Clamp
      if ((tappingDown && currentAngles[tapIndex] > target) ||
          (!tappingDown && currentAngles[tapIndex] < target)) {
        currentAngles[tapIndex] = target;
      }

      targetAngles[tapIndex] = currentAngles[tapIndex];

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
          tapPauseAtTop = random(300, 800);
        }
      } else {
        // Minimal pause at bottom
        if (now - tapTimer >= 50) {
          tappingDown = false;
          isPaused = false;
        }
      }
    }
  }
}

void updateServoPositions(unsigned long now) {
  // FIXED: Smooth movement for ALL servos, with special handling for tap finger
  for (int i = 0; i < NUM_SERVOS; i++) {
    // In autonomous mode, tap finger is handled by runAutonomousBehavior
    // In consciousness mode, ALL fingers (including tap finger) use smooth movement
    if (!consciousnessMode && i == tapIndex) {
      // Tap finger movement is handled in runAutonomousBehavior, just write current position
      writeMapped(i, currentAngles[i]);
      continue;
    }
    
    if (now - lastUpdate[i] >= speeds[i]) {
      // VIOLENT STARTLE MOVEMENT: Multiple degrees per step for speed 1-3
      int stepSize = 1;  // Default smooth movement
      if (speeds[i] <= 3) {
        stepSize = 8;  // VIOLENT: 8 degrees per step for startle!
      } else if (speeds[i] <= 10) {
        stepSize = 3;  // Fast: 3 degrees per step
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

void writeMapped(int i, int angle) {
  int mappedAngle = isMirrored[i] ? (180 - angle) : angle;
  servos[i].write(mappedAngle);
}
