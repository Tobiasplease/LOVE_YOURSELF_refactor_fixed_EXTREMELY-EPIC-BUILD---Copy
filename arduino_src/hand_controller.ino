#include <Servo.h>

const int NUM_SERVOS = 4;
const int pins[NUM_SERVOS] = {8, 9, 10, 11};
const bool isMirrored[NUM_SERVOS] = {false, false, true, true};

Servo servos[NUM_SERVOS];
int currentAngles[NUM_SERVOS];
int targetAngles[NUM_SERVOS];
unsigned long lastUpdate[NUM_SERVOS];
int speeds[NUM_SERVOS];

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
 
  delay(1000);
  Serial.println("Ready for consciousness commands (0-180 degree range)");
}

void loop() {
  unsigned long now = millis();
 
  // ONLY handle serial commands - no autonomous behavior whatsoever
  handleSerialInput();
 
  // ONLY update servo positions toward targets set by consciousness
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
   
    // Set targets for pure consciousness control
    for (int i = 0; i < NUM_SERVOS; i++) {
      int newTarget = constrain(fingerPositions[i], minAngle, maxAngle);
      targetAngles[i] = newTarget;
     
      // Adaptive servo speeds based on movement distance
      int distance = abs(newTarget - currentAngles[i]);
     
      // Check if this looks like a startle reaction (large simultaneous movement)
      bool isStartleMovement = false;
      if (distance > 25) {  // Increased threshold for larger range (was 15)
        int totalMovement = 0;
        for (int j = 0; j < NUM_SERVOS; j++) {
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
  else if (command.startsWith("HEARTBEAT")) {
    // Acknowledge heartbeat but don't do anything special
    // This is purely for Python's peace of mind
    Serial.println("Heartbeat acknowledged");
  }
}

void updateServoPositions(unsigned long now) {
  // Simple, clean servo movement toward targets
  for (int i = 0; i < NUM_SERVOS; i++) {
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

void writeMapped(int i, int angle) {
  int mappedAngle = isMirrored[i] ? (180 - angle) : angle;
  servos[i].write(mappedAngle);
}