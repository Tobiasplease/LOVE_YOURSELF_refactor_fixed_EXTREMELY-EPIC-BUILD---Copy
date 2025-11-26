// Debug version - shows what commands are being received
// Only receives commands, never sends responses

#define PWM_PIN 9

// Current state
int target_brightness = 0;
bool is_flashing = false;
unsigned long flash_start = 0;

// Smooth brightness changes
float smooth_brightness = 0.0;

void setup() {
  // Fast PWM on pin 9
  TCCR1A = _BV(COM1A1) | _BV(WGM11);
  TCCR1B = _BV(WGM13) | _BV(WGM12) | _BV(CS10);
  ICR1 = 255;
  
  pinMode(PWM_PIN, OUTPUT);
  OCR1A = 0;
  
  Serial.begin(9600);
  Serial.println("DEVICE_ID:LIGHTBULB_CONTROLLER");
  Serial.println("Debug lightbulb controller ready - will show received commands");
}

void loop() {
  // Process serial commands
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    
    // DEBUG: Print received command
    Serial.print("Received: ");
    Serial.println(cmd);
    
    if (cmd.startsWith("B:")) {
      // Set brightness from frame diff
      target_brightness = constrain(cmd.substring(2).toInt(), 0, 255);
      Serial.print("Set target brightness to: ");
      Serial.println(target_brightness);
    }
    else if (cmd == "F") {
      // Flash for caption
      is_flashing = true;
      flash_start = millis();
      Serial.println("Flash triggered!");
    }
  }
  
  // Handle flash
  if (is_flashing) {
    if (millis() - flash_start < 1000) {
      OCR1A = 255;  // Full brightness
      return;  // Skip other processing
    } else {
      is_flashing = false;
      Serial.println("Flash ended");
    }
  }
  
  // Smooth brightness transitions (faster response)
  smooth_brightness += (target_brightness - smooth_brightness) * 0.3;
  
  // Use smoothed brightness directly (no mood oscillation)
  int final_brightness = constrain((int)smooth_brightness, 0, 255);
  
  // Set PWM
  OCR1A = final_brightness;
  
  // DEBUG: Show brightness occasionally
  static unsigned long last_debug = 0;
  if (millis() - last_debug > 2000) {  // Every 2 seconds
    Serial.print("PWM: target=");
    Serial.print(target_brightness);
    Serial.print(", smooth=");
    Serial.print(smooth_brightness);
    Serial.print(", final=");
    Serial.print(final_brightness);
    Serial.print(", OCR1A=");
    Serial.println(OCR1A);
    last_debug = millis();
  }
  
  delay(10);  // 100Hz update rate
}