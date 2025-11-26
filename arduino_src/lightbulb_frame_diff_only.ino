// Simple lightbulb controller - frame diff only
// Expects BASE:<value> commands for brightness
// Supports BOOST:<duration> for caption flash

#define PWM_PIN 9

// State variables
int target_base_brightness = 0;  // Start at 0, not 18
float current_base_brightness = 0.0;
bool is_boosting = false;
unsigned long boost_start_time = 0;
unsigned long boost_duration = 0;
float current_boost = 0.0;
int target_boost = 255;  // Full brightness for caption flash

void setup() {
  // Set high frequency PWM on pin 9 (Timer 1)
  TCCR1A = _BV(COM1A1) | _BV(WGM11);
  TCCR1B = _BV(WGM13) | _BV(WGM12) | _BV(CS10);
  ICR1 = 255;

  pinMode(PWM_PIN, OUTPUT);
  OCR1A = 0;  // Start at 0

  Serial.begin(9600);
  Serial.println("Frame diff lightbulb controller ready");
}

void loop() {
  // Handle serial commands
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    
    if (input.startsWith("BASE:")) {
      // Set base brightness from frame diff (0-255)
      target_base_brightness = constrain(input.substring(5).toInt(), 0, 255);
      Serial.print("Target base brightness: ");
      Serial.println(target_base_brightness);
    }
    else if (input.startsWith("BOOST:")) {
      // Caption boost flash
      boost_duration = constrain(input.substring(6).toInt(), 100, 2000);
      boost_start_time = millis();
      is_boosting = true;
      Serial.print("Caption boost for ");
      Serial.print(boost_duration);
      Serial.println("ms");
    }
  }
  
  // Update every 10ms for smooth response
  static unsigned long last_update = 0;
  if (millis() - last_update >= 10) {
    unsigned long now = millis();
    
    // Very slow easing for gradual changes
    float ease_speed = 0.01;  // Much slower for smooth transitions
    current_base_brightness += (target_base_brightness - current_base_brightness) * ease_speed;
    
    float working_base = current_base_brightness;
    
    // Handle smooth caption boost
    if (is_boosting) {
      unsigned long elapsed = now - boost_start_time;
      if (elapsed >= boost_duration) {
        is_boosting = false;
        current_boost = 0.0;
      } else {
        // Create smooth ease-in-out curve for boost
        float progress = (float)elapsed / (float)boost_duration; // 0 to 1
        float ease_curve;
        if (progress < 0.5) {
          // Ease in: accelerate to peak
          ease_curve = 2.0 * progress * progress;
        } else {
          // Ease out: decelerate from peak
          float t = 1.0 - progress;
          ease_curve = 1.0 - 2.0 * t * t;
        }
        current_boost = target_boost * ease_curve;
        working_base = current_boost; // Override base during boost
      }
    }
    
    // Set final PWM (no mood oscillation, no artificial minimum)
    int final_pwm = constrain((int)working_base, 0, 255);
    
    OCR1A = final_pwm;
    last_update = now;
  }
}