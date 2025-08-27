#define PWM_PIN 9

// Base variables with smooth easing
int target_base_brightness = 8;
float current_base_brightness = 8.0; // Smooth float value
float mood_speed = 0.5;
float mood_randomness = 0.1;
bool is_boosting = false;
unsigned long boost_end_time = 0;
int boost_brightness = 0;

// Fluctuation variables
unsigned long last_update = 0;
unsigned long next_random_change = 0;
float current_speed_offset = 0;
const int UPDATE_INTERVAL = 2; // ~500Hz updates (2ms) - ultra smooth
const float AMPLITUDE = 3.0; // Fixed subtle amplitude

void setup() {
  // Set high frequency PWM on pin 9 (Timer 1) - eliminates flicker
  TCCR1A = _BV(COM1A1) | _BV(WGM11);
  TCCR1B = _BV(WGM13) | _BV(WGM12) | _BV(CS10);
  ICR1 = 255;

  pinMode(PWM_PIN, OUTPUT);
  OCR1A = 0;  // Start completely OFF - this fixes the bright default

  Serial.begin(9600);
  Serial.println("Autonomous lightbulb controller ready...");
}

void loop() {
  // Handle serial commands
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    
    if (input.startsWith("BASE:")) {
      // Set target base brightness - will ease smoothly to this value
      target_base_brightness = constrain(input.substring(5).toInt(), 8, 255);
      Serial.print("Target base brightness: ");
      Serial.println(target_base_brightness);
    }
    else if (input.startsWith("MOOD:")) {
      // Format: MOOD:speed:randomness
      int colonPos = input.indexOf(':', 5);
      if (colonPos > 0) {
        mood_speed = constrain(input.substring(5, colonPos).toFloat(), 0.1, 2.0);
        mood_randomness = constrain(input.substring(colonPos + 1).toFloat(), 0, 1);
        Serial.print("Mood updated - Speed: ");
        Serial.print(mood_speed);
        Serial.print(", Randomness: ");
        Serial.println(mood_randomness);
      }
    }
    else if (input.startsWith("BOOST:")) {
      // Caption brightness boost
      int duration = constrain(input.substring(6).toInt(), 100, 2000);
      boost_brightness = 180;
      boost_end_time = millis() + duration;
      is_boosting = true;
      Serial.print("Caption boost for ");
      Serial.print(duration);
      Serial.println("ms");
    }
  }
  
  // Continuous fluctuation updates at 60Hz
  if (millis() - last_update >= UPDATE_INTERVAL) {
    unsigned long now = millis();
    
    // Update random speed offset periodically (every 1-3 seconds)
    if (now >= next_random_change) {
      current_speed_offset = ((random(0, 1000) / 1000.0) - 0.5) * mood_randomness;
      next_random_change = now + random(1000, 3000);
    }
    
    float effective_speed = mood_speed + current_speed_offset;
    effective_speed = constrain(effective_speed, 0.1, 2.0);
    
    // Smooth easing towards target base brightness (exponential ease)
    float ease_speed = 0.05; // Adjust for faster/slower easing (0.01 = slow, 0.1 = fast)
    current_base_brightness += (target_base_brightness - current_base_brightness) * ease_speed;
    
    // Calculate current brightness
    float working_base = current_base_brightness;
    
    // Handle caption boost
    if (is_boosting) {
      if (now >= boost_end_time) {
        is_boosting = false;
      } else {
        working_base = boost_brightness;
      }
    }
    
    // Organic breathing curve instead of rigid sine wave
    float time_factor = now * 0.001 * effective_speed;
    float cycle_position = fmod(time_factor, 2.0 * PI) / (2.0 * PI); // 0 to 1 cycle position
    
    // Create organic breathing curve with easing
    float breath_curve;
    if (cycle_position < 0.4) {
      // Inhale: ease in (slow start, accelerate)
      float t = cycle_position / 0.4;
      breath_curve = t * t; // quadratic ease in
    } else if (cycle_position < 0.6) {
      // Hold: slight plateau
      breath_curve = 1.0;
    } else {
      // Exhale: ease out (start fast, decelerate)  
      float t = (cycle_position - 0.6) / 0.4;
      breath_curve = 1.0 - (t * t); // quadratic ease out
    }
    
    // Apply organic fluctuation
    float fluctuation = AMPLITUDE * (breath_curve * 2.0 - 1.0); // -1 to 1 range
    int final_pwm = constrain(working_base + fluctuation, 8, 255);
    
    OCR1A = final_pwm;
    last_update = now;
  }
}