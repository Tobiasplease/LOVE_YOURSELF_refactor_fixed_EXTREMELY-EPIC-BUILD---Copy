#define PWM_PIN 9

// Base variables with smooth easing
int target_base_brightness = 18;
float current_base_brightness = 18.0;
float mood_speed = 0.5;
float mood_randomness = 0.1;
bool is_boosting = false;
unsigned long boost_start_time = 0;
unsigned long boost_duration = 0;
float current_boost = 0.0;
int target_boost = 180;

// Fluctuation variables
unsigned long last_update = 0;
unsigned long next_random_change = 0;
float current_speed_multiplier = 1.0;
float accumulated_time = 0.0;
const int UPDATE_INTERVAL = 2;
const float AMPLITUDE = 8.0;

void setup() {
  // Set high frequency PWM on pin 9 (Timer 1)
  TCCR1A = _BV(COM1A1) | _BV(WGM11);
  TCCR1B = _BV(WGM13) | _BV(WGM12) | _BV(CS10);
  ICR1 = 255;

  pinMode(PWM_PIN, OUTPUT);
  OCR1A = 0;

  Serial.begin(9600);
  Serial.println("Autonomous lightbulb controller ready");
}

void loop() {
  // Handle serial commands
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    
    if (input.startsWith("BASE:")) {
      target_base_brightness = constrain(input.substring(5).toInt(), 18, 255);
      Serial.print("Target base brightness: ");
      Serial.println(target_base_brightness);
    }
    else if (input.startsWith("MOOD:")) {
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
      boost_duration = constrain(input.substring(6).toInt(), 100, 2000);
      boost_start_time = millis();
      is_boosting = true;
      Serial.print("Caption boost for ");
      Serial.print(boost_duration);
      Serial.println("ms");
    }
  }
  
  // Continuous fluctuation updates
  if (millis() - last_update >= UPDATE_INTERVAL) {
    unsigned long now = millis();
    
    // Update random speed multiplier periodically (affects rate, not phase)
    if (now >= next_random_change) {
      float speed_variation = 1.0 + (((random(0, 1000) / 1000.0) - 0.5) * mood_randomness);
      current_speed_multiplier = constrain(speed_variation, 0.7, 1.3);
      next_random_change = now + random(2000, 4000); // Longer intervals for smoother changes
    }
    
    // Calculate time delta and accumulate with current speed
    float delta_seconds = (now - last_update) * 0.001;
    accumulated_time += delta_seconds * mood_speed * current_speed_multiplier;
    
    // Smooth easing towards target base brightness
    float ease_speed = 0.05;
    current_base_brightness += (target_base_brightness - current_base_brightness) * ease_speed;
    
    // Calculate current brightness
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
    
    // Organic breathing curve using accumulated time (no phase jumps)
    float cycle_position = fmod(accumulated_time, 2.0 * PI) / (2.0 * PI);
    
    // Create organic breathing curve with smoother easing
    float breath_curve;
    if (cycle_position < 0.45) {
      // Inhale: cubic ease in for very gradual start
      float t = cycle_position / 0.45;
      breath_curve = t * t * t;
    } else if (cycle_position < 0.55) {
      // Hold: brief plateau
      breath_curve = 1.0;
    } else {
      // Exhale: cubic ease out for very gradual end
      float t = (cycle_position - 0.55) / 0.45;
      breath_curve = 1.0 - (t * t * t);
    }
    
    // Apply organic fluctuation
    float fluctuation = AMPLITUDE * (breath_curve * 2.0 - 1.0);
    int final_pwm = constrain(working_base + fluctuation, 18, 255);
    
    OCR1A = final_pwm;
    last_update = now;
  }
}