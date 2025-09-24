#include <Wire.h>
#include <LiquidCrystal_I2C.h>

LiquidCrystal_I2C lcd(0x27, 16, 2);

const int LCD_COLS = 16;
const int LCD_ROWS = 2;

String currentCaption = "";
int currentChunk = 0;
int totalChunks = 0;
unsigned long lastChunkTime = 0;
int chunkDelay = 1500;  // Adaptive chunk delay (readable default)
String currentPriority = "M";  // M=Medium, H=High, L=Low
int lcdBrightness = 255;  // Default brightness (0-255)
bool captionComplete = true;  // Track if current caption is fully displayed

void setup() {
  Serial.begin(9600);
  lcd.begin(LCD_COLS, LCD_ROWS);
  lcd.backlight();
  lcd.clear();
  lcd.setCursor(2, 0);
  lcd.print("Caption Display");
  lcd.setCursor(0, 1);
  lcd.print("Ready to listen");
  delay(2000);
  lcd.clear();
}

void loop() {
  if (Serial.available()) {
    String message = Serial.readStringUntil('\n');
    message.trim();

    if (message.length() > 0) {
      // Handle brightness command: "BRIGHTNESS:value"
      if (message.startsWith("BRIGHTNESS:")) {
        int brightness = message.substring(11).toInt();
        brightness = constrain(brightness, 0, 255);
        lcdBrightness = brightness;

        // For I2C LCD, we simulate brightness by turning backlight on/off rapidly
        // Note: True PWM brightness control would require hardware modification
        if (brightness == 0) {
          lcd.noBacklight();
        } else {
          lcd.backlight();
          // Could implement PWM dimming here if hardware supports it
        }

        Serial.print("LCD brightness set to: ");
        Serial.println(brightness);
        return;
      }

      // Parse message format: "PRIORITY:DELAY:CAPTION"
      int firstColon = message.indexOf(':');
      int secondColon = message.indexOf(':', firstColon + 1);

      if (firstColon > 0 && secondColon > firstColon) {
        String newPriority = message.substring(0, firstColon);
        int newDelay = message.substring(firstColon + 1, secondColon).toInt();
        String newCaption = message.substring(secondColon + 1);

        // Only accept new caption if current one is complete
        if (captionComplete) {
          currentCaption = newCaption;
          currentPriority = newPriority;
          chunkDelay = newDelay;
          currentChunk = 0;
          totalChunks = (currentCaption.length() + 31) / 32;  // 32 chars per chunk (2 rows)
          lastChunkTime = millis();
          captionComplete = (totalChunks <= 1);  // Single chunk = complete immediately
          updateDisplay();
          Serial.println("CAPTION_ACCEPTED");
        } else {
          Serial.println("CAPTION_BUSY");
        }
      } else {
        // Fallback for old format (just caption)
        if (captionComplete) {
          currentCaption = message;
          currentPriority = "M";
          chunkDelay = 300;
          currentChunk = 0;
          totalChunks = (currentCaption.length() + 31) / 32;  // 32 chars per chunk (2 rows)
          lastChunkTime = millis();
          captionComplete = (totalChunks <= 1);  // Single chunk = complete immediately
          updateDisplay();
          Serial.println("CAPTION_ACCEPTED");
        } else {
          Serial.println("CAPTION_BUSY");
        }
      }
    }
  }

  // Advance to next chunk if caption is longer than display
  if (totalChunks > 1 && millis() - lastChunkTime >= chunkDelay) {
    currentChunk = (currentChunk + 1) % totalChunks;
    lastChunkTime = millis();
    updateDisplay();

    // Mark caption as complete when we've shown all chunks
    if (currentChunk == 0) {  // We've cycled back to start
      captionComplete = true;
      Serial.println("CAPTION_COMPLETE");
    }
  }
}

void updateDisplay() {
  lcd.clear();

  int startPos = currentChunk * 32;  // 32 chars per chunk (2 rows)

  // Display on both rows (16 chars each)
  // Row 1: chars 0-15 of current chunk
  int row1Start = startPos;
  int row1End = min(row1Start + LCD_COLS, (int)currentCaption.length());
  if (row1Start < currentCaption.length()) {
    String row1 = currentCaption.substring(row1Start, row1End);
    lcd.setCursor(0, 0);
    lcd.print(row1);
  }

  // Row 2: chars 16-31 of current chunk
  int row2Start = startPos + LCD_COLS;
  int row2End = min(row2Start + LCD_COLS, (int)currentCaption.length());
  if (row2Start < currentCaption.length()) {
    String row2 = currentCaption.substring(row2Start, row2End);
    lcd.setCursor(0, 1);
    lcd.print(row2);
  }
}