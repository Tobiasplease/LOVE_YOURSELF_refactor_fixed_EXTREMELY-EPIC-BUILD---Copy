#!/usr/bin/env python3
"""
Test Temporal Consciousness System
"""

import tempfile
from PIL import Image
from captioner.model_wrapper import MultimodalModel
from captioner.captioner import Captioner

print('=== TESTING TEMPORAL GROUNDING ===')

cap = Captioner()
cap.first_caption_done = True
cap.current_emotion_state = 'alert_curious'
model = MultimodalModel(memory_ref=cap)

# First: Red
img1 = Image.new('RGB', (100, 100), color='red')
with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
    img1.save(f.name)
    temp_path1 = f.name

print('1. RED (first):')
result1 = model.caption_image(temp_path1, flowing=True, first_time=False)
print(f'   "{result1}"')
cap.observe(result1, 0.6, temp_path1)

# Second: Blue (with memory of red)
img2 = Image.new('RGB', (100, 100), color='blue')
with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
    img2.save(f.name)
    temp_path2 = f.name

print('\n2. BLUE (after red):')
result2 = model.caption_image(temp_path2, flowing=True, first_time=False)
print(f'   "{result2}"')

# Check what context was built
from captioner.prompts import build_caption_prompt
prompt = build_caption_prompt(cap, 0.6, 0.2, 0.8)
print('\nCONTEXT SENT TO AI:')
lines = prompt.split('\n')
for line in lines:
    if 'Just before:' in line or 'Feeling:' in line:
        print(f'   {line.strip()}')

import os
os.unlink(temp_path1)
os.unlink(temp_path2)

print('\n✅ Temporal consciousness test completed!')
