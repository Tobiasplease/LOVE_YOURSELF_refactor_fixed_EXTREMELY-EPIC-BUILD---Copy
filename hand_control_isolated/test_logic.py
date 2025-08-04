import json

# Load the file and test the logic
with open('movement_recordings/energized_engaged_20250804_185011.json', 'r') as file:
    data = json.load(file)

format_version = data.get('format_version', 'unknown')
markov_chain = data.get('markov_chain', {})

print(f"format_version: '{format_version}'")
print(f"markov_chain exists: {bool(markov_chain)}")
if markov_chain:
    print(f"servo_transitions in markov_chain: {'servo_transitions' in markov_chain}")
print(f"movements in data: {'movements' in data}")
print(f"servo_movements in data: {'servo_movements' in data}")
print(f"movement_count in data: {'movement_count' in data}")

# Test the conditions
is_current_format = (format_version == '2.0_servo_based' and 
                  'servo_transitions' in markov_chain and
                  'servo_movements' in data)

is_compatible_old_format = (format_version == 'unknown' and
                          'servo_transitions' in markov_chain and
                          'movements' in data)

is_recent_format = (format_version == 'unknown' and
                  'movement_count' in data and
                  ('movements' in data or 'servo_movements' in data))

print(f"\nis_current_format: {is_current_format}")
print(f"is_compatible_old_format: {is_compatible_old_format}")
print(f"is_recent_format: {is_recent_format}")
print(f"Should load: {is_current_format or is_compatible_old_format or is_recent_format}")
