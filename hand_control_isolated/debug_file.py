import json
import os

print('Files in movement_recordings:')
for f in os.listdir('movement_recordings'):
    print(f'  {f}')
    
print('\nChecking energized_engaged file:')
with open('movement_recordings/energized_engaged_20250804_185011.json', 'r') as file:
    data = json.load(file)
    print(f'emotion: {data.get("emotion", "MISSING")}')
    print(f'format_version: {data.get("format_version", "MISSING")}')
    print(f'movement_count: {data.get("movement_count", "MISSING")}')
    print(f'has movements: {"movements" in data}')
    print(f'has servo_movements: {"servo_movements" in data}')
    print(f'has markov_chain: {"markov_chain" in data}')
    if 'markov_chain' in data:
        mc = data['markov_chain']
        print(f'markov_chain keys: {list(mc.keys()) if mc else "None"}')
        if mc and 'servo_transitions' in mc:
            print(f'servo_transitions count: {len(mc["servo_transitions"])}')
        else:
            print('No servo_transitions found')
