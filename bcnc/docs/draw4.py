 import os

import time

import pyautogui
import pygetwindow as gw

# === FILVÄGAR ===
base_path = r'C:\Users\Tobia\Tobias_robot'
svg_input = fr'{base_path}\svg_input\original.svg'
raw_gcode = fr'{base_path}\gcode_output\raw.ngc'
converted_gcode = fr'{base_path}\gcode_output\final.ngc'
bcnc_gcode_path = fr'{base_path}\gcode_output\drawing.ngc'

origin_offset = (-40, -40, 0)

# === 1. Importera SVG med kommandorad ===
def import_svg_in_bcnc(svg_file, output_gcode_file, origin=(0, 0, 0)):
    print("[INFO] Startar import och export i bCNC...")

    bcnc_windows = [w for w in gw.getWindowsWithTitle('bCNC') if w.visible]
    if not bcnc_windows:
        print("[FEL] Inget bCNC-fönster hittades.")
        return False

    bcnc_windows[0].activate()
    time.sleep(1)

    # Öppna kommandorad och importera SVG
    pyautogui.hotkey('ctrl', 'space')
    time.sleep(0.5)
    svg_path = svg_file.replace("\\", "/")
    pyautogui.write(f'load {svg_path}')
    pyautogui.press('enter')
    time.sleep(5)

    # Markera allt och sätt origin
    pyautogui.hotkey('ctrl', 'a')
    time.sleep(0.3)
    pyautogui.hotkey('ctrl', 'space')
    time.sleep(0.5)
    pyautogui.write(f'origin [{origin[0]}] [{origin[1]}] [{origin[2]}]')
    pyautogui.press('enter')
    time.sleep(1)

    # Spara G-kod
    pyautogui.hotkey('ctrl', 'space')
    time.sleep(0.5)
    output_path = output_gcode_file.replace("\\", "/")
    pyautogui.write(f'save {output_path}')
    pyautogui.press('enter')
    time.sleep(2)

    return True

# === 2. Konvertera Z-kommandon till servo-kommandon ===
def convert_z_to_servo(input_file, output_file):
    print("[INFO] Konverterar Z-kommandon till servo...")
    current_pen_state = None
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            clean = line.strip()
            if clean.startswith('G0'):
                if current_pen_state != 'up':
                    outfile.write('M3 S40 ; PEN UP\n')
                    current_pen_state = 'up'
                outfile.write(line)
            elif clean.startswith('G1'):
                if current_pen_state != 'down':
                    outfile.write('M3 S50 ; PEN DOWN\n')
                    current_pen_state = 'down'
                outfile.write(line)
            elif 'Z' in clean:
                for part in clean.split():
                    if part.startswith('Z'):
                        try:
                            z = float(part[1:])
                            if z > 0 and current_pen_state != 'up':
                                outfile.write('M3 S30 ; PEN UP\n')
                                current_pen_state = 'up'
                            elif z <= 0 and current_pen_state != 'down':
                                outfile.write('M3 S90 ; PEN DOWN\n')
                                current_pen_state = 'down'
                        except:
                            pass
                outfile.write(line)
            else:
                outfile.write(line)
    print(f"[INFO] Optimerad G-kod sparad: {output_file}")

# === 3. Kör G-koden i bCNC ===
def run_bcnc_gui(gcode_path):
    print("[INFO] Kör G-kod i bCNC...")
    bcnc_windows = [w for w in gw.getWindowsWithTitle('bCNC') if w.visible]
    if not bcnc_windows:
        print("[FEL] bCNC-fönster hittades inte.")
        return

    bcnc_windows[0].activate()
    time.sleep(1)

    gcode_path_fixed = gcode_path.replace("\\", "/")

    pyautogui.hotkey('ctrl', 'space')
    time.sleep(0.3)
    pyautogui.write('cle')
    pyautogui.press('enter')
    time.sleep(0.5)

    pyautogui.hotkey('ctrl', 'space')
    time.sleep(0.3)
    pyautogui.write(f'load {gcode_path_fixed}')
    pyautogui.press('enter')
    time.sleep(2)

    pyautogui.hotkey('ctrl', 'space')
    time.sleep(0.3)
    pyautogui.write('run')
    pyautogui.press('enter')
    print("[INFO] Körning startad.")

# === MAIN ===
def main():
    if not import_svg_in_bcnc(svg_input, raw_gcode, origin_offset):
        print("[FEL] Misslyckades med import eller export.")
        return

    if not os.path.exists(raw_gcode):
        print(f"[FEL] Filen {raw_gcode} finns inte efter export.")
        return

    convert_z_to_servo(raw_gcode, converted_gcode)

    try:
        os.replace(converted_gcode, bcnc_gcode_path)
        print(f"[INFO] Fil kopierad till: {bcnc_gcode_path}")
    except Exception as e:
        print(f"[FEL] Kunde inte kopiera fil: {e}")
        return

    run_bcnc_gui(bcnc_gcode_path)

if __name__ == "__main__":
    main()