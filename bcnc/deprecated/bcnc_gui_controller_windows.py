import os
import time
import pygetwindow as gw
import pyautogui
from bcnc_utils import convert_z_to_servo, try_bcnc_cli_run

# === FILVÄGAR ===
# base_path = r"C:\Users\Tobia\Tobias_robot"
base_path = "/home/jbe/Dropbox/_outputs"

svg_input = f"{base_path}/impostor-20250725_185854_00001_.png.svg"
raw_gcode = f"{base_path}/raw.ngc"
converted_gcode = f"{base_path}/final.ngc"
bcnc_gcode_path = rf"{base_path}/drawing.ngc"

origin_offset = (-40, -40, 0)


# === 1. Importera SVG med kommandorad ===
def import_svg_in_bcnc(svg_file, output_gcode_file, origin=(0, 0, 0)):
    print("[INFO] Startar import och export i bCNC...")

    bcnc_windows = [w for w in gw.getWindowsWithTitle("bCNC") if w.visible]  # type: ignore
    if not bcnc_windows:
        print("[FEL] Inget bCNC-fönster hittades.")
        return False

    bcnc_windows[0].activate()
    time.sleep(1)

    # Öppna kommandorad och importera SVG
    pyautogui.hotkey("ctrl", "space")
    time.sleep(0.5)
    svg_path = svg_file.replace("\\", "/")
    pyautogui.write(f"load {svg_path}")
    pyautogui.press("enter")
    time.sleep(5)

    # Markera allt och sätt origin
    pyautogui.hotkey("ctrl", "a")
    time.sleep(0.3)
    pyautogui.hotkey("ctrl", "space")
    time.sleep(0.5)
    pyautogui.write(f"origin [{origin[0]}] [{origin[1]}] [{origin[2]}]")
    pyautogui.press("enter")
    time.sleep(1)

    # Spara G-kod
    pyautogui.hotkey("ctrl", "space")
    time.sleep(0.5)
    output_path = output_gcode_file.replace("\\", "/")
    pyautogui.write(f"save {output_path}")
    pyautogui.press("enter")
    time.sleep(2)

    return True


# === 2. Konvertera Z-kommandon till servo-kommandon ===


# === 3. Kör G-koden i bCNC ===
def run_bcnc_gui(gcode_path):
    print("[INFO] Kör G-kod i bCNC...")
    bcnc_windows = [w for w in gw.getWindowsWithTitle("bCNC") if w.visible]  # type: ignore
    if not bcnc_windows:
        print("[FEL] bCNC-fönster hittades inte.")
        return

    bcnc_windows[0].activate()
    time.sleep(1)

    gcode_path_fixed = gcode_path.replace("\\", "/")

    pyautogui.hotkey("ctrl", "space")
    time.sleep(0.3)
    pyautogui.write("cle")
    pyautogui.press("enter")
    time.sleep(0.5)

    pyautogui.hotkey("ctrl", "space")
    time.sleep(0.3)
    pyautogui.write(f"load {gcode_path_fixed}")
    pyautogui.press("enter")
    time.sleep(2)

    pyautogui.hotkey("ctrl", "space")
    time.sleep(0.3)
    pyautogui.write("run")
    pyautogui.press("enter")
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
