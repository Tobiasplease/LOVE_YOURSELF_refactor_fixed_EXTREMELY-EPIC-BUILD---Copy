import os
import time
import subprocess
import pyautogui
import shutil

# === FILVÄGAR ===
base_path = "/home/jbe/Dropbox/_outputs"

svg_input = f"{base_path}/impostor-20250725_185854_00001_.png.svg"
raw_gcode = f"{base_path}/raw.ngc"
converted_gcode = f"{base_path}/final.ngc"
bcnc_gcode_path = f"{base_path}/drawing.ngc"

origin_offset = (-40, -40, 0)


class LinuxWindowManager:
    """Handle window management on Linux with multiple fallback methods"""

    @staticmethod
    def check_dependencies():
        """Check available window management tools"""
        tools = {"xdotool": shutil.which("xdotool"), "wmctrl": shutil.which("wmctrl"), "xwininfo": shutil.which("xwininfo")}
        available = [name for name, path in tools.items() if path]

        if not available:
            raise RuntimeError("No window management tools found. Install with:\n" "sudo apt install xdotool wmctrl x11-utils")

        print(f"[INFO] Tillgängliga verktyg: {', '.join(available)}")
        return available[0]  # Return preferred tool

    @staticmethod
    def find_window_xdotool(title):
        """Find window using xdotool - search for titles that START with the given string"""
        try:
            # Search for windows where title starts with the given string
            result = subprocess.run(["xdotool", "search", "--name", f"^{title}"], capture_output=True, text=True, check=False)
            if result.returncode == 0 and result.stdout.strip():
                return [int(wid) for wid in result.stdout.strip().split("\n") if wid]
        except Exception:
            pass
        return []

    @staticmethod
    def find_window_wmctrl(title):
        """Find window using wmctrl - search for titles that START with the given string"""
        try:
            result = subprocess.run(["wmctrl", "-l"], capture_output=True, text=True, check=False)
            if result.returncode == 0:
                windows = []
                for line in result.stdout.split("\n"):
                    # Check if window title starts with the given string
                    parts = line.split(None, 4)  # Split into max 5 parts to preserve title
                    if len(parts) >= 5:
                        window_title = parts[4].lower()
                        if window_title.startswith(title.lower()):
                            try:
                                windows.append(int(parts[0], 16))  # wmctrl uses hex
                            except ValueError:
                                continue
                return windows
        except Exception:
            pass
        return []

    @staticmethod
    def activate_window_xdotool(window_id):
        """Activate window using xdotool"""
        try:
            subprocess.run(["xdotool", "windowactivate", str(window_id)], check=True)
            return True
        except subprocess.CalledProcessError:
            return False

    @staticmethod
    def activate_window_wmctrl(window_id):
        """Activate window using wmctrl"""
        try:
            subprocess.run(["wmctrl", "-i", "-a", hex(window_id)], check=True)
            return True
        except subprocess.CalledProcessError:
            return False

    @staticmethod
    def find_window_xwininfo(title):
        """Find window using xwininfo - search for titles that START with the given string"""
        try:
            # Get list of all window IDs
            result = subprocess.run(["xwininfo", "-root", "-tree"], capture_output=True, text=True, check=False)
            if result.returncode == 0:
                windows = []
                # Parse window IDs from the tree output
                import re

                for line in result.stdout.split("\n"):
                    # Look for pattern: 0x123456 "Window Title":
                    # bCNC 0.9.16 (linux py3.10.12)
                    match = re.search(r'(0x[0-9a-f]+)\s+"([^"]+)":', line)
                    if match:
                        window_id = match.group(1)
                        window_title = match.group(2)
                        # Check if window title starts with our search term
                        if window_title.lower().startswith(title.lower()):
                            try:
                                wid = int(window_id, 16)
                                # Filter out tiny windows (likely dialogs/tooltips)
                                if "960x1016" in line or not re.search(r"\b1x1\b", line):
                                    windows.append(wid)
                                    print(f"[DEBUG] Hittade matchande fönster: {window_title} (ID: {window_id})")
                            except ValueError:
                                continue
                return windows
        except Exception:
            pass
        return []

    @staticmethod
    def activate_window_xwininfo(window_id):
        """Activate window using xdotool (xwininfo can't activate)"""
        # xwininfo can't activate, so fall back to xdotool if available
        if shutil.which("xdotool"):
            return LinuxWindowManager.activate_window_xdotool(window_id)
        return False

    @staticmethod
    def debug_list_all_windows():
        """Debug function to list all visible windows"""
        print("[DEBUG] Listar alla fönster:")

        # Try wmctrl first
        if shutil.which("wmctrl"):
            try:
                result = subprocess.run(["wmctrl", "-l"], capture_output=True, text=True, check=False)
                if result.returncode == 0:
                    for line in result.stdout.split("\n"):
                        if line.strip():
                            print(f"[DEBUG] wmctrl: {line}")
                    return
            except Exception:
                pass

        # Try xwininfo as fallback
        if shutil.which("xwininfo"):
            try:
                result = subprocess.run(["xwininfo", "-root", "-tree"], capture_output=True, text=True, check=False)
                if result.returncode == 0:
                    for line in result.stdout.split("\n"):
                        if "bCNC" in line or "bcnc" in line.lower():
                            print(f"[DEBUG] xwininfo: {line}")
            except Exception:
                pass

    @staticmethod
    def find_and_activate_window(title):
        """Try multiple methods to find and activate window"""
        methods = [
            ("xdotool", LinuxWindowManager.find_window_xdotool, LinuxWindowManager.activate_window_xdotool),
            ("wmctrl", LinuxWindowManager.find_window_wmctrl, LinuxWindowManager.activate_window_wmctrl),
            ("xwininfo", LinuxWindowManager.find_window_xwininfo, LinuxWindowManager.activate_window_xwininfo),
        ]

        for tool_name, find_func, activate_func in methods:
            if shutil.which(tool_name):
                window_ids = find_func(title)
                if window_ids:
                    for wid in window_ids:
                        if activate_func(wid):
                            print(f"[INFO] Aktiverade fönster {wid} med {tool_name}")
                            return True

        return False


def find_and_activate_bcnc():
    """Find and activate bCNC window - Ubuntu compatible with fallbacks"""
    try:
        LinuxWindowManager.check_dependencies()
    except RuntimeError as e:
        print(f"[FEL] {e}")
        return False

    # Debug: List all windows to see what's available
    LinuxWindowManager.debug_list_all_windows()

    # Try different possible window titles - including version numbers
    possible_titles = ["bCNC", "bcnc", "BCNC", "bCNC 0.9"]

    for title in possible_titles:
        print(f"[INFO] Söker efter fönster med titel: {title}")
        if LinuxWindowManager.find_and_activate_window(title):
            time.sleep(1)  # Wait for window to activate
            return True

    print("[FEL] Inget bCNC-fönster hittades.")
    print("[TIPS] Kontrollera att bCNC är öppet och synligt.")
    return False


def import_svg_in_bcnc(svg_file, output_gcode_file, origin=(0, 0, 0)):
    """Import SVG and export G-code using bCNC - Ubuntu compatible"""
    print("[INFO] Startar import och export i bCNC...")

    if not find_and_activate_bcnc():
        return False

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


def convert_z_to_servo(input_file, output_file):
    """Convert Z commands to servo commands - same as original"""
    print("[INFO] Konverterar Z-kommandon till servo...")
    current_pen_state = None
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            clean = line.strip()
            if clean.startswith("G0"):
                if current_pen_state != "up":
                    outfile.write("M3 S40 ; PEN UP\n")
                    current_pen_state = "up"
                outfile.write(line)
            elif clean.startswith("G1"):
                if current_pen_state != "down":
                    outfile.write("M3 S50 ; PEN DOWN\n")
                    current_pen_state = "down"
                outfile.write(line)
            elif "Z" in clean:
                for part in clean.split():
                    if part.startswith("Z"):
                        try:
                            z = float(part[1:])
                            if z > 0 and current_pen_state != "up":
                                outfile.write("M3 S30 ; PEN UP\n")
                                current_pen_state = "up"
                            elif z <= 0 and current_pen_state != "down":
                                outfile.write("M3 S90 ; PEN DOWN\n")
                                current_pen_state = "down"
                        except ValueError:
                            print(f"[FEL] Kunde inte konvertera Z-värde: {part}")
                            pass
                outfile.write(line)
            else:
                outfile.write(line)
    print(f"[INFO] Optimerad G-kod sparad: {output_file}")


def run_bcnc_gui(gcode_path):
    """Run G-code in bCNC - Ubuntu compatible"""
    print("[INFO] Kör G-kod i bCNC...")

    if not find_and_activate_bcnc():
        return

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


def main():
    """Main function - Ubuntu compatible"""
    try:
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

    except RuntimeError as e:
        print(f"[FEL] Systemkrav: {e}")
        print("Installera med: sudo apt install xdotool")
    except Exception as e:
        print(f"[FEL] Oväntat fel: {e}")


if __name__ == "__main__":
    main()
