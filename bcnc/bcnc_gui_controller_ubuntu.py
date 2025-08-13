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
        """Activate window using xdotool with multiple methods"""
        try:
            # Method 1: Raise and focus the window
            subprocess.run(["xdotool", "windowactivate", "--sync", str(window_id)], check=True)
            time.sleep(0.5)
            
            # Method 2: Also raise it to front
            subprocess.run(["xdotool", "windowraise", str(window_id)], check=True)
            time.sleep(0.5)
            
            # Method 3: Focus it explicitly
            subprocess.run(["xdotool", "windowfocus", str(window_id)], check=True)
            time.sleep(0.5)
            
            return True
        except subprocess.CalledProcessError as e:
            print(f"[DEBUG] xdotool activation failed: {e}")
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
        """Activate window using alternative methods since xwininfo can't activate"""
        print(f"[DEBUG] Försöker aktivera fönster {hex(window_id)}")
        
        # Try xdotool if available
        if shutil.which("xdotool"):
            print("[DEBUG] Använder xdotool för aktivering")
            return LinuxWindowManager.activate_window_xdotool(window_id)
        
        # Try wmctrl if available  
        if shutil.which("wmctrl"):
            print("[DEBUG] Använder wmctrl för aktivering")
            return LinuxWindowManager.activate_window_wmctrl(window_id)
        
        # Last resort: try using xprop to raise the window
        try:
            print("[DEBUG] Försöker med xprop som sista utväg")
            subprocess.run(["xprop", "-id", hex(window_id), "-f", "_NET_ACTIVE_WINDOW", "32a", "-set", "_NET_ACTIVE_WINDOW", "1"], check=True)
            return True
        except subprocess.CalledProcessError:
            pass
        
        print("[DEBUG] Ingen metod fungerade för att aktivera fönster")
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


def get_window_position(window_id):
    """Get window position using xdotool"""
    try:
        result = subprocess.run(
            ["xdotool", "getwindowgeometry", str(window_id)],
            capture_output=True, text=True, check=True
        )
        # Parse output like: "Position: 960,64 (screen: 0)"
        for line in result.stdout.split('\n'):
            if "Position:" in line:
                pos_part = line.split("Position:")[1].split("(")[0].strip()
                x, y = map(int, pos_part.split(','))
                return x + 100, y + 100  # Click a bit inside the window
        return None, None
    except subprocess.CalledProcessError:
        return None, None


def import_svg_in_bcnc(svg_file, output_gcode_file, origin=(0, 0, 0)):
    """Import SVG and export G-code using bCNC - Ubuntu compatible"""
    print("[INFO] Startar import och export i bCNC...")

    if not find_and_activate_bcnc():
        return False

    # Get the specific window that was activated
    bcnc_window_id = None
    for title in ["bCNC", "bcnc", "BCNC", "bCNC 0.9"]:
        if shutil.which("xdotool"):
            window_ids = LinuxWindowManager.find_window_xdotool(title)
            if window_ids:
                bcnc_window_id = window_ids[0]
                break
        elif shutil.which("wmctrl"):
            window_ids = LinuxWindowManager.find_window_wmctrl(title)
            if window_ids:
                bcnc_window_id = window_ids[0]
                break

    if bcnc_window_id:
        # Get window position and click on it
        x, y = get_window_position(bcnc_window_id)
        if x and y:
            print(f"[INFO] Klickar på bCNC-fönster på position ({x}, {y})")
            pyautogui.click(x, y)
            time.sleep(1)
        
        # Force focus using xdotool again
        if shutil.which("xdotool"):
            subprocess.run(["xdotool", "windowfocus", str(bcnc_window_id)], check=False)
            time.sleep(1)

    # Wait longer for window to become active and focused
    print("[INFO] Väntar på att fönstret ska bli aktivt...")
    time.sleep(2)

    # Öppna kommandorad och importera SVG
    print("[INFO] Öppnar kommandorad...")
    pyautogui.hotkey("ctrl", "space")
    time.sleep(1)
    svg_path = svg_file.replace("\\", "/")
    print(f"[INFO] Skriver load-kommando: load {svg_path}")
    pyautogui.write(f"load {svg_path}")
    pyautogui.press("enter")
    time.sleep(5)

    # Markera allt och sätt origin
    print("[INFO] Markerar allt och sätter origin...")
    pyautogui.hotkey("ctrl", "a")
    time.sleep(0.5)
    pyautogui.hotkey("ctrl", "space")
    time.sleep(1)
    origin_cmd = f"origin [{origin[0]}] [{origin[1]}] [{origin[2]}]"
    print(f"[INFO] Skriver origin-kommando: {origin_cmd}")
    pyautogui.write(origin_cmd)
    pyautogui.press("enter")
    time.sleep(2)

    # Spara G-kod
    print("[INFO] Sparar G-kod...")
    pyautogui.hotkey("ctrl", "space")
    time.sleep(1)
    output_path = output_gcode_file.replace("\\", "/")
    save_cmd = f"save {output_path}"
    print(f"[INFO] Skriver save-kommando: {save_cmd}")
    pyautogui.write(save_cmd)
    pyautogui.press("enter")
    time.sleep(3)

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
