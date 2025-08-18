# setup_grbl_grid.py

import time
import serial
from serial.tools import list_ports

# ======= Konfiguration =======
BAUD = 115200
STATUS_POLL = 0.2
HOME_TIMEOUT = 300
MOVE_TIMEOUT = 500

# Din önskade arbets-nolla relativt HOME:
ORIGIN_X = 0  # +66 mm från HOME i X
ORIGIN_Y = 0  # -2 mm från HOME i Y
ORIGIN_Z = 0.0

# Grid settings
GRID_SIZE = 60  # 100mm x 100mm grid
GRID_SPACING = 10  # 10mm spacing between lines
PEN_DOWN_CMD = "M3 S50"  # Command to lower pen
PEN_UP_CMD = "M3 S30"  # Command to raise pen
FEED_RATE = 3000
# =============================


def find_grbl_port(baud=BAUD, timeout=0.5):
    ports = list(list_ports.comports())
    if not ports:
        raise RuntimeError("Hittar inga seriella portar.")
    print("[INFO] Tillgängliga portar:", ", ".join(p.device for p in ports))
    for p in ports:
        try:
            print(f"[INFO] Testar {p.device} ...")
            ser = serial.Serial(p.device, baud, timeout=timeout)
            time.sleep(2.0)
            ser.reset_input_buffer()
            ser.write(b"?")
            ser.flush()
            line = ser.readline().decode(errors="ignore").strip()
            if line.startswith("<") or "Grbl" in line:
                print(f"[INFO] {p.device} svarar som GRBL: {line}")
                return ser
            ser.close()
        except Exception as e:
            print(f"[VARNING] {p.device} fungerade inte ({e}).")
    raise RuntimeError("Hittade ingen GRBL-port.")


def read_until_ok_or_error(ser, timeout=5.0):
    start = time.time()
    log = []
    last = None
    while time.time() - start < timeout:
        line = ser.readline().decode(errors="ignore").strip()
        if line:
            log.append(line)
            lower_case_line = line.lower()
            if lower_case_line == "ok" or lower_case_line.startswith("error"):
                last = line
                break
    return last, log


def send_cmd(ser, cmd, wait_ok=True, timeout=5.0):
    print(f"[SEND] {cmd}")
    ser.write((cmd + "\n").encode())
    ser.flush()
    if not wait_ok:
        return []
    last, log = read_until_ok_or_error(ser, timeout)
    if last is None:
        raise TimeoutError(f"Timeout på {cmd}, svar={log}")
    if last.lower().startswith("error"):
        raise RuntimeError(f"GRBL error på {cmd}, svar={log}")
    return log


def status(ser):
    ser.write(b"?")
    ser.flush()
    return ser.readline().decode(errors="ignore").strip()


def parse_state(sline):
    if not sline.startswith("<"):
        return ""
    body = sline[1:-1] if sline.endswith(">") else sline[1:]
    for sep in ("|", ","):
        idx = body.find(sep)
        if idx != -1:
            return body[:idx]
    return body


def wait_until_idle(ser, max_wait):
    start = time.time()
    while time.time() - start < max_wait:
        s = status(ser)
        st = parse_state(s)
        if st == "Idle":
            return
        time.sleep(STATUS_POLL)
    raise TimeoutError("Blev inte Idle inom tidsgränsen.")


def ensure_homed(ser):
    s0 = status(ser)
    if parse_state(s0) == "Alarm":
        print("[INFO] Alarm-läge före homing: $X.")
        send_cmd(ser, "$X", wait_ok=True)
        time.sleep(0.2)
    print("[INFO] Kör homing ($H)...")
    send_cmd(ser, "$H", wait_ok=False)

    start = time.time()
    while time.time() - start < HOME_TIMEOUT:
        s = status(ser)
        st = parse_state(s)
        if st == "Idle":
            print("[INFO] Homing klart.")
            send_cmd(ser, "G54")
            wait_until_idle(ser, 5.0)
            send_cmd(ser, "G10 L20 P1 X0 Y0 Z0")
            wait_until_idle(ser, 5.0)
            send_cmd(ser, f"G0 X0 Y0 F{FEED_RATE}")
            print("[INFO] Work coordinate system G54 set to 0,0,0 at home position")
            return
        if st == "Alarm":
            raise RuntimeError(f"Homing misslyckades: {s}")
        time.sleep(STATUS_POLL)
    raise TimeoutError("Homing tog för lång tid.")


def pen_up(ser):
    send_cmd(ser, PEN_UP_CMD)
    wait_until_idle(ser, 5.0)


def pen_down(ser):
    send_cmd(ser, PEN_DOWN_CMD)
    wait_until_idle(ser, 5.0)


def draw_grid(ser):
    print("[INFO] Starting grid drawing...")

    # Start with pen up
    pen_up(ser)

    # Draw vertical lines
    print("[INFO] Drawing vertical lines...")
    for x in range(0, GRID_SIZE + 1, GRID_SPACING):
        # Move to start of line (pen up)
        send_cmd(ser, f"G0 X{x} Y0", timeout=MOVE_TIMEOUT)
        wait_until_idle(ser, MOVE_TIMEOUT)

        # Pen down and draw line
        pen_down(ser)
        send_cmd(ser, f"G0 X{x} Y{GRID_SIZE}", timeout=MOVE_TIMEOUT)
        wait_until_idle(ser, MOVE_TIMEOUT)
        pen_up(ser)

    # Draw horizontal lines
    print("[INFO] Drawing horizontal lines...")
    for y in range(0, GRID_SIZE + 1, GRID_SPACING):
        # Move to start of line (pen up)
        send_cmd(ser, f"G0 X0 Y{y}", timeout=MOVE_TIMEOUT)
        wait_until_idle(ser, MOVE_TIMEOUT)

        # Pen down and draw line
        pen_down(ser)
        send_cmd(ser, f"G0 X{GRID_SIZE} Y{y}", timeout=MOVE_TIMEOUT)
        wait_until_idle(ser, MOVE_TIMEOUT)
        pen_up(ser)

    # Return to origin
    print("[INFO] Returning to origin...")
    send_cmd(ser, "G0 X0 Y0", timeout=MOVE_TIMEOUT)
    wait_until_idle(ser, MOVE_TIMEOUT)

    print("[INFO] Grid drawing complete!")


def main():
    ser = find_grbl_port()
    try:
        # 1) HOME and set G54 coordinate system
        ensure_homed(ser)

        # 2) Basic setup
        # send_cmd(ser, "G21")  # mm
        # wait_until_idle(ser, 5.0)
        # send_cmd(ser, "G90")  # absolute positioning
        # wait_until_idle(ser, 5.0)
        send_cmd(ser, "G17")  # XY-plane
        wait_until_idle(ser, 5.0)

        # 3) Ensure we're using G54 coordinate system
        send_cmd(ser, "G54")
        wait_until_idle(ser, 5.0)

        # 4) Optional: Move to work origin if you want offset
        if ORIGIN_X != 0 or ORIGIN_Y != 0:
            print(f"[INFO] Moving to work origin: X{ORIGIN_X} Y{ORIGIN_Y}")
            send_cmd(ser, f"G0 X{ORIGIN_X} Y{ORIGIN_Y}", timeout=MOVE_TIMEOUT)
            wait_until_idle(ser, MOVE_TIMEOUT)
            send_cmd(ser, "G55")
            wait_until_idle(ser, 5.0)
            send_cmd(ser, "G10 L20 P2 X0 Y0 Z0")
            wait_until_idle(ser, 5.0)
            print("[INFO] Work origin set in G55 coordinate system")

        # 5) Draw the grid
        draw_grid(ser)

        # Status check
        s = status(ser)
        print("[INFO] Slutstatus:", s)
        print("[KLART] Grid drawing completed successfully!")

    finally:
        ser.close()


if __name__ == "__main__":
    main()
