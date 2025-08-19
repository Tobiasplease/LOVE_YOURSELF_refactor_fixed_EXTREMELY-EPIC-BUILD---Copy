# setup_grbl_move_to_origin.py


import time
import serial
from serial.tools import list_ports

# ======= Konfiguration =======
BAUD = 115200
STATUS_POLL = 0.2
HOME_TIMEOUT = 300
MOVE_TIMEOUT = 120

# Din önskade arbets-nolla relativt HOME:
ORIGIN_X = 66.0  # +66 mm från HOME i X
ORIGIN_Y = -2.0  # -2 mm från HOME i Y
ORIGIN_Z = 0.0
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
            # Set work coordinate system G54 to current position (home) = 0,0,0
            send_cmd(ser, "G54")  # Select coordinate system G54
            send_cmd(ser, "G10 L20 P1 X0 Y0 Z0")  # Set current position as 0,0,0 in G54
            print("[INFO] Work coordinate system G54 set to 0,0,0 at home position")
            return
        if st == "Alarm":
            raise RuntimeError(f"Homing misslyckades: {s}")
        time.sleep(STATUS_POLL)
    raise TimeoutError("Homing tog för lång tid.")


def main():
    ser = find_grbl_port()
    try:
        # 1) HOME and set G54 coordinate system
        ensure_homed(ser)

        # 2) Basic setup
        send_cmd(ser, "G21")  # mm
        send_cmd(ser, "G90")  # absolute positioning
        send_cmd(ser, "G17")  # XY-plane

        # 3) Ensure we're using G54 coordinate system
        send_cmd(ser, "G54")

        # 4) Optional: Move to work origin if you want offset
        if ORIGIN_X != 0 or ORIGIN_Y != 0:
            print(f"[INFO] Moving to work origin: X{ORIGIN_X} Y{ORIGIN_Y}")
            send_cmd(ser, f"G0 X{ORIGIN_X} Y{ORIGIN_Y}", timeout=MOVE_TIMEOUT)
            wait_until_idle(ser, MOVE_TIMEOUT)
            # Set this position as new 0,0,0 in G55
            send_cmd(ser, "G55")
            send_cmd(ser, "G10 L20 P2 X0 Y0 Z0")
            print("[INFO] Work origin set in G55 coordinate system")

        # Status check
        s = status(ser)
        print("[INFO] Slutstatus:", s)
        print("[KLART] Machine ready. bCNC will show correct WPos when opened.")
        if ORIGIN_X != 0 or ORIGIN_Y != 0:
            print("[INFO] Use G55 coordinate system in bCNC for offset work origin.")
        else:
            print("[INFO] Use G54 coordinate system in bCNC (home = 0,0,0).")

    finally:
        ser.close()


if __name__ == "__main__":
    main()
