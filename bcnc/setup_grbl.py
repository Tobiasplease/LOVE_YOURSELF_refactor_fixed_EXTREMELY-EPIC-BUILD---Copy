# setup_grbl_move_to_origin.py
# Nu börjar jag få till det med home, nollpunkt och origin.
# Man sätter nollpunkt och så i arduinon (GRBL heter programvaran) därför blir det nog svårt att testa utan roboten.

# det verkar som att det funkar och ligger kvar när man startar bCNC, dock är jag inte helt säker på att det kommer att funka med origin offset,
# så eventuellt får vi sätta det i g-koden så det finns med när vi laddar in filen. att HOME-a måste man nog göra som ett seriellt kommando medans n
# ollpunkt och origin osv är g-kod som då borde gå att baka in t ex alla våra filer.

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
            return
        if st == "Alarm":
            raise RuntimeError(f"Homing misslyckades: {s}")
        time.sleep(STATUS_POLL)
    raise TimeoutError("Homing tog för lång tid.")


def main():
    ser = find_grbl_port()
    try:
        # 1) HOME
        ensure_homed(ser)

        # 2) Grundläge
        send_cmd(ser, "G21")  # mm
        send_cmd(ser, "G90")  # absolut
        send_cmd(ser, "G17")  # XY-plan

        # 3) G54 = nolla vid HOME
        send_cmd(ser, "G54")
        send_cmd(ser, "G10 L20 P1 X0 Y0 Z0")  # sätt nuvarande pos till (0,0,0) i G54

        # 4) G55 = nolla vid HOME + (66,-2,0) — utan att röra maskinen
        send_cmd(ser, "G55")
        send_cmd(ser, f"G10 L20 P2 X{-ORIGIN_X} Y{-ORIGIN_Y} Z{-ORIGIN_Z}")

        # 5) Växla till G55 och kör till G55: X0 Y0 (dvs. fysiskt HOME + (66,-2))
        send_cmd(ser, "G55")  # säkerställ aktivt
        send_cmd(ser, "G90")  # absolut (för säkerhets skull)
        send_cmd(ser, "G0 X0 Y0", timeout=MOVE_TIMEOUT)  # snabbflytt till origin
        wait_until_idle(ser, MOVE_TIMEOUT)

        # Klar
        s = status(ser)
        print("[INFO] Slutstatus:", s)
        print("[KLART] Maskinen står nu vid din origin (G55: X0 Y0).")
        print("[INFO] I bCNC: håll G55 aktiv när du kör dina jobb.")

    finally:
        ser.close()


if __name__ == "__main__":
    main()
