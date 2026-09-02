# Unattended runs (Sep 2, 2026)

The machine can now run without the artist present. What made that true, what
the commands are, and the one setup step (remote kill) that needs the artist's
phone once.

## Start / watch / stop

```
./start_impostor.sh              # tmux session, 3.8 stack, auto-restart loop
tmux attach -t impostor-system   # watch (detach: Ctrl+B, D)
./stop_machine.sh                # graceful stop from ANY shell — safe twice
```

`stop_machine.sh` = `touch STOP` (ends the restart loop) + `pkill -INT`
(the same clean shutdown as Ctrl+C: state saves, servos detach, pen up).
ComfyUI is auto-launched by the machine itself (utils/comfy_launcher.py)
and left running across restarts — no separate window needed.

## Why unattended is safe by construction

- **The pen cannot move without a verified blank sheet.** The VLM paper gate
  fails CLOSED at every layer (occluded view → blocked; no get-clear
  recording → blocked; unclear → blocked). Nobody there to place fresh paper
  = no drawing possible. Wants accumulate refusals instead (B3) — which is
  material, not failure.
- Server wedges → watchdog restart. Crashes → supervisor loop restart with
  full state restore (blink resume). RTC skew → clock_guard. Double launch →
  single-instance guard. Homing failures → retry with soft reset.
- The becoming systems (B4 stillness, story beat, silence, want ledger) all
  run on the machine's own clock and are MORE active in an empty room.

## Remote kill over the phone hotspot (one-time setup, artist's auth)

The studio has no wifi — the machine sees the internet only through the
phone hotspot. Tailscale traverses that NAT, so when the hotspot is up, the
phone itself can reach the machine:

```
curl -fsSL https://tailscale.com/install.sh | sh   # on the studio machine
sudo tailscale up                                   # login link, once
```

Then install the Tailscale app + any SSH client (e.g. Termius) on the phone;
the machine appears as a device. The kill from the phone:

```
ssh impostor@<tailscale-name> './LOVE_YOURSELF_refactor_fixed_EXTREMELY-EPIC-BUILD---Copy/stop_machine.sh'
```

No hotspot = no remote reach, but also: the machine is fail-safe alone (see
above), and the local stop is one command at the keyboard. For exhibitions
(fully offline, artist present daily) nothing changes.

## Checking on it after a stretch away

- `python debug/log_viewer.py` — the session story.
- `event_log/want_ledger.json` — what it wanted, what became of it.
- grep the newest `*-event-log.json` for `story_beat` / `chosen_silence` /
  `"Nothing has happened"` — the solitude record.
- `git log` on the remote is the only backup — push anything it wrote that
  matters (standing rule).
