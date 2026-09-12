#!/usr/bin/env python3
"""Sep 12 — the model reload after a drawing waits for the card (utils/llama_server)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.llama_server import _parse_free_mib  # noqa: E402
fails = 0
def check(name, ok, got=""):
    global fails
    print(("PASS  " if ok else "FAIL  ") + name + (f"   → {got!r}" if not ok else "")); fails += 0 if ok else 1
check("parses nvidia-smi output", _parse_free_mib("4148\n") == 4148)
check("parses with unit", _parse_free_mib(" 20428 MiB") == 20428)
check("None on garbage", _parse_free_mib("") is None and _parse_free_mib("N/A") is None)
print("\n" + ("ALL PASS" if not fails else f"{fails} FAILED")); sys.exit(1 if fails else 0)
