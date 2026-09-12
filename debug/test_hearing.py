#!/usr/bin/env python3
"""Standalone hearing test — can whisper pick up a voice in this room, or from a phone?

Deliberately isolated: imports nothing from the project, touches no machine
state, runs with machine.py up or down. This answers the only question that
matters first — does it hear you at all — before any of it goes near a prompt.

    # what can I capture with?
    python debug/test_hearing.py --list-devices

    # the studio mic (the webcam capsule, probably)
    python debug/test_hearing.py --local
    python debug/test_hearing.py --local --device 4 --model base

    # the phone, over Tailscale
    sudo tailscale cert "$(tailscale status --json | python3 -c 'import json,sys;print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))')"
    python debug/test_hearing.py --phone --cert <name>.crt --key <name>.key
    # then open https://<name>.ts.net:8810 on the phone

    # no mic at all — model sanity check on a file
    python debug/test_hearing.py --wav some_speech.wav

    # transport only, no model load (segmentation + POST + page)
    python debug/test_hearing.py --phone --fake

Install (the studio box, inside .venv):
    pip install faster-whisper sounddevice

faster-whisper is CPU int8 here on purpose — the 27B is resident in
llama-server and the Aug 9 cliff says what fighting it for the GPU costs.
"""

import argparse
import json
import math
import os
import ssl
import struct
import sys
import threading
import time
import wave
from collections import deque

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000

DEFAULT_PORT = 8810
CONF_HIGH = -0.5
CONF_LOW = -1.1


# --- audio helpers -----------------------------------------------------------


def pcm16_to_floats(raw):
    if not raw:
        return []
    count = len(raw) // 2
    return [s / 32768.0 for s in struct.unpack(f"<{count}h", raw[: count * 2])]


def floats_to_pcm16(samples):
    clipped = [max(-1.0, min(1.0, s)) for s in samples]
    return struct.pack(f"<{len(clipped)}h", *[int(s * 32767) for s in clipped])


def write_wav(path, samples, rate=SAMPLE_RATE):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(floats_to_pcm16(samples))


def read_wav(path_or_bytes):
    """Returns (samples, rate). Accepts a path or raw WAV bytes; mono-mixes and returns floats."""
    import io

    src = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, (bytes, bytearray)) else path_or_bytes
    with wave.open(src, "rb") as wf:
        channels, width, rate, frames = wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes()
        raw = wf.readframes(frames)
    if width != 2:
        raise ValueError(f"expected 16-bit WAV, got {width * 8}-bit")
    samples = pcm16_to_floats(raw)
    if channels > 1:
        samples = [sum(samples[i : i + channels]) / channels for i in range(0, len(samples) - channels + 1, channels)]
    return samples, rate


def resample(samples, src_rate, dst_rate=SAMPLE_RATE):
    if src_rate == dst_rate or not samples:
        return samples
    ratio = dst_rate / src_rate
    out_len = int(len(samples) * ratio)
    out = []
    for i in range(out_len):
        pos = i / ratio
        lo = int(pos)
        hi = min(lo + 1, len(samples) - 1)
        frac = pos - lo
        out.append(samples[lo] * (1 - frac) + samples[hi] * frac)
    return out


def rms(samples):
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


def dbfs(samples):
    r = rms(samples)
    return 20 * math.log10(r) if r > 1e-9 else -99.0


# --- utterance segmentation --------------------------------------------------


class Segmenter:
    """Energy-gated utterance detector.

    Fixed chunks were the 2025 POC's mistake — 5s of wall clock cuts words in
    half and transcribes silence at full cost. This opens on speech and closes
    on a real pause, so fragments arrive whole.

    Calibrates its floor from the room's own noise, which is the difference
    between a quiet studio and a gallery with people in it.
    """

    def __init__(self, threshold_db=None, silence_hold_s=0.8, min_utterance_s=0.4, max_utterance_s=12.0, pre_roll_s=0.3):
        self.threshold_db = threshold_db
        self.silence_hold_s = silence_hold_s
        self.min_utterance_s = min_utterance_s
        self.max_utterance_s = max_utterance_s
        self.pre_roll = deque(maxlen=max(1, int(pre_roll_s * 1000 / FRAME_MS)))
        self.noise_floor_db = None
        self._noise_frames = []
        self.active = False
        self.buffer = []
        self.silence_run = 0.0

    def _effective_threshold(self):
        if self.threshold_db is not None:
            return self.threshold_db
        if self.noise_floor_db is None:
            return -45.0
        return min(-25.0, self.noise_floor_db + 10.0)

    def push(self, frame):
        """Feed one FRAME_MS frame. Returns a completed utterance (list of samples) or None."""
        level = dbfs(frame)

        if self.noise_floor_db is None:
            self._noise_frames.append(level)
            if len(self._noise_frames) >= int(1000 / FRAME_MS):
                quiet = sorted(self._noise_frames)[: max(1, len(self._noise_frames) // 2)]
                self.noise_floor_db = sum(quiet) / len(quiet)

        speaking = level > self._effective_threshold()

        if not self.active:
            self.pre_roll.append(frame)
            if speaking:
                self.active = True
                self.buffer = [s for f in self.pre_roll for s in f]
                self.pre_roll.clear()
                self.silence_run = 0.0
            return None

        self.buffer.extend(frame)
        self.silence_run = 0.0 if speaking else self.silence_run + FRAME_MS / 1000.0

        too_long = len(self.buffer) / SAMPLE_RATE >= self.max_utterance_s
        if self.silence_run >= self.silence_hold_s or too_long:
            return self.flush()
        return None

    def flush(self):
        utterance, self.buffer, self.active, self.silence_run = self.buffer, [], False, 0.0
        if len(utterance) / SAMPLE_RATE < self.min_utterance_s:
            return None
        return utterance


# --- transcription -----------------------------------------------------------


def confidence_band(avg_logprob):
    if avg_logprob is None:
        return "unknown"
    if avg_logprob >= CONF_HIGH:
        return "high"
    if avg_logprob >= CONF_LOW:
        return "middle"
    return "low"


_HALLUCINATION_PHRASES = (
    "thank you for watching",
    "thanks for watching",
    "subtitles by",
    "subscribe to",
    "amara.org",
    "please subscribe",
    "see you next time",
)


def _looks_like_hallucination(text):
    stripped = text.lower().strip(" .!?,-—")
    return any(p in stripped for p in _HALLUCINATION_PHRASES) or stripped in ("thank you", "you", "bye", "thanks")


class Transcriber:
    def __init__(self, model_size="base", device="cpu", compute_type="int8", language=None, fake=False, no_speech_threshold=0.6):
        self.model_size, self.device, self.compute_type, self.language, self.fake = model_size, device, compute_type, language, fake
        self.no_speech_threshold = no_speech_threshold
        self._model = None
        self._lock = threading.Lock()

    def load(self):
        if self.fake or self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            sys.exit("faster-whisper is not installed.\n    pip install faster-whisper")
        print(f"[hearing] loading {self.model_size} on {self.device} ({self.compute_type}) — first run downloads weights...")
        t0 = time.time()
        self._model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
        print(f"[hearing] model ready in {time.time() - t0:.1f}s")

    def transcribe(self, samples):
        """Returns {text, avg_logprob, band, no_speech_prob, hallucinated, duration_s, elapsed_s}."""
        duration = len(samples) / SAMPLE_RATE
        if self.fake:
            time.sleep(0.05)
            return {"text": f"[fake] {duration:.1f}s of audio, peak {dbfs(samples):.0f} dBFS", "avg_logprob": -0.3, "band": "high",
                    "no_speech_prob": 0.0, "hallucinated": False, "duration_s": duration, "elapsed_s": 0.05}

        self.load()
        t0 = time.time()
        with self._lock:
            segments, _ = self._model.transcribe(
                _as_float32_array(samples), language=self.language, beam_size=1, vad_filter=False, condition_on_previous_text=False
            )
            segments = list(segments)

        text = " ".join(s.text.strip() for s in segments).strip()
        logprobs = [s.avg_logprob for s in segments if getattr(s, "avg_logprob", None) is not None]
        avg = sum(logprobs) / len(logprobs) if logprobs else None
        no_speech = max((getattr(s, "no_speech_prob", 0.0) or 0.0 for s in segments), default=0.0)

        # Whisper invents text over silence and room tone — "Thank you.", "Subtitles by..." —
        # and a far-field mic in a quiet room is the ideal condition for it. Flagged, not
        # dropped: seeing what it invents is part of what this test is for.
        hallucinated = no_speech >= self.no_speech_threshold or (text != "" and _looks_like_hallucination(text))

        return {"text": text, "avg_logprob": avg, "band": confidence_band(avg), "no_speech_prob": no_speech,
                "hallucinated": hallucinated, "duration_s": duration, "elapsed_s": time.time() - t0}


def _as_float32_array(samples):
    try:
        import numpy as np

        return np.asarray(samples, dtype="float32")
    except ImportError:
        return samples


def report(result, source):
    if not result["text"]:
        print(f"  [{source}] {result['duration_s']:.1f}s — nothing transcribed")
        return
    rt = result["elapsed_s"] / result["duration_s"] if result["duration_s"] else 0
    lp = f"{result['avg_logprob']:.2f}" if result["avg_logprob"] is not None else "n/a"
    flag = "  << probably invented over silence" if result.get("hallucinated") else ""
    print(f"  [{source}] {result['band'].upper():6} logprob={lp} no_speech={result.get('no_speech_prob', 0):.2f} "
          f"{result['duration_s']:.1f}s audio in {result['elapsed_s']:.1f}s (x{rt:.2f}){flag}")
    print(f"      {result['text']}")


# --- local microphone --------------------------------------------------------


def list_devices():
    try:
        import sounddevice as sd
    except ImportError:
        print("sounddevice not installed (pip install sounddevice). Falling back to `arecord -l`:\n")
        os.system("arecord -l")
        return
    print("Input devices:\n")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            default = " (default)" if i == sd.default.device[0] else ""
            print(f"  {i:3}  {d['name']}{default}")
            print(f"       {d['max_input_channels']}ch, {d['default_samplerate']:.0f} Hz")
    print("\nThe room mic is most likely the webcam. Pass its index with --device.")


def run_local(args, transcriber):
    try:
        import sounddevice as sd
    except ImportError:
        sys.exit("sounddevice is not installed.\n    pip install sounddevice\n(or use --phone, which needs no local mic)")

    segmenter = Segmenter(threshold_db=args.threshold, silence_hold_s=args.silence_hold, max_utterance_s=args.max_utterance)
    transcriber.load()

    pending = deque()
    stop = threading.Event()

    def on_audio(indata, _frames, _time_info, status):
        if status:
            print(f"[hearing] stream status: {status}", file=sys.stderr)
        mono = [float(f[0]) for f in indata]
        for i in range(0, len(mono) - FRAME_SAMPLES + 1, FRAME_SAMPLES):
            utterance = segmenter.push(mono[i : i + FRAME_SAMPLES])
            if utterance:
                pending.append(utterance)

    def meter():
        while not stop.wait(2.0):
            if segmenter.noise_floor_db is not None and not segmenter.active:
                print(f"[hearing] listening — noise floor {segmenter.noise_floor_db:.0f} dBFS, gate {segmenter._effective_threshold():.0f} dBFS")

    print(f"[hearing] opening device {args.device if args.device is not None else 'default'} at {SAMPLE_RATE} Hz")
    print("[hearing] speak — utterances transcribe on each pause. Ctrl-C to stop.\n")

    threading.Thread(target=meter, daemon=True).start()
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=FRAME_SAMPLES * 4,
                            device=args.device, callback=on_audio):
            while True:
                if pending:
                    utterance = pending.popleft()
                    if args.save_dir:
                        _save_utterance(args.save_dir, utterance)
                    report(transcriber.transcribe(utterance), "room")
                else:
                    time.sleep(0.05)
    except KeyboardInterrupt:
        stop.set()
        print("\n[hearing] stopped")


def _save_utterance(save_dir, samples):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"utterance_{time.strftime('%H%M%S')}_{int(time.time() * 1000) % 1000:03d}.wav")
    write_wav(path, samples)
    print(f"      saved {path}")


# --- phone over the network --------------------------------------------------

PHONE_PAGE = r"""<!doctype html>
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1">
<title>hearing test</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; padding:20px; background:#111; color:#d8d8d8; font:15px/1.5 ui-monospace,Menlo,Consolas,monospace; }
  h1 { font-size:14px; font-weight:400; color:#888; text-transform:uppercase; letter-spacing:.1em; margin:0 0 16px; }
  button { width:100%; padding:22px; font:inherit; font-size:17px; border:1px solid #444; border-radius:6px;
           background:#1b1b1b; color:#d8d8d8; cursor:pointer; -webkit-tap-highlight-color:transparent; }
  button.on { background:#2d4a2d; border-color:#4a7a4a; color:#cfe8cf; }
  #meter { height:5px; background:#222; border-radius:3px; margin:14px 0; overflow:hidden; }
  #bar { height:100%; width:0; background:#4a7a4a; transition:width .06s linear; }
  #status { color:#777; font-size:13px; min-height:1.4em; }
  .line { border-top:1px solid #262626; padding:11px 0; }
  .meta { color:#666; font-size:11px; text-transform:uppercase; letter-spacing:.06em; }
  .high .meta { color:#6a9a6a; } .middle .meta { color:#9a8a4a; } .low .meta { color:#8a5a5a; }
  .text { margin-top:3px; word-wrap:break-word; }
  .empty { color:#555; font-style:italic; }
  .invented .text { color:#7a6a5a; text-decoration:line-through; }
</style>
<h1>hearing test</h1>
<button id="go">start listening</button>
<div id="meter"><div id="bar"></div></div>
<div id="status">idle</div>
<div id="log"></div>
<script>
const RATE = 16000, FRAME = 480;
const SILENCE_HOLD = 0.8, MIN_UTT = 0.4, MAX_UTT = 12.0, PRE_ROLL = 0.3;
let ctx, stream, node, src, on = false;
let floor = null, floorFrames = [], active = false, buf = [], pre = [], silence = 0;

const $ = id => document.getElementById(id);
const setStatus = t => $('status').textContent = t;

function db(frame) {
  let s = 0; for (let i = 0; i < frame.length; i++) s += frame[i] * frame[i];
  const r = Math.sqrt(s / frame.length);
  return r > 1e-9 ? 20 * Math.log10(r) : -99;
}
function gate() { return floor === null ? -45 : Math.min(-25, floor + 10); }

function pushFrame(frame) {
  const level = db(frame);
  $('bar').style.width = Math.max(0, Math.min(100, (level + 60) / 60 * 100)) + '%';

  if (floor === null) {
    floorFrames.push(level);
    if (floorFrames.length >= 33) {
      const q = floorFrames.slice().sort((a, b) => a - b).slice(0, 16);
      floor = q.reduce((a, b) => a + b, 0) / q.length;
      setStatus('listening — floor ' + floor.toFixed(0) + ' dBFS, gate ' + gate().toFixed(0));
    }
  }
  const speaking = level > gate();

  if (!active) {
    pre.push(frame);
    if (pre.length > Math.ceil(PRE_ROLL * RATE / FRAME)) pre.shift();
    if (speaking) { active = true; buf = [].concat(...pre); pre = []; silence = 0; setStatus('hearing something...'); }
    return;
  }
  buf = buf.concat(Array.from(frame));
  silence = speaking ? 0 : silence + FRAME / RATE;
  if (silence >= SILENCE_HOLD || buf.length / RATE >= MAX_UTT) {
    const utt = buf; buf = []; active = false; silence = 0;
    if (utt.length / RATE >= MIN_UTT) send(utt); else setStatus('too short, ignored');
  }
}

function wav(samples) {
  const b = new ArrayBuffer(44 + samples.length * 2), v = new DataView(b);
  const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  str(0, 'RIFF'); v.setUint32(4, 36 + samples.length * 2, true); str(8, 'WAVEfmt ');
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, RATE, true); v.setUint32(28, RATE * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  str(36, 'data'); v.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(44 + i * 2, s * 32767, true);
  }
  return new Blob([b], { type: 'audio/wav' });
}

async function send(samples) {
  setStatus('transcribing ' + (samples.length / RATE).toFixed(1) + 's...');
  try {
    const r = await fetch('/utterance', { method: 'POST', headers: { 'Content-Type': 'audio/wav' }, body: wav(samples) });
    render(await r.json());
    setStatus(on ? 'listening' : 'idle');
  } catch (e) { setStatus('send failed: ' + e.message); }
}

function render(res) {
  const d = document.createElement('div');
  d.className = 'line ' + (res.band || '') + (res.hallucinated ? ' invented' : '');
  const lp = res.avg_logprob == null ? 'n/a' : res.avg_logprob.toFixed(2);
  const ns = res.no_speech_prob == null ? '' : ' · no_speech ' + res.no_speech_prob.toFixed(2);
  const halluc = res.hallucinated ? ' · INVENTED?' : '';
  d.innerHTML = '<div class="meta">' + (res.band || '?') + ' · logprob ' + lp + ns + halluc + ' · ' +
                (res.duration_s || 0).toFixed(1) + 's in ' + (res.elapsed_s || 0).toFixed(1) + 's</div>' +
                '<div class="text"></div>';
  const t = d.querySelector('.text');
  if (res.text) t.textContent = res.text; else { t.className = 'text empty'; t.textContent = '(nothing transcribed)'; }
  $('log').prepend(d);
}

async function start() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
  } catch (e) {
    setStatus('mic refused: ' + e.name + (location.protocol !== 'https:' ? ' — page is not HTTPS, iOS will always refuse' : ''));
    return;
  }
  ctx = new (window.AudioContext || window.webkitAudioContext)();
  await ctx.resume();
  src = ctx.createMediaStreamSource(stream);
  node = ctx.createScriptProcessor(4096, 1, 1);
  let carry = [];
  node.onaudioprocess = e => {
    if (!on) return;
    const input = e.inputBuffer.getChannelData(0);
    const ratio = RATE / ctx.sampleRate;
    const out = new Float32Array(Math.floor(input.length * ratio));
    for (let i = 0; i < out.length; i++) {
      const pos = i / ratio, lo = Math.floor(pos), hi = Math.min(lo + 1, input.length - 1);
      out[i] = input[lo] * (1 - (pos - lo)) + input[hi] * (pos - lo);
    }
    carry = carry.concat(Array.from(out));
    while (carry.length >= FRAME) pushFrame(carry.splice(0, FRAME));
  };
  src.connect(node); node.connect(ctx.destination);
  on = true;
  $('go').textContent = 'stop'; $('go').className = 'on';
  setStatus('calibrating room noise — stay quiet a moment');
}

function stop() {
  on = false; active = false; buf = []; pre = []; floor = null; floorFrames = [];
  if (node) node.disconnect();
  if (src) src.disconnect();
  if (stream) stream.getTracks().forEach(t => t.stop());
  if (ctx) ctx.close();
  $('go').textContent = 'start listening'; $('go').className = '';
  setStatus('idle'); $('bar').style.width = '0';
}

$('go').onclick = () => on ? stop() : start();
if (location.protocol !== 'https:' && location.hostname !== 'localhost') {
  setStatus('WARNING: not HTTPS — the browser will refuse the mic. See --cert/--key.');
}
</script>
"""


def run_phone(args, transcriber):
    import http.server

    if not args.fake:
        transcriber.load()

    heard = []

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_a):
            pass

        def _send(self, body, content_type, status=200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.split("?")[0] in ("/", "/index.html"):
                return self._send(PHONE_PAGE.encode(), "text/html; charset=utf-8")
            if self.path.split("?")[0] == "/heard":
                return self._send(json.dumps(heard[-50:]).encode(), "application/json")
            return self._send(b"not found", "text/plain", 404)

        def do_POST(self):
            if self.path.split("?")[0] != "/utterance":
                return self._send(b'{"error":"unknown endpoint"}', "application/json", 404)
            try:
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length)
                samples, rate = read_wav(raw)
                samples = resample(samples, rate)
                if args.save_dir:
                    _save_utterance(args.save_dir, samples)
                result = transcriber.transcribe(samples)
            except Exception as e:
                print(f"[hearing] phone utterance failed: {e}", file=sys.stderr)
                return self._send(json.dumps({"error": str(e)}).encode(), "application/json", 500)

            heard.append({"at": time.strftime("%H:%M:%S"), **result})
            report(result, "phone")
            return self._send(json.dumps(result).encode(), "application/json")

    server = http.server.ThreadingHTTPServer((args.host, args.port), Handler)

    scheme = "http"
    if args.cert:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(args.cert, args.key or args.cert)
        server.socket = ctx.wrap_socket(server.socket, server_side=True)
        scheme = "https"
    else:
        print("\n  !! No --cert given, serving plain HTTP.")
        print("  !! Browsers refuse getUserMedia outside a secure context, so the phone")
        print("  !! will NOT give up its mic. Fine for localhost; useless over Tailscale.")
        print("  !! Get a real cert:  sudo tailscale cert <name>.<tailnet>.ts.net\n")

    print(f"[hearing] serving on {scheme}://{args.host}:{args.port}")
    print("[hearing] open that on the phone (use the MagicDNS name, not the IP, or the cert won't match)")
    print("[hearing] Ctrl-C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[hearing] stopped")
        server.shutdown()


# --- entry -------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description="Standalone whisper hearing test — room mic or phone.")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--local", action="store_true", help="capture from a local microphone")
    mode.add_argument("--phone", action="store_true", help="serve a mic page for a phone on the tailnet")
    mode.add_argument("--wav", metavar="FILE", help="transcribe a WAV file and exit (no mic needed)")
    mode.add_argument("--list-devices", action="store_true", help="list input devices and exit")

    p.add_argument("--model", default="base", help="faster-whisper size: tiny|base|small|medium (default: base)")
    p.add_argument("--compute-type", default="int8", help="int8|int8_float16|float16|float32 (default: int8)")
    p.add_argument("--whisper-device", default="cpu", help="cpu|cuda — keep cpu while the VLM is resident (default: cpu)")
    p.add_argument("--language", default=None, help="force a language code, e.g. en or sv (default: autodetect)")
    p.add_argument("--fake", action="store_true", help="skip the model entirely — tests capture, segmentation and transport")
    p.add_argument("--no-speech-threshold", type=float, default=0.6, help="flag output above this no_speech_prob as invented (default: 0.6)")

    p.add_argument("--device", type=int, default=None, help="input device index for --local (see --list-devices)")
    p.add_argument("--threshold", type=float, default=None, help="gate in dBFS; default calibrates from room noise")
    p.add_argument("--silence-hold", type=float, default=0.8, help="seconds of quiet that end an utterance (default: 0.8)")
    p.add_argument("--max-utterance", type=float, default=12.0, help="hard cap per utterance in seconds (default: 12)")

    p.add_argument("--host", default="0.0.0.0", help="bind address for --phone (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port for --phone (default: {DEFAULT_PORT})")
    p.add_argument("--cert", default=None, help="TLS cert (from `tailscale cert`) — required for a real phone mic")
    p.add_argument("--key", default=None, help="TLS key; defaults to --cert if the file holds both")
    p.add_argument("--save-dir", default=None, help="also write each utterance to this directory as a WAV")

    args = p.parse_args()

    if args.list_devices:
        return list_devices()

    transcriber = Transcriber(args.model, args.whisper_device, args.compute_type, args.language, args.fake, args.no_speech_threshold)

    if args.wav:
        samples, rate = read_wav(args.wav)
        print(f"[hearing] {args.wav}: {len(samples) / rate:.1f}s at {rate} Hz, peak {dbfs(samples):.0f} dBFS")
        report(transcriber.transcribe(resample(samples, rate)), "file")
        return

    if args.local:
        return run_local(args, transcriber)
    if args.phone:
        return run_phone(args, transcriber)

    p.print_help()
    print("\nPick a mode: --local, --phone, --wav FILE, or --list-devices")


if __name__ == "__main__":
    main()
