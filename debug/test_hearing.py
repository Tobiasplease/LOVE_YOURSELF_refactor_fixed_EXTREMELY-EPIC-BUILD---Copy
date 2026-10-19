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
    pip install faster-whisper sounddevice silero-vad

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


class SpeechGate:
    """Silero VAD as a second witness. Whisper must emit text; this one is allowed to say nothing.

    That asymmetry is the whole point. On the Sep 12 subway clips the two separated cleanly —
    real speech came in at no_speech 0.06-0.12, every ghost at 0.47-0.57, and the VAD was right
    on all of them, including the clip where whisper heard "Neska Haruna" for "Nasta Skarholmen":
    wrong words, real speech, and the VAD said so. Presence and accuracy are separate questions
    and only this one answers presence.
    """

    def __init__(self, enabled=True):
        self.enabled = enabled
        self._model = None
        self._timestamps = None
        self._failed = False
        self._lock = threading.Lock()

    def load(self):
        if self._model is not None or self._failed or not self.enabled:
            return
        try:
            from silero_vad import get_speech_timestamps, load_silero_vad

            self._model, self._timestamps = load_silero_vad(), get_speech_timestamps
            print("[hearing] silero vad ready — lines it disagrees with get flagged as ghosts")
        except Exception as e:
            self._failed = True
            print(f"[hearing] no silero vad ({e}) — falling back to no_speech_prob alone.  pip install silero-vad")

    def spans(self, samples):
        """[(start_s, end_s), ...] where it hears speech, [] for none, None if the gate is unavailable."""
        self.load()
        if self._model is None:
            return None
        try:
            import torch

            with self._lock:
                stamps = self._timestamps(torch.as_tensor(_as_float32_array(samples)), self._model, sampling_rate=SAMPLE_RATE, return_seconds=True)
            return [(round(t["start"], 1), round(t["end"], 1)) for t in stamps]
        except Exception as e:
            self._failed = True
            print(f"[hearing] vad failed, disabling it: {e}")
            return None


class SoundTagger:
    """AudioSet labels beside the words, so a train can arrive as a train and not as a sentence.

    Weakest of the three witnesses and never to be read alone: over near-silence it reported
    "Chicken, rooster 0.33" and "Writing 0.56" with the same confidence it tags a real train.
    It earns its place over actual signal. Pair every reading with the VAD verdict.
    """

    MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"

    def __init__(self, enabled=True, topk=3):
        self.enabled, self.topk = enabled, topk
        self._pipe = None
        self._failed = False
        self._lock = threading.Lock()

    def load(self):
        if self._pipe is not None or self._failed or not self.enabled:
            return
        try:
            from transformers import pipeline

            print("[hearing] loading AudioSet tagger — first run downloads weights...")
            self._pipe = pipeline("audio-classification", model=self.MODEL, device=-1)
            print("[hearing] sound tagger ready — non-speech gets named, not just transcribed")
        except Exception as e:
            self._failed = True
            print(f"[hearing] no sound tagger ({e}) — words only.  pip install transformers")

    def tags(self, samples):
        """[(label, score), ...] or None if the tagger is unavailable."""
        self.load()
        if self._pipe is None:
            return None
        try:
            with self._lock:
                out = self._pipe({"raw": _as_float32_array(samples), "sampling_rate": SAMPLE_RATE}, top_k=self.topk)
            return [(t["label"], round(float(t["score"]), 3)) for t in out]
        except Exception as e:
            self._failed = True
            print(f"[hearing] tagger failed, disabling it: {e}")
            return None


class Transcriber:
    def __init__(self, model_size="base", device="cpu", compute_type="int8", language=None, fake=False, no_speech_threshold=0.6,
                 gate=None, tagger=None, max_pending=1):
        self.model_size, self.device, self.compute_type, self.language, self.fake = model_size, device, compute_type, language, fake
        self.no_speech_threshold = no_speech_threshold
        self.gate, self.tagger, self.max_pending = gate, tagger, max_pending
        self._model = None
        self._pending = 0
        self._pending_lock = threading.Lock()
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
        """Drops what it cannot keep up with. A transcript that lands a minute late describes a room
        that has already moved on, which is worse than admitting the gap — and the queue only grows."""
        if self.fake:
            return self._transcribe(samples)
        with self._pending_lock:
            if self._pending > self.max_pending:
                return {"text": "", "avg_logprob": None, "band": "unknown", "no_speech_prob": 0.0, "hallucinated": False,
                        "vad_spans": None, "vad_witness": "skipped", "tags": None, "dropped": True,
                        "duration_s": len(samples) / SAMPLE_RATE, "elapsed_s": 0.0}
            self._pending += 1
        try:
            return self._transcribe(samples)
        finally:
            with self._pending_lock:
                self._pending -= 1

    def _transcribe(self, samples):
        """Returns {text, avg_logprob, band, no_speech_prob, hallucinated, vad_spans, vad_witness, tags, duration_s, elapsed_s}."""
        duration = len(samples) / SAMPLE_RATE
        if self.fake:
            time.sleep(0.05)
            return {"text": f"[fake] {duration:.1f}s of audio, peak {dbfs(samples):.0f} dBFS", "avg_logprob": -0.3, "band": "high",
                    "no_speech_prob": 0.0, "hallucinated": False, "vad_spans": None, "vad_witness": "fake",
                    "tags": None, "dropped": False, "duration_s": duration, "elapsed_s": 0.05}

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
        # and a far-field mic in a quiet room is the ideal condition for it. Flagged, never
        # dropped: seeing what it invents is most of what this test is for.
        by_phrase = text != "" and _looks_like_hallucination(text)
        spans = self.gate.spans(samples) if self.gate is not None else None
        if spans is None:
            hallucinated, witness = (no_speech >= self.no_speech_threshold or by_phrase), "no_speech"
        elif text and not spans:
            hallucinated, witness = True, "vad"
        else:
            hallucinated, witness = by_phrase, "vad"

        tags = self.tagger.tags(samples) if self.tagger is not None else None

        return {"text": text, "avg_logprob": avg, "band": confidence_band(avg), "no_speech_prob": no_speech,
                "hallucinated": hallucinated, "vad_spans": spans, "vad_witness": witness, "tags": tags, "dropped": False,
                "duration_s": duration, "elapsed_s": time.time() - t0}


def _as_float32_array(samples):
    try:
        import numpy as np

        return np.asarray(samples, dtype="float32")
    except ImportError:
        return samples


def report(result, source):
    spans = result.get("vad_spans")
    tags = result.get("tags")
    tagline = "      sounds: " + ", ".join(f"{lbl} {sc:.2f}" for lbl, sc in tags) if tags else ""
    if result.get("dropped"):
        print(f"  [{source}] {result['duration_s']:.1f}s — skipped, still catching up on the last one")
        return
    if not result["text"]:
        missed = "  (vad heard speech here — whisper gave nothing)" if spans else ""
        print(f"  [{source}] {result['duration_s']:.1f}s — nothing transcribed{missed}")
        if tagline:
            print(tagline)
        return
    rt = result["elapsed_s"] / result["duration_s"] if result["duration_s"] else 0
    lp = f"{result['avg_logprob']:.2f}" if result["avg_logprob"] is not None else "n/a"
    vad = "" if spans is None else (f" vad={len(spans)}sp" if spans else " vad=SILENT")
    if result.get("hallucinated") and spans == []:
        flag = "  << GHOST — vad heard no speech at all"
    elif result.get("hallucinated"):
        flag = "  << probably invented over silence"
    else:
        flag = ""
    print(f"  [{source}] {result['band'].upper():6} logprob={lp} no_speech={result.get('no_speech_prob', 0):.2f}{vad} "
          f"{result['duration_s']:.1f}s audio in {result['elapsed_s']:.1f}s (x{rt:.2f}){flag}")
    print(f"      {result['text']}")
    if tagline:
        print(tagline)


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
  .ghost .meta { color:#8a6a9a; } .ghost .text { color:#9a7aaa; text-decoration:none; font-style:italic; }
  .tags { color:#5a7a8a; font-size:12px; margin-top:4px; }
  .dropped .meta { color:#555; }
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
  const spans = res.vad_spans;
  const isGhost = spans != null && spans.length === 0 && !!res.text;
  d.className = 'line ' + (res.band || '') + (res.hallucinated ? ' invented' : '') + (isGhost ? ' ghost' : '') + (res.dropped ? ' dropped' : '');
  if (res.dropped) {
    d.innerHTML = '<div class="meta">skipped — still catching up · ' + (res.duration_s || 0).toFixed(1) + 's</div>';
    $('log').prepend(d);
    return;
  }
  const lp = res.avg_logprob == null ? 'n/a' : res.avg_logprob.toFixed(2);
  const ns = res.no_speech_prob == null ? '' : ' · no_speech ' + res.no_speech_prob.toFixed(2);
  const vad = spans == null ? '' : (spans.length ? ' · vad ' + spans.length + 'sp' : ' · vad SILENT');
  const halluc = isGhost ? ' · GHOST' : (res.hallucinated ? ' · INVENTED?' : '');
  d.innerHTML = '<div class="meta">' + (res.band || '?') + ' · logprob ' + lp + ns + vad + halluc + ' · ' +
                (res.duration_s || 0).toFixed(1) + 's in ' + (res.elapsed_s || 0).toFixed(1) + 's</div>' +
                '<div class="text"></div>';
  const t = d.querySelector('.text');
  if (res.text) t.textContent = res.text; else { t.className = 'text empty'; t.textContent = '(nothing transcribed)'; }
  if (res.tags && res.tags.length) {
    const g = document.createElement('div');
    g.className = 'tags';
    g.textContent = res.tags.map(x => x[0] + ' ' + x[1].toFixed(2)).join(' · ');
    d.appendChild(g);
  }
  $('log').prepend(d);
}

async function start() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
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

    class Server(http.server.ThreadingHTTPServer):
        daemon_threads = True

        def handle_error(self, request, client_address):
            if isinstance(sys.exc_info()[1], (ssl.SSLError, ConnectionError)):
                return
            super().handle_error(request, client_address)

    server = Server((args.host, args.port), Handler)

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
    p.add_argument("--no-vad", action="store_true", help="skip the silero second witness and judge on no_speech_prob alone")
    p.add_argument("--no-tags", action="store_true", help="skip AudioSet sound tagging (words only)")
    p.add_argument("--max-pending", type=int, default=1, help="clips allowed to wait before new ones are dropped (default: 1)")

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

    gate = SpeechGate(enabled=not args.no_vad and not args.fake)
    tagger = SoundTagger(enabled=not args.no_tags and not args.fake)
    transcriber = Transcriber(args.model, args.whisper_device, args.compute_type, args.language, args.fake, args.no_speech_threshold,
                              gate, tagger, args.max_pending)

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
