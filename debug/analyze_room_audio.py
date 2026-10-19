#!/usr/bin/env python3
"""Three witnesses over the same clip: level, VAD, AudioSet tags, transcript.

The point is disagreement. Whisper must emit text; Silero and the tagger may
say nothing is there. When whisper speaks over a clip both others call empty,
that line is invented — which is the only reliable way to tell a ghost from a
badly-heard announcement (see "Neska Haruna" = "Nästa Skärholmen", Sep 12).

    python debug/analyze_room_audio.py <dir-or-wav> [--language sv] [--model small]
"""

import argparse
import glob
import math
import os
import struct
import wave


def level_dbfs(path):
    with wave.open(path) as w:
        raw = w.readframes(w.getnframes())
        rate, dur = w.getframerate(), w.getnframes() / w.getframerate()
    s = struct.unpack("<%dh" % (len(raw) // 2), raw)
    rms = math.sqrt(sum(x * x for x in s) / len(s)) / 32768
    peak = max(abs(x) for x in s) / 32768
    return dur, rate, (20 * math.log10(rms) if rms else -99.0), (20 * math.log10(peak) if peak else -99.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("target", help="a WAV file or a directory of them")
    p.add_argument("--language", default=None, help="force a whisper language, e.g. sv")
    p.add_argument("--model", default="small", help="whisper size (default: small)")
    p.add_argument("--topk", type=int, default=4, help="AudioSet labels to show (default: 4)")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.target, "*.wav"))) if os.path.isdir(args.target) else [args.target]
    if not files:
        raise SystemExit(f"no WAVs under {args.target}")

    from faster_whisper import WhisperModel
    from silero_vad import get_speech_timestamps, load_silero_vad, read_audio
    from transformers import pipeline

    print(f"[analyze] loading whisper {args.model}, silero vad, AudioSet tagger...")
    whisper = WhisperModel(args.model, device="cpu", compute_type="int8")
    vad = load_silero_vad()
    tagger = pipeline("audio-classification", model="MIT/ast-finetuned-audioset-10-10-0.4593", device=-1)
    print(f"[analyze] {len(files)} clip(s)\n")

    for f in files:
        dur, rate, rms, peak = level_dbfs(f)
        print(f"=== {os.path.basename(f)}  {dur:.1f}s @ {rate} Hz   RMS {rms:.1f} dBFS   peak {peak:.1f} dBFS")

        stamps = get_speech_timestamps(read_audio(f, sampling_rate=16000), vad, sampling_rate=16000, return_seconds=True)
        if stamps:
            spans = ", ".join(f"{t['start']:.1f}-{t['end']:.1f}s" for t in stamps[:6])
            print(f"    VAD      speech in {len(stamps)} span(s): {spans}")
        else:
            print("    VAD      no speech")

        tags = tagger(f, top_k=args.topk)
        print("    SOUND    " + ", ".join(f"{t['label']} {t['score']:.2f}" for t in tags))

        segs, info = whisper.transcribe(f, language=args.language, beam_size=5)
        segs = list(segs)
        lang = args.language or f"auto->{info.language} p={info.language_probability:.2f}"
        if not segs:
            print(f"    WHISPER  ({lang}) nothing\n")
            continue
        for s in segs:
            print(f"    WHISPER  ({lang}) no_speech={s.no_speech_prob:.2f} logprob={s.avg_logprob:.2f}  {s.text.strip()}")
        if not stamps:
            print("    >>> INVENTED — whisper produced text where VAD heard no speech")
        print()


if __name__ == "__main__":
    main()
