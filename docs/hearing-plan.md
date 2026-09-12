# Hearing — plan (Sep 12 2026)

A second sense: overheard room speech, transcribed locally, reaching the
machine as fact. Nothing here is built yet. This document is the design and
the open rulings, written against `rebuild/pre-mind` @ 6141460.

## What exists today: nothing

Worth stating plainly, because until Sep 12 three things claimed otherwise:

- `requirements.txt` — `openai-whisper` and `pyaudio`, both commented out since
  2025 and never uncommented on any branch. NOW corrected to name the packages
  this actually uses (`faster-whisper`, `sounddevice`), still commented.
- `README.md` — advertised "Speech Recognition: Whisper-based local speech
  processing (optional)". NOW removed; it described nothing.
- `CLAUDE.md` (old copy on `main`) — listed Whisper under optional models;
  already clean on `rebuild/pre-mind`.

The only audio code this project has ever had was `speech_logger.py`, added
2025-07-26 (`ae99bfe`) and deleted 2025-08-15 (`91820b2`). It was a side-car
script: PyAudio → 5s fixed chunks → temp WAV → `whisper.transcribe()` → append
to a flat `speech_log.json`. It was never imported by `machine.py` and never
fed a prompt. Recoverable with `git show 42de5b4:speech_logger.py`; useful as
a reminder of what was tried, not as a starting point. (It also loaded `small`
under a comment claiming `tiny`, and cut words in half on a wall clock.)

So: the docs promise a sense the machine does not have, and the sense has
never once reached the voice.

## Constraints that shape this

1. **Offline, always.** Inference is local and stays local. Whisper satisfies
   this — but the weights download once, on first load, from a CDN. Either
   pre-seed `models/` from a machine that has a route, or accept one fetch at
   install time. After that it never touches the network. No exhibition run
   ever needs a route.
2. **The GPU is full.** A 27B VLM is resident in llama-server and ComfyUI
   auto-launches beside it. Whisper must not compete for VRAM — the Aug 9
   CPU-fallback cliff is the scar that says what GPU contention costs here.
   `faster-whisper` (CTranslate2) on **CPU**, `tiny` or `base`, int8. A 5s
   chunk is well under a second on a modern CPU; the box has cores to spare
   while the GPU does the thinking.
3. **The artist is remote for a month.** The real mic is in the studio; the
   only mic to hand is a phone. This forces a good design decision early —
   audio has to arrive from an abstract source, not from `pyaudio` directly.
4. **Failure must be silent and total.** Per the runtime map's own warning,
   features fail silently here. Hearing must degrade to "no hearing" without
   touching the caption loop, and must be provable from the event log.

## Architecture

The sidecar owns hearing. This follows the room cam precedent exactly
(`CAMERA_2_DEVICE` is sidecar-owned, `machine.py` never touches it) and buys
three things: whisper never runs in the caption loop's process, hearing works
while the machine is down, and a crash in transcription cannot take the
machine with it.

```
phone mic ─┐
           ├─► :8800 dashboard/server.py  (SIDECAR)
studio mic ┘        │  VAD gate → faster-whisper (CPU) → fragment
                    │
                    └─► POST /machine/hearing ──► :8801 machine_api.py
                                                       │
                                                  utils/hearing.py
                                                  (rolling deque, in RAM)
                                                       │
                                                  prompts.get_heard_line()
                                                       │
                                                  USER prompt, dosed
```

Both mics land on the same endpoint. The phone is one audio source among two,
not a special mode — which means the remote-testing path can be deleted in a
month without touching anything that matters.

### Why VAD, not fixed chunks

The old POC recorded 5s of wall clock regardless of whether anyone spoke. In a
gallery that is mostly silence, transcribed at full cost, and every fragment
is sliced at an arbitrary boundary. `webrtcvad` (or a plain RMS gate, which may
be enough) → transcribe only speech, bounded by real pauses. Cheaper and the
fragments are whole.

### Storage: none, by default

A rolling in-memory deque of the last N fragments, fed to the prompt and
dropped. No `speech_log.json` accumulating everything said in the room. An
optional debug tape behind a flag for tuning, off in exhibition. This is a
gallery with members of the public in it; the machine should hear and forget,
the way it already sees and forgets.

## The phone as a remote mic

The dashboard is already phone-friendly, already on Tailscale, already has no
auth because "Tailscale is the access control". Two things stand in the way:

**1. Secure context.** `getUserMedia` is refused outside a secure context, and
the dashboard is served as `http://<tailscale-ip>:8800`. An IP over plain HTTP
gives no mic on iOS Safari, no prompt, just a rejected promise. The fix is
small and you have the pieces: `tailscale cert <host>.<tailnet>.ts.net` issues
a real Let's Encrypt cert, wrap the listener in `ssl.SSLContext`, reach it by
MagicDNS name instead of IP. No warnings, no profile install. The sidecar stays
plain HTTP on LAN if that's simpler — HTTPS only needs to be true for the
mic page.

**2. `do_POST` parses every body as JSON** (`dashboard/server.py`, the
`json.loads(self.rfile.read(length))` before the route switch). An audio upload
is binary and must branch before that, or arrive base64-wrapped in JSON. The
former is cleaner; it's a three-line reorder.

Then: a `GET /mic` page — one button, `MediaRecorder`, POST each utterance —
and the phone is a microphone in the room you're not in.

**What this does and does not test.** It exercises capture, VAD, chunking,
transcription, transport, the fact line, and how the voice responds. It tells
you nothing about whether the real thing will work, because a phone held to
your face is close-talk audio at high SNR and the studio mic is a far-field
UVC capsule in a reverberant room several metres away. Whisper's accuracy gap
between those two conditions is very large. Expect the studio mic to be worse
than the phone by a margin that may decide the whole feature.

Which mic the studio machine actually has is worth checking before any of this:
`CAMERA_2_DEVICE` is a XIFT USB webcam, and `arecord -l` on the box will say
whether it exposes a capture device and what it's called.

## The part that needs your ruling

The plumbing above is uncontroversial. Where the fragments go is not, and it
runs straight into doctrine this system already has scars from.

**1. The interlocutor problem.** The July 28 rewrite of `situation.reflexive`
exists because the frame didn't say what the per-cycle user turns were, the
model inferred a speaker, and it "bred 'What do you think?' into full assistant
mode". The current frame fixes this by naming the channel honestly: *the
fragments that arrive between thoughts are your own senses reporting*.

Overheard speech is the first fragment that is **not** the machine's own sense
reporting — it is another person's voice, in words, arriving in the same
channel as its inner monologue. This is the single biggest risk in the feature.
Done carelessly it re-opens the exact door July 28 closed, and it will not
present as a bug; it will present as the voice going subtly conversational
again over a week.

My instinct is that heard speech must be framed as **sound the machine
perceived**, never as address, and never as turn-taking: *"Someone in the room
said something that sounded like '…'"* rather than anything that reads as
a line spoken to it. But this is a voice decision and it's yours.

**2. The conflation law.** The lore ledger has a firewall — invention can never
become a familiar concept, a compressed fact, or reflection material. Heard
speech needs the same firewall and for the same reason: someone saying *"the
lamp is broken"* is not attestation that the lamp is broken. Speech is
attested as **having been said**, never as being true. So: heard fragments must
not reach `observe()`, `add_caption`, concept extraction, or the events ledger
as world facts. Whether they may form their own ledger — things the room has
said, as its own memory class alongside lore — is a real question and probably
a second round, not this one.

**3. One channel per fact, and dosing.** Every fact line in this system is
dosed (identity every 6th, familiarity every ~3rd, unchanged after 1200s).
A heard fragment on every cycle would be the loudest channel in the prompt and
would flatten everything else. It should behave like salience: hot on arrival,
then gone. A fragment older than a caption cycle or two is probably not worth
saying at all.

**4. Mishearing is the interesting case.** Whisper's `avg_logprob` gives a
confidence band per segment. The obvious design throws away low-confidence
output. But a machine that half-hears a room and is sometimes wrong about what
it heard is more in character than one with a clean transcript — and the
system already has a vocabulary for holding uncertain things honestly (the
presence belief states real uncertainty so the machine can wonder instead of
narrating). Three bands, as a starting proposal:

| Band | Treatment |
|------|-----------|
| high | the fragment, as heard |
| middle | admitted **as** uncertain — something that sounded like *"…"* |
| low | discarded, or admitted as "someone spoke, too far off to make out" |

The middle band is where the feature earns its place, and it's exactly the
kind of call that wants your ear on real output rather than my judgement.

## Staging

Deliberately ordered so everything before the prompt is provable in isolation,
and nothing reaches the voice until you've heard what it does.

- **Stage 0 — does it hear at all?** `debug/test_hearing.py` (BUILT, Sep 12) —
  standalone, imports nothing from the project, runs with the machine up or
  down. Room mic via `--local`, phone via `--phone` over Tailscale HTTPS,
  a WAV file via `--wav`, and `--fake` to exercise capture/segmentation/
  transport with no model loaded. This answers the go/no-go question before
  any of it is wired anywhere.
- **Stage 1 — the local mic.** Same pipeline, `arecord`/`sounddevice` source
  instead of HTTP. Confirms the real acoustic case, which is the go/no-go.
- **Stage 2 — the fact reaches the machine.** `POST /machine/hearing`,
  `utils/hearing.py` deque, `/state` exposure so the dashboard shows what's
  being heard. Still no prompt injection — you can watch it hear without it
  affecting a single caption.
- **Stage 3 — the voice.** The fact line, dosed, behind `HEARING_IN_PROMPT`,
  default off. A/B revertable like `FELT_FRAME_ENABLED`. Only after the
  rulings above.

Stages 0-2 are safe to build now and cannot affect a running exhibition.
Stage 3 is a voice change and shouldn't be written until §"the part that needs
your ruling" is settled.

## Config (proposed)

```python
HEARING_ENABLED = False          # master gate; False = no capture, no model load
HEARING_SOURCE = "http"          # "http" (phone) | "local" (studio mic)
HEARING_MODEL = "base"           # faster-whisper size; tiny|base|small
HEARING_DEVICE = "cpu"           # never "cuda" while the VLM is resident
HEARING_VAD_AGGRESSIVENESS = 2   # webrtcvad 0-3
HEARING_MAX_UTTERANCE_S = 12.0   # hard cap per fragment
HEARING_WINDOW = 4               # fragments kept in the rolling deque
HEARING_FRAGMENT_TTL_S = 60.0    # older than this is never said
HEARING_CONF_HIGH = -0.5         # avg_logprob bands
HEARING_CONF_LOW = -1.1
HEARING_TAPE = False             # debug transcript to disk; OFF in exhibition
HEARING_IN_PROMPT = False        # Stage 3 gate
```

## Tests (`debug/`, per CLAUDE.md)

- `test_hearing_vad.py` — VAD gate against a WAV, no model, no mic.
- `test_hearing_transcribe.py` — fixture WAV → fragment + confidence band.
- `test_hearing_endpoint.py` — POST a WAV at the sidecar, assert the fragment
  arrives; no hardware.
- `test_hearing_line.py` — deque → fact line, all three bands, dosing and TTL.

The first three are runnable without a mic, which matters while this is being
built remotely.

## Open questions for the artist

1. Framing of heard speech in the prompt — perceived sound, never address.
   Wording is yours.
2. May heard speech form its own memory class, or is it strictly ephemeral?
3. The middle confidence band — admit as uncertain, or discard?
4. Should the machine know *that* someone spoke even when the words are lost?
   (This is cheap, has no transcription risk, and may be most of the value.)
