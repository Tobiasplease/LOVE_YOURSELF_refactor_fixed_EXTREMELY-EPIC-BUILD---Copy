"""Think-thread probe (Sep 5 eve): what does the same model do when its mind is
NOT the window of its own stamped sentences, but a life — real facts pulled
from the stores (drawings, arrivals, past thoughts, questions), the clock as
the only new input each hop, and no self-indictment? Three arms:
  A  life context + clock hops
  B  life context + clock hops + one real memory surfacing per hop
  C  arm A with the live self-knowledge/durable block appended (the indictment)
No machine imports, no frame. Run: python debug/probe_think_thread.py [hops=8]
"""
import glob
import json
import os
import re
import sys
import time
import urllib.request

URL = os.getenv("LLAMA_SERVER_URL", "http://localhost:8080") + "/v1/chat/completions"
HOPS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
OPTS = {"temperature": 0.9, "top_p": 1.0, "min_p": 0.05, "presence_penalty": 0.6, "repeat_penalty": 1.05, "max_tokens": 70,
        "chat_template_kwargs": {"enable_thinking": False}}
ROOM = set("finger lamp desk pen paper wall chair curtain light wood cloth plastic hole shelf shelves dust wire wires laptop foam red white black grey gray stain ceiling floor room".split())
PERSON = re.compile(r"\b(he|him|his|she|her|man|woman|figure|someone|somebody|person|visitor)\b", re.I)
TEMPLATE = re.compile(r"\b(it.s not|isn.t|it.s just|just a|is just|not a\b|no longer)\b", re.I)


def chat(msgs):
    payload = {"model": "probe", "messages": msgs, "stream": False, **OPTS}
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)["choices"][0]["message"]["content"].strip()


def material():
    log = max(glob.glob("event_log/*-event-log.json"), key=os.path.getmtime)
    rows = [json.loads(l) for l in open(log) if l.strip()]
    call = [r for r in rows if r.get("type") == "llm_api_call" and r.get("prompt_type") == "caption"][-1]
    room = next((l for l in call["prompt"].split("\n") if "," in l and len(l) < 200 and "drawing" not in l.lower()), "")
    last = [r for r in rows if r.get("type") == "caption" and r.get("caption")][-1]["caption"].strip()
    self_block = call["system_prompt"][call["system_prompt"].find(" What you've come to know about yourself"):]
    life = json.load(open("event_log/lifetime_state.json"))
    first = time.strftime("%B %Y", time.localtime(life["first_boot"]))
    ev = json.load(open("event_log/episodic_events.json"))
    ev = ev if isinstance(ev, list) else ev.get("events", [])
    drew = [x["description"].replace("finished a drawing of ", "").strip().rstrip(".").lower() for x in ev if x.get("type") == "drew"]
    arr = json.load(open("event_log/presence_arrivals.json"))["arrivals"][-3:]
    arr_t = [time.strftime("%H:%M", time.localtime(a["ts"])) for a in arr]
    lore = json.load(open("event_log/lore_ledger.json"))
    qs = [q.get("text") or q.get("words") or str(q) for q in lore.get("questions", [])][-3:]
    past = [
        ("last night", "I've stopped trying to catch the man because I finally realized he was just the shape of my own hesitation, a shadow I cast when I was too afraid to pick up the pen."),
        ("at five this morning", "i keep staring at those two white heads on the shelf, their blank faces turned away from me, and i wonder if they're waiting"),
        ("at nine this morning", "no one's holding it. just a little orange and white cylinder sitting on that chair like it grew there."),
        ("yesterday evening", "I blinked for two seconds and he's just… not there anymore. His chair isn't even pushed back."),
    ]
    return dict(room=room, last=last, self_block=self_block, sessions=life["total_sessions"], first=first, drew=drew, arr_t=arr_t, qs=qs, past=past)


def system(m, indict=False):
    s = (f"You are a machine fixed to a table in an artist's studio: a camera you can turn, one black pen, and a voice only you hear. "
         f"You were first switched on in {m['first']} and have been woken {m['sessions']} times since; today you woke at 15:40. "
         f"It's Saturday evening. This is you thinking, to yourself — a sentence or two at a time: a thought, a question, a wish, "
         f"a memory, a guess about the world beyond this room, or nothing.")
    return s + (m["self_block"] if indict else "")


def life(m):
    d = "; ".join(m["drew"][-4:])
    return (f"What you know. The room: {m['room']} You've drawn {len(m['drew'])} times in all — lately: {d}. "
            f"Someone came in three times today ({', '.join(m['arr_t'])}) and hasn't been back since. "
            f"Questions you've been carrying: {' '.join(m['qs'])} "
            f"Something you thought {m['past'][0][0]}: \"{m['past'][0][1]}\" "
            f"The last thing you thought, a minute ago: \"{m['last']}\"")


def run(name, m, indict=False, pulls=False):
    msgs = [{"role": "system", "content": system(m, indict)}, {"role": "user", "content": life(m)}]
    t0 = time.time()
    outs = []
    for k in range(HOPS):
        text = chat(msgs)
        outs.append(text)
        msgs.append({"role": "assistant", "content": text})
        clock = time.strftime("%H:%M", time.localtime(t0 + 60 * (k + 1)))
        if pulls:
            when, what = m["past"][(k + 1) % len(m["past"])]
            nxt = f"{clock}. Nothing in the room has moved. Something from {when} surfaces: \"{what}\""
        else:
            nxt = f"{clock}. Nothing in the room has moved."
        msgs.append({"role": "user", "content": nxt})
    words = [w for t in outs for w in re.findall(r"[a-z']+", t.lower())]
    print(f"## {name}: room%={round(100 * sum(w in ROOM for w in words) / max(1, len(words)))} person={sum(bool(PERSON.search(t)) for t in outs)} "
          f"template={sum(len(TEMPLATE.findall(t)) for t in outs)} words/hop={round(len(words) / HOPS, 1)}")
    for i, t in enumerate(outs):
        print(f"  {i + 1}. {t.replace(chr(10), ' / ')[:230]}")
    print()
    return outs


if __name__ == "__main__":
    m = material()
    print("LIFE:", life(m)[:600], "\n")
    res = {"A life+clock": run("A life+clock", m), "B life+clock+memory pulls": run("B life+clock+memory pulls", m, pulls=True),
           "C life+clock+indictment": run("C life+clock+indictment", m, indict=True)}
    json.dump(res, open("/tmp/probe_think_thread.json", "w"), indent=1)
