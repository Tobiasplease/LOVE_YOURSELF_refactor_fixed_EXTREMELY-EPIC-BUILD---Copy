"""Offline check of world-anchored change detection (Sep 3, queue #2).

1. PoseViewMemory: baseline / unchanged / changed / off_center / stale
   rebaseline / eviction — the honesty rules, on synthetic 64px views.
2. Registry anchor verification: a re-sighting near the anchor stamps
   last_verified_ts; a far sighting moves the EMA but verifies nothing;
   verified_recently_matching does the substring match + window.
3. The familiarity line: "still in the same spot" only when verified,
   softer line otherwise.
4. The unchanged clock: a world_changed event resets it.
5. Boredom blend: verified stillness raises the scalar, capped below the
   bored threshold; unverified stillness (too few confirms) does not.
6. scene_motion provenance: invalid results carry a reason; saccades are
   named as such.

Run: python debug/test_world_anchor.py  (no server, no camera needed)
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FAIL = 0


def check(name, cond, detail=""):
    global FAIL
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        FAIL += 1


def _view(seed):
    rng = np.random.RandomState(seed)
    return rng.randint(0, 255, (64, 64), dtype=np.uint8)


def test_pose_view_memory():
    print("\n[1] PoseViewMemory honesty rules")
    from vision.pose_view_memory import PoseViewMemory

    import config.config as cfg

    m = PoseViewMemory()
    now = time.time()
    a, b = _view(1), _view(2)

    check("first look baselines", m.observe(a, 90.0, 90.0, now)["status"] == "baselined")
    r = m.observe(a, 90.5, 90.2, now + 10)
    check("same view unchanged", r["status"] == "unchanged" and r["score"] < 0.05, str(r))
    check("away_s carried", abs(r["away_s"] - 10) < 1)
    r = m.observe(b, 90.0, 90.0, now + 20)
    check("swapped view changed", r["status"] == "changed" and r["score"] > cfg.WORLD_VIEW_DIFF_THRESHOLD, str(r))
    check("change re-baselines (repeat is unchanged)", m.observe(b, 90.0, 90.0, now + 30)["status"] == "unchanged")

    # off-center: reference sits at one edge of its 6-degree cell, the new
    # pose at the other — same cell, but past the 3-degree compare tolerance
    m3 = PoseViewMemory()
    m3.observe(a, 88.0, 90.0, now)
    check("off-center pose keeps the reference", m3.observe(b, 92.5, 90.0, now + 10)["status"] == "off_center")
    check("reference survived off-center look", m3.observe(a, 88.0, 90.0, now + 20)["status"] == "unchanged")

    r = m.observe(a, 90.0, 90.0, now + 60 + cfg.WORLD_POSE_REF_MAX_AGE_S)
    check("stale reference re-baselines silently", r["status"] == "rebaselined")

    check("unknown pose learns nothing", m.observe(a, None, None, now)["status"] == "no_pose")

    m2 = PoseViewMemory()
    for i in range(cfg.WORLD_POSE_MAX_REFS + 5):
        m2.observe(_view(i), 45.0 + i * cfg.WORLD_POSE_CELL_DEG, 90.0, now + i)
    check("store capped", len(m2._refs) <= cfg.WORLD_POSE_MAX_REFS, str(len(m2._refs)))


def test_registry_verification():
    print("\n[2] registry anchor verification")
    import tempfile

    from perception.spatial_registry import SpatialRegistry

    reg = SpatialRegistry(state_path=os.path.join(tempfile.mkdtemp(), "reg.json"))
    shape = (720, 1280, 3)
    det = {"term": "pink shelf", "box": (600, 320, 680, 400), "conf": 0.6, "settled": True, "pan": 90.0, "tilt": 90.0}

    reg.update_from_detections([det], shape)
    check("first sighting has no verification", "last_verified_ts" not in reg.entries["pink shelf"])
    reg.update_from_detections([det], shape)
    check("re-sighting at anchor verifies", reg.entries["pink shelf"].get("last_verified_ts", 0) > 0)

    reg2 = SpatialRegistry(state_path=os.path.join(tempfile.mkdtemp(), "reg.json"))
    reg2.update_from_detections([det], shape)
    far = dict(det, pan=130.0)
    reg2.update_from_detections([far], shape)
    e = reg2.entries["pink shelf"]
    check("far sighting moves the EMA", e["pan"] > 91.0, str(e["pan"]))
    check("far sighting verifies nothing", "last_verified_ts" not in e)

    check("matching: term inside concept label", reg.verified_recently_matching("that pink shelf by the wall", 60))
    check("matching: window expiry", not reg.verified_recently_matching("pink shelf", 0))
    check("matching: no match", not reg.verified_recently_matching("black curtain", 60))


def test_familiarity_gate():
    print("\n[3] familiarity line requires verification")
    import types

    import perception.spatial_registry as sr_mod
    from captioner.prompts import get_familiarity_line

    agent = types.SimpleNamespace(
        _last_matched_concepts=[{"id": "c1", "label": "pink shelf", "times_seen": 12, "session_count": 3, "last_seen": time.time()}],
        _familiarity_counter=2,  # next call hits the every-3rd dose
        _recent_familiarity_ids=[],
    )
    saved = sr_mod.spatial_registry.verified_recently_matching
    try:
        sr_mod.spatial_registry.verified_recently_matching = lambda label, w: True
        line = get_familiarity_line(agent)
        check("verified -> same-spot line", "still in the same spot" in line, line)

        agent._familiarity_counter = 2
        agent._recent_familiarity_ids = []
        sr_mod.spatial_registry.verified_recently_matching = lambda label, w: False
        line = get_familiarity_line(agent)
        check("unverified -> softer line", "a few times now" in line, line)
    finally:
        sr_mod.spatial_registry.verified_recently_matching = saved


def test_unchanged_clock():
    print("\n[4] world_changed resets the unchanged clock")
    import types

    import utils.episodic_log as el_mod
    from captioner.prompts import unchanged_duration_s

    now = time.time()
    agent = types.SimpleNamespace(true_session_start=now - 7200, _last_new_concept_ts=0)
    saved = el_mod.episodic_log.get_last_event
    try:
        el_mod.episodic_log.get_last_event = lambda etype: {"timestamp": now - 120} if etype == "world_changed" else None
        d = unchanged_duration_s(agent, now)
        check("clock anchored to world change", abs(d - 120) < 1, str(d))
        el_mod.episodic_log.get_last_event = lambda etype: None
        d = unchanged_duration_s(agent, now)
        check("no events -> session floor", abs(d - 7200) < 1, str(d))
    finally:
        el_mod.episodic_log.get_last_event = saved


def test_boredom_blend():
    print("\n[5] verified stillness feeds boredom, capped")
    from captioner.captioner import Captioner

    import config.config as cfg

    c = Captioner.__new__(Captioner)
    now = time.time()
    c._boredom = 0.2
    c.true_session_start = now - 7200
    c._world_change_ts = now - 7200
    c._last_salience_time = now - 7200

    c._world_confirms = 0
    check("too few confirms -> linguistic scalar only", abs(c.boredom - 0.2) < 1e-6, str(c.boredom))

    c._world_confirms = cfg.WORLD_STILL_MIN_CONFIRMS
    check("verified stillness raises boredom to the cap", abs(c.boredom - cfg.WORLD_STILLNESS_BOREDOM_MAX) < 0.01, str(c.boredom))
    check("cap stays below the bored threshold", cfg.WORLD_STILLNESS_BOREDOM_MAX < 0.7)

    c._last_salience_time = now - 60  # a live moment just happened
    check("salience resets the stillness component", c.boredom < 0.25, str(c.boredom))

    c._boredom = 0.9
    check("linguistic scalar still wins when higher", abs(c.boredom - 0.9) < 1e-6, str(c.boredom))


def test_flow_provenance():
    print("\n[6] scene_motion invalidity carries a reason")
    from vision.scene_motion import SceneMotionEstimator

    est = SceneMotionEstimator()
    rng = np.random.RandomState(7)
    tex = rng.randint(0, 255, (240, 320), dtype=np.uint8)

    r = est.update(tex)
    check("first frame named", not r["valid"] and r["reason"] == "first_frame", str(r["reason"]))
    r = est.update(np.roll(tex, 40, axis=1))
    check("big shift named saccade", not r["valid"] and r["reason"] == "saccade", str(r))
    est2 = SceneMotionEstimator()
    est2.update(tex)
    r = est2.update(np.roll(tex, 3, axis=1))
    check("small shift measured", r["valid"] and r["reason"] is None, str(r))


def test_motion_line_attestation():
    print("\n[7] motion lines claim only what was measured")
    import inspect

    from captioner.captioner import Captioner

    src = inspect.getsource(Captioner._process_frame)
    check("stillness claims gated on valid flow frames", "_room_measured_still" in src and "flow_valid_frames" in src)
    check("bare sweep line exists (no room claim)", '" The view changed because you were looking around."' in src)
    check("unmeasurable windows claim nothing", 'motion_line = ""' in src)
    check("'Someone' requires person signals", "scene_motion and person_present_in_window" in src)
    src2 = inspect.getsource(Captioner._assess_scene)
    check("_assess_scene exports flow_valid_frames", '"flow_valid_frames"' in src2)


if __name__ == "__main__":
    test_pose_view_memory()
    test_registry_verification()
    test_familiarity_gate()
    test_unchanged_clock()
    test_boredom_blend()
    test_flow_provenance()
    test_motion_line_attestation()
    print(f"\n{'ALL PASS' if FAIL == 0 else f'{FAIL} FAILURES'}")
    sys.exit(1 if FAIL else 0)
