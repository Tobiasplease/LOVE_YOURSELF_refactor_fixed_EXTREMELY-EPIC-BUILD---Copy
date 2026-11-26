import { Curtain } from './cloth.js';
import { WindField, makeGusts } from './wind.js';
import { ROOM, createStage } from './stage.js';
import { CurtainView, RigView } from './views.js';
import { buildGroup, buildMatrix } from './ui.js';
import { buildRig } from './layout.js';
import { CueEngine } from './cue.js';
import { Tape } from './tape.js';
import { SpreadMeter } from './metrics.js';

const FABRICS = {
  voile: { fabricGsm: 50, stiffDown: 1, stiffAcross: 1 },
  organza: { fabricGsm: 35, stiffDown: 10, stiffAcross: 10 },
  taffeta: { fabricGsm: 90, stiffDown: 20, stiffAcross: 20 },
  ribbed: { fabricGsm: 55, stiffDown: 10, stiffAcross: 1 },
};

const RIG_X = -1.25;
const REF_X = 1.25;
const CURTAIN_Z = 0.3;
const BASE_DT = 1 / 60;
const LIGHT = 1.5;
const narrow = matchMedia('(max-width: 700px)').matches;

const DEFAULTS = {
  speed: 1,
  mode: 'gusts',
  tempo: 1,
  direction: 0,
  waveSpeed: 2.2,
  invert: false,
  period: 5,
  master: 0,
  hemLag: 0.35,
  topPull: 0.25,
  bellyPull: 0.5,
  hemPull: 0.8,
  topDist: 3.6,
  bellyDist: 2.6,
  hemDist: 1.0,
  pulleySide: 0,
  columns: 5,
  rows: 3,
  legReach: 0.3,
  rods: true,
  elasticK: 40,
  motorSpeed: 0.8,
  motorAccel: 3,
  shaper: 0,
  preload: 0,
  fabric: 'voile',
  fabricGsm: 50,
  stiffDown: 1,
  stiffAcross: 1,
  patchStiff: 1,
  patchRadius: 0.12,
  windStrength: 0.7,
  show: narrow ? 'rig' : 'both',
  quality: narrow ? 'phone' : 'fine',
  kelvin: 5200,
  view: 'angle',
};

function start(data) {
  const params = { ...DEFAULTS, ...(data?.params ?? {}) };
  window.claude?.hot?.snapshot?.(() => ({ params }));

  const canvas = document.getElementById('view');
  const stage = createStage(canvas, [RIG_X, REF_X]);
  const gusts = makeGusts(7);
  const cue = new CueEngine(gusts);
  const tape = new Tape(20, 30);
  const $ = (id) => document.getElementById(id);

  let world = null;
  let matrix = null;
  let transport = 'play';
  let cursor = null;
  let nextRecord = 0;
  let physicsMs = 0;

  function build() {
    if (world) for (const v of world.views) v.dispose(stage.scene);
    const spacing = params.quality === 'phone' ? 0.055 : 0.04;
    const fabric = { fabricGsm: params.fabricGsm, stiffDown: params.stiffDown, stiffAcross: params.stiffAcross };
    const rig = new Curtain({ x: RIG_X, z: CURTAIN_Z, spacing, ...fabric });
    const ref = params.show === 'both' ? new Curtain({ x: REF_X, z: CURTAIN_Z, spacing, ...fabric }) : null;
    const { lines, rods, kinds } = buildRig(rig, params, ROOM.ceiling);
    const patchCentres = lines.flatMap((l) => l.bridle.attach).map((k) => rig.fabricCoords(k));
    rig.stiffenZones(patchCentres, params.patchRadius, params.patchStiff);
    const wind = new WindField({ gusts, xRef: REF_X - 0.7, yTop: rig.o.railY });
    const curtains = ref ? [rig, ref] : [rig];

    for (let i = 0; i < 60; i++) {
      if (ref) wind.fill(ref, 0);
      for (const c of curtains) c.step(BASE_DT, 8);
    }

    const views = [new CurtainView(stage.scene, rig, RIG_X), new RigView(stage.scene, rig, lines, rods)];
    if (ref) views.push(new CurtainView(stage.scene, ref, REF_X));
    world = { rig, ref, curtains, lines, rods, kinds, wind, views, t: 0, cueT: 0, meter: new SpreadMeter(rig, lines) };

    matrix = buildMatrix($('matrix'), lines, kinds, params.columns);
    tape.clear();
    cue.reset();
    cursor = null;
    nextRecord = 0;
    applyLive();
    applyLight();
    stage.setFocus(ref ? 0 : RIG_X, ref ? 1 : 0.75);
    stage.setView(params.view);
    $('hudsub').textContent = `${lines.length} lines · ${lines.length * 2} attachment points · ${params.columns} columns × ${kinds.length} rows`;
  }

  function applyLive() {
    for (const l of world.lines) {
      l.bridle.setAnchor([l.railX + params.pulleySide, ROOM.ceiling - 0.03, CURTAIN_Z + params[`${l.kind}Dist`]]);
      l.bridle.elasticK = params.elasticK;
      l.motor.maxSpeed = params.motorSpeed;
      l.motor.maxAccel = params.motorAccel;
    }
    world.wind.strength = params.windStrength;
    world.wind.setWave(params.direction, params.waveSpeed);
  }

  function applyLight() {
    const color = stage.setKelvin(params.kelvin, LIGHT);
    for (const v of world.views) v.setLight?.(color, 1.25);
  }

  function advance(dt) {
    const w = world;
    const started = performance.now();
    w.cueT += dt * params.tempo;
    if (params.mode === 'manual') cue.logMaster(w.cueT, params.master);
    cue.apply(w.lines, params, w.cueT, w.rig.o.railWidth, w.rig.o.height);
    for (const l of w.lines) l.motor.update(dt);
    if (w.ref) w.wind.fill(w.ref, w.cueT, params.invert && params.mode === 'gusts');
    const substeps = Math.max(2, Math.ceil(8 * dt * 60));
    for (const c of w.curtains) c.step(dt, substeps);
    w.t += dt;
    w.meter.measure();
    physicsMs = physicsMs * 0.9 + (performance.now() - started) * 0.1;
    if (w.t >= nextRecord) {
      tape.record(w);
      nextRecord = w.t + 1 / tape.rate - 1e-6;
    }
  }

  function setTransport(state) {
    if (state === 'play' && cursor !== null) {
      tape.resume(Math.round(cursor), world);
      nextRecord = world.t + 1 / tape.rate;
      cursor = null;
    }
    transport = state;
    for (const [id, s] of [['tPlay', 'play'], ['tPause', 'pause'], ['tRev', 'reverse']]) $(id).classList.toggle('on', transport === s);
    for (const b of document.querySelectorAll('.qPlay')) b.textContent = transport === 'play' ? '❚❚ PAUSE' : '▶ PLAY';
  }

  function tick() {
    if (transport === 'play') advance(BASE_DT * params.speed);
    else if (transport === 'reverse' && tape.length) {
      if (cursor === null) cursor = tape.length - 1;
      cursor -= (params.speed * tape.rate) / 60;
      if (cursor <= 0) {
        cursor = 0;
        setTransport('pause');
      }
      tape.show(Math.round(cursor), world);
    }
  }

  let frames = 0;
  let fpsClock = performance.now();
  function frame() {
    tick();
    for (const v of world.views) v.update();
    stage.renderer.render(stage.scene, stage.camera);
    matrix.update(Math.max(params.topPull, params.bellyPull, params.hemPull));

    frames++;
    const now = performance.now();
    if (now - fpsClock > 500) {
      $('stFps').textContent = Math.round((frames * 1000) / (now - fpsClock));
      $('stMs').textContent = physicsMs.toFixed(0);
      frames = 0;
      fpsClock = now;
    }
    const span = tape.length ? (tape.length - 1) / tape.rate : 0;
    const at = cursor === null ? span : cursor / tape.rate;
    $('scrub').max = Math.max(0, tape.length - 1);
    if (document.activeElement !== $('scrub')) $('scrub').value = cursor === null ? tape.length - 1 : cursor;
    $('tapeRead').textContent = cursor === null ? `LIVE · ${world.t.toFixed(1)} s` : `REVIEW −${(span - at).toFixed(1)} s`;
    $('stTape').textContent = cursor === null ? 'live' : 'review';
    const limit = world.lines.filter((l) => l.motor.saturated).length;
    const peak = Math.max(...matrix.peaks());
    const tent = world.meter.smoothed;
    $('matrixFoot').textContent = `peak ${Math.round(peak)} g · ${limit}/${world.lines.length} at speed limit · tent ${tent === null ? '—' : (tent * 100).toFixed(1)} cm`;
    requestAnimationFrame(frame);
  }

  const refreshers = [];
  const refreshAll = () => refreshers.forEach((r) => r());
  const onChange = (item) => {
    if (item.key === 'fabric') Object.assign(params, FABRICS[params.fabric]);
    if (item.rebuild) build();
    else if (item.key === 'kelvin') applyLight();
    else if (item.key === 'view') stage.setView(params.view);
    else applyLive();
    refreshAll();
  };

  const panel = $('groups');
  refreshers.push(
    buildGroup(
      panel,
      'CUE — WHAT THE MOTORS PLAY',
      [
        { key: 'mode', type: 'choice', label: 'SOURCE', options: [['gusts', 'GUSTS'], ['swell', 'SWELL'], ['manual', 'FADER']] },
        { key: 'master', label: 'FADER (in FADER mode, ripples out)', min: 0, max: 1 },
        { key: 'tempo', label: 'TEMPO', min: 0.25, max: 3, unit: '×' },
        { key: 'period', label: 'SWELL PERIOD', min: 1.5, max: 12, step: 0.1, digits: 1, unit: ' s' },
        { key: 'invert', type: 'toggle', label: 'POLARITY ▸ LIFTED, WAVES DROP' },
      ],
      params,
      onChange,
    ),
    buildGroup(
      panel,
      'WAVE — HOW IT TRAVELS',
      [
        {
          key: 'direction',
          type: 'choice',
          label: 'FROM',
          options: [[0, 'LEFT'], [180, 'RIGHT'], [90, 'TOP'], [270, 'HEM'], [35, 'DIAG']],
        },
        { key: 'direction', label: 'DIRECTION', min: 0, max: 359, step: 1, digits: 0, unit: '°' },
        { type: 'action', label: '⇄ FLIP DIRECTION', run: () => (params.direction = (params.direction + 180) % 360) },
        { key: 'waveSpeed', label: 'WAVE SPEED ACROSS RIG', min: 0.3, max: 8, step: 0.05, unit: ' m/s' },
        { key: 'hemLag', label: 'HEM TRAILS BY', min: 0, max: 1.5, unit: ' s' },
      ],
      params,
      onChange,
    ),
    buildGroup(
      panel,
      'ROWS — PULL AND PULLEY DISTANCE',
      [
        { key: 'topPull', label: 'TOP PULL', min: 0, max: 1.2, unit: ' m' },
        { key: 'bellyPull', label: 'BELLY PULL', min: 0, max: 1.2, unit: ' m' },
        { key: 'hemPull', label: 'HEM PULL', min: 0, max: 1.6, unit: ' m' },
        { key: 'topDist', label: 'TOP PULLEYS OUT', min: 0.5, max: 4.5, unit: ' m' },
        { key: 'bellyDist', label: 'BELLY PULLEYS OUT', min: 0.4, max: 4.5, unit: ' m' },
        { key: 'hemDist', label: 'HEM PULLEYS OUT', min: 0.2, max: 3.5, unit: ' m' },
        { key: 'pulleySide', label: 'ALL PULLEYS SHIFTED SIDEWAYS', min: -1.5, max: 1.5, unit: ' m' },
      ],
      params,
      onChange,
    ),
    buildGroup(
      panel,
      'RIG BUILD',
      [
        { key: 'columns', label: 'COLUMNS', min: 3, max: 8, step: 1, digits: 0, rebuild: true },
        { key: 'rows', type: 'choice', label: 'ROWS', options: [[2, 'BELLY+HEM'], [3, 'TOP+BELLY+HEM']], rebuild: true },
        { key: 'legReach', label: 'BRIDLE LEG REACH', min: 0.08, max: 0.6, unit: ' m', rebuild: true },
        { key: 'rods', type: 'toggle', label: 'SPREADER RODS', rebuild: true },
        { key: 'elasticK', label: 'ELASTIC', min: 5, max: 300, step: 1, digits: 0, unit: ' N/m' },
        { key: 'motorSpeed', label: 'MOTOR MAX LINE SPEED', min: 0.1, max: 2.5, unit: ' m/s' },
      ],
      params,
      onChange,
    ),
    buildGroup(
      panel,
      'FABRIC — STIFFNESS AND WEIGHT',
      [
        { key: 'fabric', type: 'choice', label: 'START FROM', options: [['voile', 'VOILE'], ['organza', 'ORGANZA'], ['taffeta', 'TAFFETA'], ['ribbed', 'RIBBED']], rebuild: true },
        { key: 'fabricGsm', label: 'WEIGHT', min: 20, max: 150, step: 1, digits: 0, unit: ' g/m²', rebuild: true },
        { key: 'stiffDown', label: 'STIFFNESS DOWN THE DROP', min: 1, max: 20, step: 0.5, digits: 1, unit: '×', rebuild: true },
        { key: 'stiffAcross', label: 'STIFFNESS ACROSS THE WIDTH', min: 1, max: 20, step: 0.5, digits: 1, unit: '×', rebuild: true },
        { key: 'patchStiff', label: 'STIFFENER AROUND ATTACHMENTS', min: 1, max: 20, step: 0.5, digits: 1, unit: '×', rebuild: true },
        { key: 'patchRadius', label: 'STIFFENER PATCH RADIUS', min: 0.05, max: 0.25, unit: ' m', rebuild: true },
      ],
      params,
      onChange,
    ),
    buildGroup(
      panel,
      'MOTION — STOPPING ON A DIME',
      [
        { key: 'motorAccel', label: 'MOTOR BRAKING / ACCEL', min: 0.3, max: 6, step: 0.1, digits: 1, unit: ' m/s²' },
        { key: 'shaper', label: 'SWAY CANCEL (0 = off, set to sway rhythm)', min: 0, max: 3.5, step: 0.05, unit: ' s' },
        { key: 'preload', label: 'PRELOAD ON ALL LINES', min: 0, max: 0.15, unit: ' m' },
      ],
      params,
      onChange,
    ),
    buildGroup(
      panel,
      'ROOM',
      [
        { key: 'show', type: 'choice', label: 'SHOW', options: [['rig', 'RIG'], ['both', 'RIG + REAL WIND']], rebuild: true },
        { key: 'view', type: 'choice', label: 'VIEW', options: [['front', 'FRONT'], ['angle', 'ANGLE'], ['side', 'SIDE']] },
        { key: 'quality', type: 'choice', label: 'CLOTH', options: [['phone', 'LIGHT'], ['fine', 'FINE']], rebuild: true },
        { key: 'windStrength', label: 'REAL WIND STRENGTH', min: 0, max: 2, unit: '×' },
        { key: 'kelvin', label: 'LIGHTBOX', min: 2700, max: 6500, step: 50, digits: 0, unit: ' K' },
      ],
      params,
      onChange,
    ),
  );

  refreshers.push(
    buildGroup($('speedBox'), 'TRANSPORT SPEED', [{ key: 'speed', type: 'choice', label: '', options: [[0.1, '0.1×'], [0.25, '¼×'], [0.5, '½×'], [1, '1×'], [2, '2×']] }], params, () => refreshAll()),
  );

  $('tPlay').addEventListener('click', () => setTransport('play'));
  $('tPause').addEventListener('click', () => setTransport('pause'));
  $('tRev').addEventListener('click', () => setTransport('reverse'));
  $('tReset').addEventListener('click', () => {
    build();
    setTransport('play');
  });
  $('tPeak').addEventListener('click', () => matrix.resetPeaks());
  $('scrub').addEventListener('input', () => {
    if (!tape.length) return;
    if (transport !== 'pause') setTransport('pause');
    cursor = Number($('scrub').value);
    tape.show(cursor, world);
  });
  for (const b of document.querySelectorAll('.qPlay')) b.addEventListener('click', () => setTransport(transport === 'play' ? 'pause' : 'play'));
  for (const b of document.querySelectorAll('.qRev')) b.addEventListener('click', () => setTransport('reverse'));
  const toggleClean = () => document.body.classList.toggle('clean');
  $('tbClose').addEventListener('click', toggleClean);
  $('qPanel').addEventListener('click', toggleClean);
  window.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT') return;
    if (e.code === 'Space') {
      e.preventDefault();
      setTransport(transport === 'play' ? 'pause' : 'play');
    } else if (e.code === 'ArrowLeft') setTransport('reverse');
    else if (e.code === 'KeyH') toggleClean();
  });

  build();
  setTransport('play');
  window.curtainSim = {
    params,
    get world() {
      return world;
    },
    get t() {
      return world.t;
    },
    simulate: () => advance(BASE_DT * params.speed),
    build,
    setTransport,
    refreshAll,
    setView: (v) => {
      params.view = v;
      stage.setView(v);
      refreshAll();
    },
  };
  document.getElementById('boot')?.remove();
  requestAnimationFrame(frame);
}

if (window.claude?.hot?.ready) window.claude.hot.ready(start);
else start(window.claude?.hot?.data ?? {});
