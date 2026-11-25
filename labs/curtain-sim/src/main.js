import { Curtain } from './cloth.js';
import { Bridle, Motor } from './rig.js';
import { WindField, gustEnvelope, makeGusts } from './wind.js';
import { ROOM, createStage } from './stage.js';
import { CurtainView, RigView } from './views.js';
import { Scope, createPanel } from './ui.js';

const RIG_X = -1.25;
const REF_X = 1.25;
const CURTAIN_Z = 0.3;
const FRAME_DT = 1 / 60;
const SUBSTEPS = 8;
const LIGHT_BRIGHTNESS = 1.5;

const params = {
  auto: true,
  windStrength: 0.7,
  bellyPull: 0.6,
  hemPull: 1.0,
  hemDelay: 0.4,
  bellyPulleyDist: 3.0,
  hemPulleyDist: 1.0,
  legReach: 0.35,
  rods: true,
  elasticK: 40,
  motorSpeed: 0.8,
  kelvin: 5200,
  paused: false,
};

const spec = [
  { heading: 'Cue' },
  { key: 'auto', type: 'check', label: 'Auto gusts (rig follows the same gusts as the real wind)' },
  { key: 'paused', type: 'check', label: 'Pause' },
  { key: 'windStrength', label: 'Real wind strength', min: 0, max: 2, unit: '×' },
  { heading: 'Rig — one column' },
  { key: 'bellyPull', label: 'Belly line pull', min: 0, max: 1.2, unit: 'm' },
  { key: 'hemPull', label: 'Hem line pull', min: 0, max: 1.6, unit: 'm' },
  { key: 'hemDelay', label: 'Hem follows belly after', min: 0, max: 1.5, unit: 's' },
  { key: 'bellyPulleyDist', label: 'Belly pulley P1 out from curtain', min: 0.4, max: 4.5, unit: 'm' },
  { key: 'hemPulleyDist', label: 'Hem pulley P2 out from curtain', min: 0.2, max: 3.5, unit: 'm' },
  { key: 'elasticK', label: 'Elastic stiffness', min: 5, max: 300, step: 1, digits: 0, unit: 'N/m' },
  { key: 'motorSpeed', label: 'Motor max line speed', min: 0.1, max: 2, unit: 'm/s' },
  { key: 'legReach', label: 'Bridle leg reach (floating pulley to fabric, resets)', min: 0.08, max: 0.8, unit: 'm', rebuild: true },
  { key: 'rods', type: 'check', label: 'Spreader rods on belly legs (off = tied to points, resets)', rebuild: true },
  { heading: 'Light' },
  { key: 'kelvin', label: 'Lightbox colour temperature', min: 2700, max: 6500, step: 50, digits: 0, unit: 'K' },
];

const canvas = document.getElementById('view');
const stage = createStage(canvas, [RIG_X, REF_X]);
const gusts = makeGusts(7);
const scope = new Scope(document.getElementById('scope'), 12, [
  { color: '#f2efe6', max: 1 },
  { color: '#f0b45a', max: 1.6 },
  { color: '#7fb4e8', max: 1.6 },
]);
const readout = document.getElementById('readout');

let world = null;
let t = 0;

const anchorAt = (distance) => [RIG_X, ROOM.ceiling - 0.03, CURTAIN_Z + distance];

function build() {
  if (world) for (const v of world.views) v.dispose(stage.scene);

  const rig = new Curtain({ x: RIG_X, z: CURTAIN_Z });
  const ref = new Curtain({ x: REF_X, z: CURTAIN_Z });

  const centre = Math.floor(rig.nx / 2);
  const bellyRow = Math.round(rig.ny * 0.58);
  const hemRow = rig.ny - 1;
  const bellyCols = [centre - Math.round(0.3 / rig.dx), centre + Math.round(0.3 / rig.dx)];
  const hemCols = [centre - Math.round(0.35 / rig.dx), centre + Math.round(0.35 / rig.dx)];

  const rods = params.rods ? bellyCols.map((col) => rig.addRod(bellyRow, col, 0.2, 0.0015)) : [];

  const bellyMotor = new Motor({ maxSpeed: params.motorSpeed });
  const hemMotor = new Motor({ maxSpeed: params.motorSpeed });
  const belly = new Bridle(rig, {
    attach: bellyCols.map((c) => rig.index(c, bellyRow)),
    anchor: anchorAt(params.bellyPulleyDist),
    motor: bellyMotor,
    elasticK: params.elasticK,
    reach: params.legReach,
  });
  const hem = new Bridle(rig, {
    attach: hemCols.map((c) => rig.index(c, hemRow)),
    anchor: anchorAt(params.hemPulleyDist),
    motor: hemMotor,
    elasticK: params.elasticK,
    reach: params.legReach,
  });

  const wind = new WindField({ gusts, xRef: REF_X - 0.7, yTop: rig.o.railY });
  const bellyY = rig.pos[rig.index(centre, bellyRow) * 3 + 1];

  for (let i = 0; i < 90; i++) {
    wind.fill(ref, 0);
    rig.step(FRAME_DT, SUBSTEPS);
    ref.step(FRAME_DT, SUBSTEPS);
  }

  const views = [new CurtainView(stage.scene, rig, RIG_X), new CurtainView(stage.scene, ref, REF_X), new RigView(stage.scene, rig, [belly, hem], rods)];
  world = { rig, ref, belly, hem, bellyMotor, hemMotor, wind, views, bellyDelay: wind.arrivalDelay(REF_X, bellyY) };
  t = 0;
  scope.clear();
  applyLight();
}

function applyLight() {
  const color = stage.setKelvin(params.kelvin, LIGHT_BRIGHTNESS);
  world.views[0].setLight(color, 1.25);
  world.views[1].setLight(color, 1.25);
}

function onChange(item) {
  if (item.rebuild) return build();
  if (item.key === 'kelvin') return applyLight();
  world.belly.setAnchor(anchorAt(params.bellyPulleyDist));
  world.hem.setAnchor(anchorAt(params.hemPulleyDist));
  world.belly.elasticK = world.hem.elasticK = params.elasticK;
  world.bellyMotor.maxSpeed = world.hemMotor.maxSpeed = params.motorSpeed;
}

function simulate() {
  const w = world;
  const gustNow = gustEnvelope(gusts, t - w.bellyDelay);
  if (params.auto) {
    w.bellyMotor.target = params.bellyPull * gustNow;
    w.hemMotor.target = params.hemPull * gustEnvelope(gusts, t - w.bellyDelay - params.hemDelay);
  } else {
    w.bellyMotor.target = params.bellyPull;
    w.hemMotor.target = params.hemPull;
  }
  w.bellyMotor.update(FRAME_DT);
  w.hemMotor.update(FRAME_DT);

  w.wind.strength = params.windStrength;
  w.wind.fill(w.ref, t);
  w.rig.step(FRAME_DT, SUBSTEPS);
  w.ref.step(FRAME_DT, SUBSTEPS);
  t += FRAME_DT;
  scope.push(t, [gustNow, w.bellyMotor.pos, w.hemMotor.pos]);
}

function grams(newtons) {
  return `${Math.round((newtons / 9.81) * 1000)} g`;
}

function frame() {
  if (!params.paused) simulate();
  for (const v of world.views) v.update();
  stage.renderer.render(stage.scene, stage.camera);
  scope.draw();
  readout.innerHTML =
    `t ${t.toFixed(1)} s<br>` +
    `belly line: ${(world.bellyMotor.pos * 100).toFixed(0)} cm reeled, tension ${grams(world.belly.tension)}<br>` +
    `hem line: ${(world.hemMotor.pos * 100).toFixed(0)} cm reeled, tension ${grams(world.hem.tension)}`;
  requestAnimationFrame(frame);
}

createPanel(document.getElementById('controls'), params, spec, onChange);
for (const button of document.querySelectorAll('[data-view]')) button.addEventListener('click', () => stage.setView(button.dataset.view));
build();
window.curtainSim = { params, get world() { return world; }, get t() { return t; }, simulate, build };
requestAnimationFrame(frame);
